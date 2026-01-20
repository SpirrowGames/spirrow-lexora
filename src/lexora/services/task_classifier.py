"""Task classifier service for recommending appropriate models."""

import json
import re
from dataclasses import dataclass, field
from typing import Any

from lexora.backends.base import Backend, BackendError
from lexora.config import ClassifierSettings
from lexora.services.model_registry import ModelRegistry
from lexora.services.router import BackendRouter
from lexora.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AlternativeModel:
    """Alternative model recommendation."""

    model: str
    score: float


@dataclass
class ClassificationResult:
    """Result of task classification."""

    recommended_model: str
    task_type: str
    confidence: float
    reasoning: str
    alternatives: list[AlternativeModel] = field(default_factory=list)


class TaskClassifierError(Exception):
    """Task classifier error."""

    pass


class TaskClassifierDisabledError(TaskClassifierError):
    """Raised when classifier is disabled."""

    pass


CLASSIFICATION_PROMPT_TEMPLATE = """あなたはタスク分類アシスタントです。
以下のタスクを最も適切なカテゴリに分類してください。

利用可能なカテゴリ: {capabilities}

タスク: {task_description}

以下のJSON形式で回答してください（JSON以外のテキストは含めないでください）:
{{"task_type": "カテゴリ名", "confidence": 0.0から1.0の数値, "reasoning": "理由"}}"""


class TaskClassifier:
    """Classifies tasks and recommends appropriate models.

    Uses an LLM to analyze task descriptions and classify them into
    capability categories, then recommends the most suitable model.

    Args:
        model_registry: Registry containing model and capability information.
        backend_router: Router for accessing backends.
        classifier_settings: Settings for the classifier.
    """

    def __init__(
        self,
        model_registry: ModelRegistry,
        backend_router: BackendRouter,
        classifier_settings: ClassifierSettings,
    ) -> None:
        """Initialize the task classifier.

        Args:
            model_registry: Registry containing model and capability information.
            backend_router: Router for accessing backends.
            classifier_settings: Settings for the classifier.
        """
        self._registry = model_registry
        self._router = backend_router
        self._settings = classifier_settings

        logger.info(
            "task_classifier_initialized",
            enabled=classifier_settings.enabled,
            model=classifier_settings.model,
            backend=classifier_settings.backend,
        )

    @property
    def enabled(self) -> bool:
        """Check if the classifier is enabled."""
        return self._settings.enabled

    def _get_classifier_backend(self) -> Backend:
        """Get the backend to use for classification.

        Returns:
            Backend instance.

        Raises:
            TaskClassifierError: If backend not found.
        """
        if self._settings.backend:
            backend = self._router.get_backend_by_name(self._settings.backend)
            if backend is None:
                raise TaskClassifierError(
                    f"Classifier backend '{self._settings.backend}' not found"
                )
            return backend
        return self._router.default_backend

    def _build_classification_prompt(self, task_description: str) -> str:
        """Build the classification prompt.

        Args:
            task_description: The task to classify.

        Returns:
            Formatted prompt string.
        """
        capabilities = self._registry.get_available_capabilities()
        return CLASSIFICATION_PROMPT_TEMPLATE.format(
            capabilities=", ".join(capabilities),
            task_description=task_description,
        )

    def _parse_classification_response(self, response_text: str) -> dict[str, Any]:
        """Parse the LLM response into classification data.

        Args:
            response_text: Raw LLM response text.

        Returns:
            Parsed classification data.

        Raises:
            TaskClassifierError: If parsing fails.
        """
        # Try to extract JSON from the response
        text = response_text.strip()

        # Try direct JSON parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in markdown code block
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find any JSON object in the text
        json_match = re.search(r"\{[^{}]*\}", text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        raise TaskClassifierError(f"Failed to parse classification response: {text[:200]}")

    def _calculate_alternatives(
        self, task_type: str, recommended_model: str
    ) -> list[AlternativeModel]:
        """Calculate alternative model recommendations.

        Args:
            task_type: The classified task type.
            recommended_model: The primary recommended model.

        Returns:
            List of alternative models with scores.
        """
        alternatives: list[AlternativeModel] = []
        all_models = self._registry.get_all_models()

        for model in all_models:
            if model.id == recommended_model:
                continue

            # Calculate a simple score based on capability match
            score = 0.0
            if task_type in model.capabilities:
                score = 0.7
            elif "general" in model.capabilities:
                score = 0.4
            else:
                score = 0.2

            alternatives.append(AlternativeModel(model=model.id, score=score))

        # Sort by score descending
        alternatives.sort(key=lambda x: x.score, reverse=True)
        return alternatives[:5]  # Return top 5 alternatives

    async def classify(self, task_description: str) -> ClassificationResult:
        """Classify a task and return recommended model.

        Args:
            task_description: Description of the task to classify.

        Returns:
            ClassificationResult with recommended model and details.

        Raises:
            TaskClassifierDisabledError: If classifier is disabled.
            TaskClassifierError: If classification fails.
        """
        if not self._settings.enabled:
            raise TaskClassifierDisabledError("Task classifier is disabled")

        logger.info(
            "classifying_task",
            task_description=task_description[:100],
        )

        # Get the classifier backend
        backend = self._get_classifier_backend()

        # Build the prompt
        prompt = self._build_classification_prompt(task_description)

        # Make the LLM request
        model = self._settings.model or "default"
        request = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,  # Deterministic for classification
            "max_tokens": 256,
        }

        try:
            response = await backend.chat_completions(request)
        except BackendError as e:
            logger.error(
                "classification_backend_error",
                error=str(e),
            )
            raise TaskClassifierError(f"Backend error during classification: {e}") from e

        # Extract the response content
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise TaskClassifierError(f"Invalid response structure: {e}") from e

        # Parse the classification
        classification = self._parse_classification_response(content)

        task_type = classification.get("task_type", "general")
        confidence = float(classification.get("confidence", 0.5))
        reasoning = classification.get("reasoning", "")

        # Find the best model for this task type
        recommended_model = self._registry.find_best_model_for_capability(task_type)
        if recommended_model is None:
            # Fall back to default
            default_model = self._registry.get_default_model_for_unknown_task()
            if default_model:
                recommended_model = default_model
            else:
                raise TaskClassifierError("No suitable model found for task")

        # Calculate alternatives
        alternatives = self._calculate_alternatives(task_type, recommended_model)

        result = ClassificationResult(
            recommended_model=recommended_model,
            task_type=task_type,
            confidence=confidence,
            reasoning=reasoning,
            alternatives=alternatives,
        )

        logger.info(
            "task_classified",
            task_type=task_type,
            confidence=confidence,
            recommended_model=recommended_model,
            alternatives_count=len(alternatives),
        )

        return result
