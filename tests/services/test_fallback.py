"""Tests for fallback service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from lexora.backends.base import (
    Backend,
    BackendConnectionError,
    BackendError,
    BackendRateLimitError,
)
from lexora.config import FallbackSettings
from lexora.services.fallback import FallbackService
from lexora.services.router import BackendRouter


class MockBackend:
    """Mock backend for testing."""

    def __init__(self, name: str):
        self.name = name


@pytest.fixture
def mock_router():
    """Create a mock router."""
    router = MagicMock(spec=BackendRouter)
    return router


@pytest.fixture
def fallback_settings():
    """Create default fallback settings."""
    return FallbackSettings(enabled=True, on_rate_limit=True)


@pytest.fixture
def fallback_service(mock_router, fallback_settings):
    """Create fallback service."""
    return FallbackService(mock_router, fallback_settings)


class TestFallbackServiceInit:
    """Tests for FallbackService initialization."""

    def test_enabled_property(self, fallback_service):
        """Test enabled property."""
        assert fallback_service.enabled is True

    def test_disabled_service(self, mock_router):
        """Test disabled fallback service."""
        settings = FallbackSettings(enabled=False)
        service = FallbackService(mock_router, settings)
        assert service.enabled is False


class TestShouldFallbackOnError:
    """Tests for should_fallback_on_error method."""

    def test_fallback_on_connection_error(self, fallback_service):
        """Test fallback on connection error."""
        error = BackendConnectionError("Connection failed")
        assert fallback_service.should_fallback_on_error(error) is True

    def test_fallback_on_backend_error(self, fallback_service):
        """Test fallback on generic backend error."""
        error = BackendError("Backend failed")
        assert fallback_service.should_fallback_on_error(error) is True

    def test_fallback_on_rate_limit_when_enabled(self, fallback_service):
        """Test fallback on rate limit when enabled."""
        error = BackendRateLimitError("Rate limited", retry_after=30)
        assert fallback_service.should_fallback_on_error(error) is True

    def test_no_fallback_on_rate_limit_when_disabled(self, mock_router):
        """Test no fallback on rate limit when disabled."""
        settings = FallbackSettings(enabled=True, on_rate_limit=False)
        service = FallbackService(mock_router, settings)

        error = BackendRateLimitError("Rate limited")
        assert service.should_fallback_on_error(error) is False

    def test_no_fallback_when_service_disabled(self, mock_router):
        """Test no fallback when service is disabled."""
        settings = FallbackSettings(enabled=False)
        service = FallbackService(mock_router, settings)

        error = BackendError("Backend failed")
        assert service.should_fallback_on_error(error) is False

    def test_no_fallback_on_non_backend_error(self, fallback_service):
        """Test no fallback on non-backend errors."""
        error = ValueError("Invalid value")
        assert fallback_service.should_fallback_on_error(error) is False


class TestExecuteWithFallback:
    """Tests for execute_with_fallback method."""

    @pytest.mark.asyncio
    async def test_success_on_primary(self, fallback_service, mock_router):
        """Test successful execution on primary backend."""
        primary_backend = MockBackend("primary")
        mock_router.get_backend_name_for_model.return_value = "primary"
        mock_router.get_backend_for_model.return_value = primary_backend

        async def operation(backend):
            return {"result": "success", "backend": backend.name}

        result, backend_name = await fallback_service.execute_with_fallback(
            "gpt-4", operation
        )

        assert result == {"result": "success", "backend": "primary"}
        assert backend_name == "primary"

    @pytest.mark.asyncio
    async def test_fallback_on_primary_failure(self, fallback_service, mock_router):
        """Test fallback when primary backend fails."""
        primary_backend = MockBackend("primary")
        fallback_backend = MockBackend("fallback")

        mock_router.get_backend_name_for_model.return_value = "primary"
        mock_router.get_backend_for_model.return_value = primary_backend
        mock_router.get_fallback_backends.return_value = [fallback_backend]

        call_count = 0

        async def operation(backend):
            nonlocal call_count
            call_count += 1
            if backend.name == "primary":
                raise BackendConnectionError("Primary failed")
            return {"result": "success", "backend": backend.name}

        result, backend_name = await fallback_service.execute_with_fallback(
            "gpt-4", operation
        )

        assert result == {"result": "success", "backend": "fallback"}
        assert backend_name == "fallback"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_fallback_on_rate_limit(self, fallback_service, mock_router):
        """Test fallback when primary hits rate limit."""
        primary_backend = MockBackend("primary")
        fallback_backend = MockBackend("fallback")

        mock_router.get_backend_name_for_model.return_value = "primary"
        mock_router.get_backend_for_model.return_value = primary_backend
        mock_router.get_fallback_backends.return_value = [fallback_backend]

        async def operation(backend):
            if backend.name == "primary":
                raise BackendRateLimitError("Rate limited", retry_after=60)
            return {"result": "success"}

        result, backend_name = await fallback_service.execute_with_fallback(
            "gpt-4", operation
        )

        assert backend_name == "fallback"

    @pytest.mark.asyncio
    async def test_all_backends_fail(self, fallback_service, mock_router):
        """Test error when all backends fail."""
        primary_backend = MockBackend("primary")
        fallback_backend = MockBackend("fallback")

        mock_router.get_backend_name_for_model.return_value = "primary"
        mock_router.get_backend_for_model.return_value = primary_backend
        mock_router.get_fallback_backends.return_value = [fallback_backend]

        async def operation(backend):
            raise BackendError(f"{backend.name} failed")

        with pytest.raises(BackendError) as exc_info:
            await fallback_service.execute_with_fallback("gpt-4", operation)

        assert "fallback failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_no_fallback_backends(self, fallback_service, mock_router):
        """Test error when no fallback backends configured."""
        primary_backend = MockBackend("primary")

        mock_router.get_backend_name_for_model.return_value = "primary"
        mock_router.get_backend_for_model.return_value = primary_backend
        mock_router.get_fallback_backends.return_value = []

        async def operation(backend):
            raise BackendError("Primary failed")

        with pytest.raises(BackendError):
            await fallback_service.execute_with_fallback("gpt-4", operation)

    @pytest.mark.asyncio
    async def test_multiple_fallbacks(self, fallback_service, mock_router):
        """Test trying multiple fallback backends."""
        primary_backend = MockBackend("primary")
        fallback1 = MockBackend("fallback1")
        fallback2 = MockBackend("fallback2")

        mock_router.get_backend_name_for_model.return_value = "primary"
        mock_router.get_backend_for_model.return_value = primary_backend
        mock_router.get_fallback_backends.return_value = [fallback1, fallback2]

        call_order = []

        async def operation(backend):
            call_order.append(backend.name)
            if backend.name in ["primary", "fallback1"]:
                raise BackendError(f"{backend.name} failed")
            return {"result": "success"}

        result, backend_name = await fallback_service.execute_with_fallback(
            "gpt-4", operation
        )

        assert call_order == ["primary", "fallback1", "fallback2"]
        assert backend_name == "fallback2"

    @pytest.mark.asyncio
    async def test_no_fallback_when_disabled(self, mock_router):
        """Test no fallback when service is disabled."""
        settings = FallbackSettings(enabled=False)
        service = FallbackService(mock_router, settings)

        primary_backend = MockBackend("primary")
        mock_router.get_backend_name_for_model.return_value = "primary"
        mock_router.get_backend_for_model.return_value = primary_backend

        async def operation(backend):
            raise BackendError("Primary failed")

        with pytest.raises(BackendError) as exc_info:
            await service.execute_with_fallback("gpt-4", operation)

        assert "Primary failed" in str(exc_info.value)


class TestExecuteStreamWithFallback:
    """Tests for execute_stream_with_fallback method."""

    @pytest.mark.asyncio
    async def test_success_on_primary(self, fallback_service, mock_router):
        """Test successful stream execution on primary backend."""
        primary_backend = MockBackend("primary")
        mock_router.get_backend_name_for_model.return_value = "primary"
        mock_router.get_backend_for_model.return_value = primary_backend

        async def stream_gen():
            yield b"data: chunk1\n"
            yield b"data: chunk2\n"

        def operation(backend):
            return stream_gen()

        stream, backend_name = await fallback_service.execute_stream_with_fallback(
            "gpt-4", operation
        )

        assert backend_name == "primary"
        # Verify stream works
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        assert len(chunks) == 2

    @pytest.mark.asyncio
    async def test_fallback_on_stream_creation_error(self, fallback_service, mock_router):
        """Test fallback when stream creation fails."""
        primary_backend = MockBackend("primary")
        fallback_backend = MockBackend("fallback")

        mock_router.get_backend_name_for_model.return_value = "primary"
        mock_router.get_backend_for_model.return_value = primary_backend
        mock_router.get_fallback_backends.return_value = [fallback_backend]

        async def stream_gen():
            yield b"data: success\n"

        def operation(backend):
            if backend.name == "primary":
                raise BackendConnectionError("Cannot create stream")
            return stream_gen()

        stream, backend_name = await fallback_service.execute_stream_with_fallback(
            "gpt-4", operation
        )

        assert backend_name == "fallback"
