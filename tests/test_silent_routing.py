"""T-silent-routing: routing decisions must not be made silently.

The tests in this file cover the requirements the T-silent-routing spec
introduces (msg-078 through msg-083):

* R-5 — a YAML file with duplicate mapping keys is rejected at load time.
  PyYAML's SafeLoader silently keeps the last value; the strict loader
  this branch installs raises ``DuplicateYamlKeyError`` instead, so a
  config where the file text disagrees with the loaded dict does not
  reach the application.
* R-1b — a config where a tier name is also declared as a backend model
  name is rejected at ``RoutingSettings`` construction. Tier lookup wins
  over by-name at request time, so the model declaration would be
  unreachable-but-present, which is the class of lie this branch is
  against.
* R-1a — a model name declared by two or more backends emits a boot-time
  WARNING and is refused with HTTP 404 at request time, rather than
  silently resolving to whichever backend happened to be the last writer
  into ``_model_to_backend``.
* R-1a-5 — the ambiguity check applies only to the requested name. A
  tier alias whose concrete model happens to be declared by three
  backends must still route: the request field is the tier name, not the
  concrete model.
* R-2 — a model name that matches no tier and no declared backend model
  is refused with HTTP 404 rather than silently falling through to
  ``default_backend``. The refusal is logged at WARNING (not DEBUG) so
  the operational surface for fall-through requests survives the
  ``INFO``-level log filter that runs in production.
* W-2 — a refusal message may only offer remedies that exist in the
  config being served. "Use a tier name" is generated from the tiers
  the loaded config actually offers (which ones qualify is R-6); when
  none do, the message says the name is not routable instead of
  inventing one.
* W-3 — ``/v1/models`` never enumerates a name the router will 404. The
  listing is narrowed to the by-name routable set, which drops both
  ambiguous names and names only the upstream knows about, so the
  advertised set is a subset of the routable set.
* T-models-advertise-side — W-3 in the other direction, which makes it
  an equality: a routable name is never hidden either. The subset rule
  used "an upstream reported it" as a proxy for "it exists", and the
  proxy is wrong for a backend that is not a model catalogue —
  ``ClaudeCodeBackend.list_models`` returns a hardcoded empty list, so
  the shipped ``claude-code-*`` names routed while ``/v1/models`` denied
  they existed, with no tier mapping to that backend to reach them by
  either.
* R-6 — a tier offered as the remedy for an ambiguous name must
  resolve to the name that was asked for, not merely reach one of the
  backends that declared it. A tier that reaches the right backend and
  serves a different model turns the refusal into a redirect.
* R-7 — the ``backend`` field on a concrete ``/v1/models`` row names
  the backend that declares the model, not whichever backend's upstream
  reported it first. Backends sharing one upstream each see the whole
  catalogue, so "routable somewhere" cannot decide whose row it is.
* 404 body shape — clients see the OpenAI standard ``model_not_found``
  code; the anthropic-shaped ``/v1/messages`` endpoint sees the
  Anthropic ``{"type": "error", ...}`` envelope. The distinction between
  "unknown" and "ambiguous" is a server-side concern (log event names)
  and does not appear in the API code.
* R-8 — the duplicate-key check reads the keys the file writes, before a
  merge key is resolved. Inheriting a block with ``<<: *anchor`` and then
  overriding one field is a legal file whose text and loaded value agree,
  so it must load; only a mapping that writes a key twice is the R-5
  failure.
* R-9 — which dialect a 404 body is written in follows the route that
  matched, not the request URL string. An ASGI ``root_path`` deployment
  whose proxy forwards the prefix leaves the prefix in ``scope["path"]``
  while the router still matches ``/v1/messages``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock

from lexora.backends.base import ModelNotFoundError
from lexora.config import (
    BackendSettings,
    DuplicateYamlKeyError,
    RoutingSettings,
    TierSettings,
    VLLMSettings,
    create_settings,
    load_yaml_config,
)
from lexora.main import create_app
from lexora.services.router import BackendRouter


# --------------------------------------------------------------------------
# R-5 — strict YAML loader
# --------------------------------------------------------------------------


class TestStrictYamlLoader:
    """R-5: reject duplicate keys at parse time.

    The bug this catches is subtle by construction: PyYAML's SafeLoader
    returns a Python dict that Python then sees as single-keyed and
    correct. Neither Pydantic nor any downstream validator can see that
    the source file disagreed with the loaded value — the disagreement
    only exists in the text, and the text is gone by the time anyone
    else looks. That is exactly the class this branch is against: a
    config the operator wrote and the application never saw.
    """

    def test_duplicate_top_level_key_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "cfg.yaml"
        path.write_text("a: 1\na: 2\n")
        with pytest.raises(DuplicateYamlKeyError) as exc_info:
            load_yaml_config(path)
        # The error must name the key and both lines so an operator can
        # find the collision without re-parsing the file.
        message = str(exc_info.value)
        assert "'a'" in message
        assert "line 1" in message
        assert "line 2" in message

    def test_duplicate_nested_key_rejected(self, tmp_path: Path) -> None:
        """Nesting is not a loophole: R-5 is applied to every mapping in the file.

        A duplicate at ``routing.tiers.frontier`` was one of Einstein's
        cited failure modes (msg-080 V-1) — the last-writer-wins would
        happen at parse time and no app-level tier validator could see
        it. This test is the general form of that concern.
        """
        path = tmp_path / "cfg.yaml"
        path.write_text(
            "routing:\n"
            "  tiers:\n"
            "    frontier:\n"
            "      backend: gemini\n"
            "    frontier:\n"
            "      backend: heavy\n"
        )
        with pytest.raises(DuplicateYamlKeyError) as exc_info:
            load_yaml_config(path)
        assert "'frontier'" in str(exc_info.value)

    def test_non_duplicate_yaml_still_loads(self, tmp_path: Path) -> None:
        """R-5 is scope-preserving: legal YAML keeps loading.

        A regression here would be worse than not having R-5 at all:
        every config in every environment would start refusing to load.
        """
        path = tmp_path / "cfg.yaml"
        path.write_text("a: 1\nb: 2\nc:\n  d: 3\n  e: 4\n")
        result = load_yaml_config(path)
        assert result == {"a": 1, "b": 2, "c": {"d": 3, "e": 4}}


# --------------------------------------------------------------------------
# R-8 — the duplicate check reads what the file writes, not what a merge
#       key resolved to
# --------------------------------------------------------------------------


def _duplicate_lines(message: str) -> tuple[int, int]:
    """Pull ``(first_seen_line, repeated_line)`` out of the error message.

    The message is the operator-facing artefact, so the test reads it the
    way an operator would rather than reaching into the exception.
    """
    match = re.search(
        r"first seen at line (\d+), column \d+; repeated at line (\d+)", message
    )
    assert match is not None, message
    return int(match.group(1)), int(match.group(2))


class TestStrictLoaderAcceptsMergeKeys:
    """R-8: a merge key that is then overridden is not a duplicate.

    ``<<: *anchor`` is how YAML expresses "inherit this block, then say
    what is different about mine". PyYAML resolves it by prepending the
    inherited pairs into the same mapping node, so after resolution the
    overridden key genuinely appears twice at that level — but the file
    does not write it twice, YAML defines which one wins, and the loaded
    config is what the file says. R-5 is about the file and the loaded
    config disagreeing; this is the case where they agree.

    Refusing these files is a regression against ``origin/develop``,
    which used the stock ``yaml.safe_load``. It is also the failure mode
    this whole branch is against, pointed the other way: the loader
    refuses to boot while naming a line that has nothing wrong with it
    and instructing the operator to "fix the source file".
    """

    def test_merge_key_with_override_loads_and_matches_stock_loader(
        self, tmp_path: Path
    ) -> None:
        """The ordinary anchor-and-override shape.

        Asserting equality with ``yaml.safe_load`` rather than merely
        "no exception": an implementation that dropped ``<<`` handling
        would also not raise, and would silently lose the inherited
        keys — which is the same class of bug R-5 exists to stop.
        """
        text = (
            "defaults: &defaults\n"
            "  type: vllm\n"
            "  timeout: 60\n"
            "backends:\n"
            "  a:\n"
            "    <<: *defaults\n"
            "    timeout: 300\n"
        )
        path = tmp_path / "cfg.yaml"
        path.write_text(text)
        assert load_yaml_config(path) == yaml.safe_load(text)
        assert load_yaml_config(path)["backends"]["a"] == {
            "type": "vllm",
            "timeout": 300,
        }

    def test_two_merged_anchors_overlapping_keys_load_with_the_first_winning(
        self, tmp_path: Path
    ) -> None:
        """``<<: [*a, *b]`` — YAML defines the precedence, the loader keeps it.

        This is the case where checking after the flatten reported the
        "repeated" line *above* the "first seen" line, because PyYAML
        prepends the merged pairs in reverse. The file is legal and the
        earlier anchor wins.
        """
        text = (
            "fast: &fast\n"
            "  timeout: 60\n"
            "slow: &slow\n"
            "  timeout: 900\n"
            "chosen:\n"
            "  <<: [*fast, *slow]\n"
        )
        path = tmp_path / "cfg.yaml"
        path.write_text(text)
        loaded = load_yaml_config(path)
        assert loaded == yaml.safe_load(text)
        assert loaded["chosen"] == {"timeout": 60}

    def test_explicit_duplicate_beside_a_merge_key_still_raises(
        self, tmp_path: Path
    ) -> None:
        """Narrowing the predicate must not narrow it past the real case.

        The two lines the message names must be the two lines the
        operator actually wrote the key on — both inside the same
        mapping. Checking after the flatten names the anchor's line as
        "first seen", which sends the operator to a mapping they did not
        write a duplicate in.
        """
        text = (
            "defaults: &defaults\n"  # line 1
            "  timeout: 60\n"  # line 2
            "backends:\n"  # line 3
            "  a:\n"  # line 4
            "    <<: *defaults\n"  # line 5
            "    timeout: 300\n"  # line 6
            "    timeout: 400\n"  # line 7
        )
        path = tmp_path / "cfg.yaml"
        path.write_text(text)
        with pytest.raises(DuplicateYamlKeyError) as exc_info:
            load_yaml_config(path)
        first, repeated = _duplicate_lines(str(exc_info.value))
        assert (first, repeated) == (6, 7)
        assert first < repeated

    def test_duplicate_inside_an_anchor_block_still_raises(
        self, tmp_path: Path
    ) -> None:
        """An anchor is not a hiding place.

        The anchored mapping is built as part of the document tree, so
        it is checked on its own terms even though the only place it is
        read from is a merge key.
        """
        path = tmp_path / "cfg.yaml"
        path.write_text(
            "defaults: &defaults\n"
            "  timeout: 60\n"
            "  timeout: 90\n"
            "backends:\n"
            "  a:\n"
            "    <<: *defaults\n"
        )
        with pytest.raises(DuplicateYamlKeyError) as exc_info:
            load_yaml_config(path)
        first, repeated = _duplicate_lines(str(exc_info.value))
        assert (first, repeated) == (2, 3)

    def test_repeated_value_key_still_raises(self, tmp_path: Path) -> None:
        """``=`` is an entry of the mapping, not an instruction to the loader.

        PyYAML gives ``=`` its own tag, which is why it needs handling
        alongside ``<<`` — but the resemblance stops there.
        ``flatten_mapping`` retags it to a plain string and the built
        mapping carries the key ``"="``, so ``yaml.safe_load`` on this
        file returns ``{'a': {'=': 2}}``: the first value is gone and
        nothing said so. Stepping over it the way ``<<`` is stepped over
        would leave that last-writer-wins inside the one check whose
        whole purpose is to stop it.
        """
        path = tmp_path / "cfg.yaml"
        path.write_text("a:\n  =: 1\n  =: 2\n")
        assert yaml.safe_load("a:\n  =: 1\n  =: 2\n") == {"a": {"=": 2}}
        with pytest.raises(DuplicateYamlKeyError) as exc_info:
            load_yaml_config(path)
        assert "'='" in str(exc_info.value)
        first, repeated = _duplicate_lines(str(exc_info.value))
        assert (first, repeated) == (2, 3)


# --------------------------------------------------------------------------
# R-1b — tier / model name collision
# --------------------------------------------------------------------------


class TestTierBackendCollision:
    """R-1b: a tier name declared as a backend model must not silently disappear.

    ``get_backend_for_model`` looks up tiers first, so if a tier
    ``frontier`` is also declared as a model on some backend, the model
    declaration is present but unreachable. Rather than accept the
    unreachable declaration, ``RoutingSettings`` refuses to construct.
    """

    def test_collision_at_boot_is_rejected(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            RoutingSettings(
                enabled=True,
                backends={
                    "b1": BackendSettings(
                        url="http://localhost:1",
                        # `frontier` also appears as a tier below
                        models=[{"name": "frontier"}],
                    ),
                },
                tiers={
                    "frontier": TierSettings(backend="b1", model="frontier"),
                },
            )
        # The message must name both sides so the operator can fix it.
        message = str(exc_info.value)
        assert "frontier" in message
        assert "b1" in message

    def test_no_collision_still_constructs(self) -> None:
        """R-1b applies narrowly; a config with disjoint tier and model
        names must keep constructing without incident."""
        settings = RoutingSettings(
            enabled=True,
            backends={
                "b1": BackendSettings(
                    url="http://localhost:1", models=[{"name": "real-model"}]
                ),
            },
            tiers={
                "light": TierSettings(backend="b1", model="real-model"),
            },
        )
        assert "light" in settings.tiers
        assert settings.backends["b1"].models[0].name == "real-model"


class TestSameTierDeclaredTwiceIsRefused:
    """R-1b acceptance test, stated the way the spec states it.

    Spec v2 (msg-081 §3.2, restated in msg-083): *"given a config where
    two backends declare the same tier, loading the configuration must
    fail"* — and explicitly **which layer refuses is not part of the
    requirement**. The point of writing it layer-agnostically is to
    catch the case where R-1b was implemented but is unreachable because
    the parser already ate the collision (Einstein, msg-080): an R-1b
    unit test alone would pass without ever proving the config is
    refused end to end.

    How the collision is expressed here follows from the schema, not
    from a choice: ``RoutingSettings.tiers`` is a ``dict[str,
    TierSettings]`` whose *key* is the tier name and whose ``backend``
    field names the backend. There is no field on a backend by which it
    could claim a tier, so "two backends declare the same tier" can only
    be written as one tier key appearing twice with two different
    ``backend`` values — which is the form below. Under stock
    ``yaml.SafeLoader`` this parses to the second entry with nothing
    logged; the R-5 loader refuses it. R-1b is therefore not an empty
    requirement, and it is satisfied one layer below where it was
    originally written.
    """

    def test_two_backends_claiming_one_tier_fails_to_load(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "cfg.yaml"
        path.write_text(
            "routing:\n"
            "  enabled: true\n"
            "  default_backend: b1\n"
            "  backends:\n"
            "    b1:\n"
            "      url: http://localhost:1\n"
            "    b2:\n"
            "      url: http://localhost:2\n"
            "  tiers:\n"
            "    heavy:\n"
            "      backend: b1\n"
            "    heavy:\n"
            "      backend: b2\n",
            encoding="utf-8",
        )
        # Deliberately broad: the requirement is "the config does not
        # load", not "layer X raised". ``DuplicateYamlKeyError`` is a
        # ``ValueError``, as is Pydantic's ``ValidationError``, so this
        # keeps passing if the refusal ever moves between layers.
        with pytest.raises(ValueError):
            create_settings(path)

    def test_stock_loader_would_have_accepted_it(self, tmp_path: Path) -> None:
        """Pin the reason the test above is not tautological.

        If PyYAML ever started rejecting duplicate keys on its own, the
        test above would pass for a reason that has nothing to do with
        this branch, and deleting ``_StrictSafeLoader`` would not turn it
        red. This asserts the stock loader still silently picks a winner,
        so the coverage above is attributable to the code we added.
        """
        import yaml

        path = tmp_path / "cfg.yaml"
        path.write_text(
            "tiers:\n"
            "  heavy:\n"
            "    backend: b1\n"
            "  heavy:\n"
            "    backend: b2\n",
            encoding="utf-8",
        )
        with open(path, "r", encoding="utf-8") as fh:
            stock = yaml.safe_load(fh)
        assert stock == {"tiers": {"heavy": {"backend": "b2"}}}


# --------------------------------------------------------------------------
# R-1a — model-name ambiguity: boot WARNING + request-time 404
# --------------------------------------------------------------------------


def _shared_model_router() -> BackendRouter:
    """Router configured so ``dup-model`` is declared by three backends.

    Kept as a helper because R-1a is the requirement most easily tested
    by symmetry: the same fixture drives the boot-time behavior, the
    request-time refusal, and (with the same name in the tier position)
    the R-1a-5 invariant.
    """
    return BackendRouter(
        routing_settings=RoutingSettings(
            enabled=True,
            default_backend="b1",
            backends={
                "b1": BackendSettings(
                    url="http://localhost:1", models=[{"name": "dup-model"}]
                ),
                "b2": BackendSettings(
                    url="http://localhost:2", models=[{"name": "dup-model"}]
                ),
                "b3": BackendSettings(
                    url="http://localhost:3", models=[{"name": "dup-model"}]
                ),
            },
            tiers={
                "light": TierSettings(backend="b1", model="dup-model"),
                "medium": TierSettings(backend="b2", model="dup-model"),
                "heavy": TierSettings(backend="b3", model="dup-model"),
            },
        ),
        vllm_settings=VLLMSettings(url="http://localhost:8000"),
    )


class TestModelAmbiguity:
    """R-1a: same model name declared by multiple backends."""

    def test_boot_emits_warning_with_all_backends(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The WARNING must list every colliding backend, not just the winner."""
        with caplog.at_level(logging.WARNING):
            _shared_model_router()
        matching = [
            r for r in caplog.records if "model_declaration_ambiguous" in r.message
        ]
        assert len(matching) == 1, (
            "Expected exactly one ambiguity WARNING for 'dup-model' "
            "(three backends collided, one WARNING). Got: "
            f"{[r.message for r in matching]}"
        )
        text = matching[0].message
        for backend in ("b1", "b2", "b3"):
            assert backend in text, (
                f"WARNING must name every colliding backend so the "
                f"operator does not have to grep INFO lines; missing {backend}"
            )

    def test_ambiguous_by_name_request_refused_with_404_reason(self) -> None:
        """Raw-name request for the ambiguous model raises ModelNotFoundError."""
        router = _shared_model_router()
        with pytest.raises(ModelNotFoundError) as exc_info:
            router.get_backend_for_model("dup-model")
        assert exc_info.value.reason == "ambiguous"
        assert exc_info.value.model_name == "dup-model"

    def test_ambiguity_logged_at_warning_on_refusal(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The refusal itself must be at WARNING so operators see it in
        production (default log level is INFO, so DEBUG would vanish)."""
        router = _shared_model_router()
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            with pytest.raises(ModelNotFoundError):
                router.get_backend_for_model("dup-model")
        matching = [
            r for r in caplog.records if "model_ambiguous_refused" in r.message
        ]
        assert matching, (
            "Every ambiguous-name refusal must log 'model_ambiguous_refused' "
            "at WARNING so the request appears in production logs. Got: "
            f"{[r.message for r in caplog.records]}"
        )

    def test_get_backend_name_for_ambiguous_also_raises(self) -> None:
        """The stats-collector path must not silently attribute an
        ambiguous request to the last-writer-wins backend."""
        router = _shared_model_router()
        with pytest.raises(ModelNotFoundError):
            router.get_backend_name_for_model("dup-model")


# --------------------------------------------------------------------------
# R-1a-5 — the ambiguity check is on the REQUESTED name, not the resolved one
# --------------------------------------------------------------------------


class TestTierRoutingSurvivesAmbiguousConcreteModel:
    """R-1a-5: tier routing must not be collateral damage of R-1a.

    The failure this pins is that a naive R-1a implementation would take
    the resolved model name after tier lookup and apply the ambiguity
    check to it — killing every tier whose concrete model was shared.
    That would make the shipping config's ``light`` / ``medium`` /
    ``heavy`` tiers all 404 because they all resolve to
    ``Qwen3.8-27B``, which is exactly what the routing spec calls the
    "shortest path to tier-routing failure and gate death" (msg-083
    W-1).
    """

    def test_each_tier_still_routes_despite_shared_concrete_model(self) -> None:
        router = _shared_model_router()
        # All three tiers resolve to 'dup-model' and each points at a
        # different backend. The ambiguity of 'dup-model' as a raw name
        # is unrelated to whether the tiers route.
        assert router.get_backend_for_model("light") is router.backends["b1"]
        assert router.get_backend_for_model("medium") is router.backends["b2"]
        assert router.get_backend_for_model("heavy") is router.backends["b3"]

    def test_tier_resolves_to_concrete_model(self) -> None:
        """The resolved model must still be the concrete name, even
        though that name would 404 as a raw request."""
        router = _shared_model_router()
        assert router.resolve_model("light") == "dup-model"
        assert router.resolve_model("heavy") == "dup-model"


# --------------------------------------------------------------------------
# R-2 — unknown model refused with 404 (no fall-through to default_backend)
# --------------------------------------------------------------------------


class TestUnknownModelRefused:
    """R-2: names that match neither tier nor declared model must 404.

    The old behavior fell through to ``default_backend`` with a single
    DEBUG line, which vanished under the INFO-level production filter.
    The point of R-2 is that "silently routed somewhere" is worse than
    "loudly refused" — even (especially) when the somewhere happens to
    be a real backend that accepts the request.
    """

    def _router(self) -> BackendRouter:
        return BackendRouter(
            routing_settings=RoutingSettings(
                enabled=True,
                default_backend="b1",
                backends={
                    "b1": BackendSettings(
                        url="http://localhost:1",
                        models=[{"name": "model-a"}],
                    ),
                },
                tiers={"light": TierSettings(backend="b1", model="model-a")},
            ),
            vllm_settings=VLLMSettings(url="http://localhost:8000"),
        )

    def test_unknown_name_raises_not_fall_through(self) -> None:
        router = self._router()
        with pytest.raises(ModelNotFoundError) as exc_info:
            router.get_backend_for_model("no-such-model")
        assert exc_info.value.reason == "unknown"
        assert exc_info.value.model_name == "no-such-model"

    def test_refusal_logged_at_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        router = self._router()
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            with pytest.raises(ModelNotFoundError):
                router.get_backend_for_model("no-such-model")
        matching = [
            r for r in caplog.records if "model_unknown_refused" in r.message
        ]
        assert matching, (
            "R-2's contract is 'do not silently fall through' — the "
            "refusal must be logged at WARNING (not DEBUG), because the "
            "shipping log level is INFO. Got records: "
            f"{[r.message for r in caplog.records]}"
        )

    def test_known_tier_and_model_still_route(self) -> None:
        """R-2 must be tight enough to not break the legal cases."""
        router = self._router()
        assert router.get_backend_for_model("light") is router.backends["b1"]
        assert router.get_backend_for_model("model-a") is router.backends["b1"]


# --------------------------------------------------------------------------
# W-2 — a refusal message may only advertise remedies that exist
# --------------------------------------------------------------------------


class TestRefusalMessageOffersOnlyRealRemedies:
    """W-3's sibling: the *message* must not describe a gateway that isn't.

    "Use a tier name instead" is sound advice only when some tier
    reaches one of the backends that declared the requested name. Two
    backends can share a model name with neither exposed as a tier, and
    in that config the advice sends the caller hunting for something
    that does not exist. Since this whole branch is about the gateway
    not stating things about itself that are untrue, a hardcoded remedy
    string would reproduce the defect one layer up from where it was
    fixed.
    """

    def test_message_lists_the_tiers_that_actually_reach_the_backends(
        self,
    ) -> None:
        router = _shared_model_router()
        with pytest.raises(ModelNotFoundError) as exc_info:
            router.get_backend_for_model("dup-model")
        message = str(exc_info.value)
        # Every colliding backend is named: the operator has to know
        # which declarations collided, not which one would have won.
        for backend_name in ("b1", "b2", "b3"):
            assert backend_name in message
        # ...and every tier offered is one that exists in this config.
        for tier in ("light", "medium", "heavy"):
            assert tier in message

    def test_message_refuses_to_invent_a_tier_when_none_exists(self) -> None:
        """The case W-2 was written for: no tier reaches the collision.

        Without the config lookup this message would still say "use a
        tier name instead" while the config declares no tiers at all.
        """
        router = BackendRouter(
            routing_settings=RoutingSettings(
                enabled=True,
                default_backend="b1",
                backends={
                    "b1": BackendSettings(
                        url="http://localhost:1", models=[{"name": "dup-model"}]
                    ),
                    "b2": BackendSettings(
                        url="http://localhost:2", models=[{"name": "dup-model"}]
                    ),
                },
                # No tiers at all — nothing to recommend.
                tiers={},
            ),
            vllm_settings=VLLMSettings(url="http://localhost:8000"),
        )
        with pytest.raises(ModelNotFoundError) as exc_info:
            router.get_backend_for_model("dup-model")
        message = str(exc_info.value)
        assert "not routable" in message
        assert "/v1/models" in message
        assert "Use one of these tier names" not in message

    def test_message_ignores_tiers_pointing_elsewhere(self) -> None:
        """Only tiers reaching a *declaring* backend are a remedy.

        A tier that routes to some unrelated backend does not help a
        caller who asked for this name, so listing it would be a
        different flavour of the same lie.
        """
        router = BackendRouter(
            routing_settings=RoutingSettings(
                enabled=True,
                default_backend="b1",
                backends={
                    "b1": BackendSettings(
                        url="http://localhost:1", models=[{"name": "dup-model"}]
                    ),
                    "b2": BackendSettings(
                        url="http://localhost:2", models=[{"name": "dup-model"}]
                    ),
                    "b3": BackendSettings(
                        url="http://localhost:3", models=[{"name": "unrelated"}]
                    ),
                },
                tiers={"elsewhere": TierSettings(backend="b3")},
            ),
            vllm_settings=VLLMSettings(url="http://localhost:8000"),
        )
        with pytest.raises(ModelNotFoundError) as exc_info:
            router.get_backend_for_model("dup-model")
        message = str(exc_info.value)
        assert "elsewhere" not in message
        assert "not routable" in message

    def test_both_lookup_paths_give_the_same_message(self) -> None:
        """``get_backend_name_for_model`` reaches the caller through the
        same 404 handler, so its advice must not be a shorter variant."""
        router = _shared_model_router()
        with pytest.raises(ModelNotFoundError) as by_backend:
            router.get_backend_for_model("dup-model")
        with pytest.raises(ModelNotFoundError) as by_name:
            router.get_backend_name_for_model("dup-model")
        assert str(by_backend.value) == str(by_name.value)


# --------------------------------------------------------------------------
# R-6 — a suggested tier must resolve to the model that was requested
# --------------------------------------------------------------------------


class TestRemedyTiersResolveToTheRequestedModel:
    """R-6: selecting remedy tiers on the backend alone is not enough.

    A backend can declare the ambiguous name *and* a second, unambiguous
    name that is exposed as a tier. Such a tier reaches a colliding
    backend, so a backend-only filter offers it — but it resolves to the
    other model. A caller who follows the advice is answered by a model
    they did not ask for, and this time the substitution is the one the
    gateway named. That is the same class as the fall-through this branch
    removed, one step further along: the refusal stops being a refusal
    and becomes a redirect.

    The retained predicate is therefore twofold — the tier reaches one of
    the declaring backends *and* resolves to the requested model. Keeping
    the backend half matters: a tier may name the requested model
    explicitly while pointing at a backend that never declared it, and
    telling a caller to send the name to a backend the config does not
    say serves it is a remedy invented rather than found.
    """

    @staticmethod
    def _router(tiers: dict[str, TierSettings]) -> BackendRouter:
        """Two backends collide on ``shared``; ``b1`` also serves ``other``."""
        return BackendRouter(
            routing_settings=RoutingSettings(
                enabled=True,
                default_backend="b1",
                backends={
                    "b1": BackendSettings(
                        url="http://localhost:1",
                        models=[{"name": "shared"}, {"name": "other"}],
                    ),
                    "b2": BackendSettings(
                        url="http://localhost:2", models=[{"name": "shared"}]
                    ),
                },
                tiers=tiers,
            ),
            vllm_settings=VLLMSettings(url="http://localhost:8000"),
        )

    def test_tier_resolving_to_a_different_model_is_not_offered(self) -> None:
        router = self._router({"vision": TierSettings(backend="b1", model="other")})
        # The tier is a live route -- to the wrong model. That is what
        # makes offering it worse than offering nothing.
        assert router.resolve_model("vision") == "other"
        with pytest.raises(ModelNotFoundError) as exc_info:
            router.get_backend_for_model("shared")
        message = str(exc_info.value)
        assert "vision" not in message, (
            "'vision' reaches a colliding backend but resolves to 'other'. "
            "Naming it as the remedy for a 'shared' request tells the "
            f"caller to accept a different model. Got: {message}"
        )

    def test_tier_resolving_to_the_requested_model_is_still_offered(self) -> None:
        """The pin against over-narrowing: "offer nothing" is not the fix."""
        router = self._router(
            {
                "vision": TierSettings(backend="b1", model="other"),
                "big": TierSettings(backend="b2", model="shared"),
            }
        )
        assert router.resolve_model("big") == "shared"
        with pytest.raises(ModelNotFoundError) as exc_info:
            router.get_backend_for_model("shared")
        message = str(exc_info.value)
        assert "big" in message, (
            "'big' reaches a declaring backend and resolves to the "
            f"requested model, so it is a real remedy. Got: {message}"
        )
        assert "vision" not in message

    def test_fallback_does_not_deny_a_tier_that_does_reach_the_backends(
        self,
    ) -> None:
        """The sentence has to be about what the filter actually tests.

        Narrowing the filter from "reaches those backends" to "resolves to
        this model" without moving the sentence with it replaces a wrong
        remedy with a correctly-shaped false statement: here a tier
        *does* reach a colliding backend, so any message denying that is
        untrue even though the remedy list is now right.
        """
        router = self._router({"vision": TierSettings(backend="b1", model="other")})
        # Stated through the public lookup so the precondition is not a
        # restatement of the implementation: 'vision' routes to 'b1',
        # and 'b1' is one of the backends the message itself names.
        assert router.get_backend_name_for_model("vision") == "b1"
        with pytest.raises(ModelNotFoundError) as exc_info:
            router.get_backend_for_model("shared")
        message = str(exc_info.value)
        assert "b1" in message
        assert "No tier routes to any of those backends" not in message, (
            "A tier does route to 'b1'. The message may say no tier "
            f"resolves to 'shared'; it may not deny reachability. Got: {message}"
        )
        assert "shared" in message


# --------------------------------------------------------------------------
# W-3 — /v1/models never enumerates a name the router will 404
# --------------------------------------------------------------------------


class TestListModelsMatchesRoutableSet:
    """W-3: the advertised set is a subset of the routable set.

    Two ways a listed id can become unroutable once R-1a / R-2 are in:
    the name is declared by several backends (ambiguous), or the
    upstream serves it and no backend declares it. Both end in a 404, so
    both have to leave the listing — filtering only the first would fix
    the case we happened to think of rather than the invariant.
    """

    @pytest.mark.asyncio
    async def test_ambiguous_name_dropped_from_listing(self) -> None:
        router = _shared_model_router()
        # Simulate three backends proxying the same upstream, each
        # returning an identical row for 'dup-model'.
        for name in ("b1", "b2", "b3"):
            router.backends[name].list_models = AsyncMock(
                return_value={
                    "object": "list",
                    "data": [
                        {
                            "id": "dup-model",
                            "object": "model",
                            "created": 1_700_000_000,
                            "owned_by": "vllm",
                        }
                    ],
                }
            )
        listing = await router.list_all_models()
        ids = [m["id"] for m in listing["data"]]
        assert "dup-model" not in ids, (
            "The router filtered ambiguous 'dup-model' from /v1/models so "
            "the advertised set is a subset of the routable set. Got "
            f"ids: {ids}"
        )
        # Tier aliases must still appear — they route.
        assert {"light", "medium", "heavy"} <= set(ids)

    @pytest.mark.asyncio
    async def test_unambiguous_name_still_listed(self) -> None:
        """The filter is scoped: a name declared by exactly one backend
        keeps its /v1/models row."""
        router = BackendRouter(
            routing_settings=RoutingSettings(
                enabled=True,
                default_backend="b1",
                backends={
                    "b1": BackendSettings(
                        url="http://localhost:1", models=[{"name": "solo"}]
                    ),
                },
            ),
            vllm_settings=VLLMSettings(url="http://localhost:8000"),
        )
        router.backends["b1"].list_models = AsyncMock(
            return_value={
                "object": "list",
                "data": [
                    {
                        "id": "solo",
                        "object": "model",
                        "created": 1,
                        "owned_by": "vllm",
                    }
                ],
            }
        )
        listing = await router.list_all_models()
        ids = {m["id"] for m in listing["data"]}
        assert "solo" in ids

    @pytest.mark.asyncio
    async def test_duplicate_rows_from_shared_upstream_deduplicated(self) -> None:
        """Two backends proxying the same upstream must not produce two
        identical rows for the same routable id."""
        router = BackendRouter(
            routing_settings=RoutingSettings(
                enabled=True,
                default_backend="b1",
                backends={
                    "b1": BackendSettings(
                        url="http://localhost:1", models=[{"name": "solo"}]
                    ),
                    "b2": BackendSettings(
                        url="http://localhost:1", models=[{"name": "other"}]
                    ),
                },
            ),
            vllm_settings=VLLMSettings(url="http://localhost:8000"),
        )
        # ``solo`` is declared by exactly one backend, so it is routable
        # and survives the W-3 filter — but both backends point at the
        # same upstream and so both report it.
        upstream_payload = {
            "object": "list",
            "data": [
                {
                    "id": "solo",
                    "object": "model",
                    "created": 1,
                    "owned_by": "vllm",
                }
            ],
        }
        router.backends["b1"].list_models = AsyncMock(return_value=upstream_payload)
        router.backends["b2"].list_models = AsyncMock(return_value=upstream_payload)
        listing = await router.list_all_models()
        ids = [m["id"] for m in listing["data"]]
        assert ids.count("solo") == 1

    @pytest.mark.asyncio
    async def test_upstream_only_name_is_not_advertised(self) -> None:
        """W-3 in full: ambiguity is not the only way to 404.

        A vLLM upstream serves the pre-rename alias next to the current
        model ID, and this gateway declares only the latter. Before R-2,
        asking for the alias fell through to ``default_backend`` and
        answered 200, so listing it was true. After R-2 it is refused —
        so listing it would make ``/v1/models`` advertise a name that
        structurally 404s, which is the exact contradiction W-3 forbids
        and the one this branch would otherwise have introduced.
        """
        router = BackendRouter(
            routing_settings=RoutingSettings(
                enabled=True,
                default_backend="b1",
                backends={
                    "b1": BackendSettings(
                        url="http://localhost:1", models=[{"name": "current-name"}]
                    ),
                },
                tiers={"light": TierSettings(backend="b1")},
            ),
            vllm_settings=VLLMSettings(url="http://localhost:8000"),
        )
        router.backends["b1"].list_models = AsyncMock(
            return_value={
                "object": "list",
                "data": [
                    {
                        "id": "current-name",
                        "object": "model",
                        "created": 1,
                        "owned_by": "vllm",
                    },
                    {
                        "id": "legacy-alias",
                        "object": "model",
                        "created": 1,
                        "owned_by": "vllm",
                    },
                ],
            }
        )
        listing = await router.list_all_models()
        ids = {m["id"] for m in listing["data"]}
        assert "current-name" in ids
        assert "light" in ids
        assert "legacy-alias" not in ids, (
            "'legacy-alias' is served by the upstream but declared by no "
            f"backend, so the router 404s it. Got ids: {sorted(ids)}"
        )
        # And the invariant itself, stated once: everything advertised
        # can actually be routed.
        for model_id in ids:
            router.get_backend_for_model(model_id)

    @pytest.mark.asyncio
    async def test_legacy_single_backend_listing_is_not_narrowed(self) -> None:
        """The filter is scoped to multi-backend mode.

        Legacy mode routes every name to the one backend and keeps an
        empty by-name index, so filtering on that index would empty the
        listing rather than narrow it — the listing would stop describing
        a gateway that does route those names.
        """
        router = BackendRouter(
            routing_settings=RoutingSettings(enabled=False),
            vllm_settings=VLLMSettings(url="http://localhost:8000"),
        )
        router.backends["default"].list_models = AsyncMock(
            return_value={
                "object": "list",
                "data": [
                    {
                        "id": "whatever-upstream-serves",
                        "object": "model",
                        "created": 1,
                        "owned_by": "vllm",
                    }
                ],
            }
        )
        listing = await router.list_all_models()
        ids = {m["id"] for m in listing["data"]}
        assert ids == {"whatever-upstream-serves"}
        # Still routable in this mode — the listing stays true.
        assert router.get_backend_for_model("whatever-upstream-serves") is (
            router.backends["default"]
        )


# --------------------------------------------------------------------------
# T-models-advertise-side — W-3 in the other direction: every routable
# name is advertised
# --------------------------------------------------------------------------


class TestListModelsAdvertisesEveryRoutableName:
    """The advertised set *is* the routable set, not merely a subset of it.

    W-3 narrowed the listing one way -- nothing advertised may 404 -- and
    left the other direction open, recorded in ``list_all_models``' own
    docstring as "a declared name whose own upstream does not report it
    is routable without being advertised".

    That asymmetry is not hypothetical. It uses "some upstream reported
    it" as a proxy for "it exists", and the proxy is simply wrong for a
    backend that is not a model catalogue: ``ClaudeCodeBackend.
    list_models`` returns a hardcoded empty list, so the shipped
    ``claude-code-opus`` / ``claude-code-sonnet`` were fully routable and
    entirely absent from ``/v1/models``. Nor did a tier rescue them --
    no shipped tier maps to the ``claude_code`` backend -- so the
    docstring's own consolation ("the rest of the upstream's catalogue is
    reachable through the tier alias that resolves to it") was false for
    that backend.

    The rule that replaces the proxy is the one the file already applies
    to tier aliases: a name this gateway *declares* is advertised on this
    gateway's own authority. It does not weaken W-3 -- everything newly
    advertised is routable by construction, and the two filters that make
    a name unroutable (ambiguity, and upstream-only) both still hold the
    name out. These tests are the deliberate pair of
    ``TestListModelsMatchesRoutableSet``: that class measures "advertised
    is contained in routable", this one measures the converse.
    """

    @staticmethod
    def _declared_but_absent_router() -> BackendRouter:
        """One backend declaring two names; its upstream reports one.

        The general shape of the shipped ``claude_code`` defect with the
        vendor specifics removed: ``list_models`` under-reports what the
        gateway declares. ``claude_code`` is the extreme case of this --
        it reports nothing at all -- not a different case.
        """
        router = BackendRouter(
            routing_settings=RoutingSettings(
                enabled=True,
                # Deliberately not ``b1``. With the declaring backend also
                # being the default, a row stamped with the default would
                # be indistinguishable from a row stamped with the
                # declaring backend, and the attribution assertion below
                # would pass against a router that ignores declarations.
                default_backend="elsewhere",
                backends={
                    "b1": BackendSettings(
                        url="http://localhost:1",
                        models=[{"name": "reported"}, {"name": "declared-only"}],
                    ),
                    "elsewhere": BackendSettings(
                        url="http://localhost:2", models=[{"name": "somewhere-else"}]
                    ),
                },
                tiers={"light": TierSettings(backend="b1")},
            ),
            vllm_settings=VLLMSettings(url="http://localhost:8000"),
        )
        router.backends["b1"].list_models = AsyncMock(
            return_value={
                "object": "list",
                "data": [
                    {
                        "id": "reported",
                        "object": "model",
                        "created": 1_700_000_000,
                        "owned_by": "vllm",
                    }
                ],
            }
        )
        router.backends["elsewhere"].list_models = AsyncMock(
            return_value={"object": "list", "data": []}
        )
        return router

    @pytest.mark.asyncio
    async def test_declared_name_the_upstream_never_reports_is_advertised(
        self,
    ) -> None:
        """The gap class itself: routable **and** advertised, not one or
        the other."""
        router = self._declared_but_absent_router()
        listing = await router.list_all_models()
        ids = {m["id"] for m in listing["data"]}

        # Precondition stated through the public lookup, so "routable" is
        # measured rather than read off the config literal above.
        assert router.get_backend_name_for_model("declared-only") == "b1"
        assert "declared-only" in ids, (
            "'declared-only' routes but its backend's upstream never "
            f"reports it, so it was hidden from /v1/models. Got: {sorted(ids)}"
        )
        # The invariant in full, both directions at once, over whatever
        # the fixture happens to contain.
        routable = set(router._model_to_backend) | set(router._tier_to_backend)
        assert ids == routable, (
            f"advertised != routable. Only advertised: {sorted(ids - routable)}; "
            f"only routable: {sorted(routable - ids)}"
        )

    @pytest.mark.asyncio
    async def test_declared_only_row_is_attributed_and_openai_shaped(self) -> None:
        """A row nobody reported still has to be a well-formed Model row.

        ``backend`` is the field a client reads to learn where a name
        goes (R-7), and the four OpenAI-declared fields are the ones
        whose absence does not raise in ``openai-python`` but comes back
        as ``None`` on ``int`` / ``str``-declared attributes and breaks
        later and elsewhere.
        """
        router = self._declared_but_absent_router()
        listing = await router.list_all_models()
        row = next(m for m in listing["data"] if m["id"] == "declared-only")

        for field in ("id", "object", "created", "owned_by"):
            assert field in row, f"declared-only row is missing {field!r}"
        assert row["backend"] == router.get_backend_name_for_model("declared-only")
        # Advertised on this gateway's authority, exactly as a tier alias
        # is: no upstream vouched for this row, and ``created: 0`` is the
        # same "no creation time exists" sentinel rather than a plausible
        # fabricated timestamp.
        assert row["owned_by"] == "lexora"
        assert row["created"] == 0
        # The marker that makes the two lines above honest. ``owned_by:
        # "lexora"`` is stamped on rows for vendor-owned IDs as well
        # (``claude-sonnet-4-20250514``, ``gemini-3.1-pro-preview``), and
        # that is a statement about who asserts the row rather than a
        # claim of ownership *only* while ``type`` says the row is this
        # gateway's own assertion. Until this line the marker was
        # unfenced: the string is emitted in exactly one place and
        # deleting that line left the whole suite green, so any tidying
        # pass could drop it and turn five production rows into
        # upstream-looking claims that Lexora owns Anthropic's and
        # Google's models.
        assert row["type"] == "declared"

    @pytest.mark.asyncio
    async def test_a_reported_row_is_not_replaced_by_a_declared_one(self) -> None:
        """Fill the gap; do not overwrite what the upstream did say.

        ``reported`` is both declared and reported, so the upstream's own
        row is the truthful one and has to survive intact -- one row,
        carrying the vendor's ``created`` / ``owned_by``. Emitting the
        declared row unconditionally would blank both fields for every
        model an upstream actually described.
        """
        router = self._declared_but_absent_router()
        listing = await router.list_all_models()
        rows = [m for m in listing["data"] if m["id"] == "reported"]
        assert len(rows) == 1, f"expected one row for 'reported', got {rows}"
        assert rows[0]["created"] == 1_700_000_000
        assert rows[0]["owned_by"] == "vllm"
        # The other half of the ``type`` fence, and the half that carries
        # the meaning: what matters is the *distinction*, not the
        # marker's presence. A suite that only asserted ``type ==
        # "declared"`` on the declared row (see the sibling test above)
        # would still pass against an emit site that stamped ``type`` on
        # every row, which would erase the distinction while satisfying
        # the assertion. ``reported`` came from the upstream, so it must
        # carry no marker at all.
        assert "type" not in rows[0], (
            "'reported' was reported by the upstream, so it must carry no "
            "'type' key: the marker is what tells a row this gateway "
            f"asserts from one an upstream confirmed. Got: {rows[0]}"
        )

    @pytest.mark.asyncio
    async def test_ambiguous_declared_name_stays_out_even_if_unreported(self) -> None:
        """W-3 is not weakened: the converse is over the *routable* set,
        not the declared one.

        ``dup`` is declared by two backends, so it 404s (R-1a) and is
        absent from ``_model_to_backend``. Advertising every *declared*
        name -- the near-miss implementation -- would put it back into the
        listing, and neither upstream reports it here, so the existing
        W-3 test that drops it at the upstream row cannot see this.
        """
        router = BackendRouter(
            routing_settings=RoutingSettings(
                enabled=True,
                default_backend="b1",
                backends={
                    "b1": BackendSettings(
                        url="http://localhost:1",
                        models=[{"name": "dup"}, {"name": "solo"}],
                    ),
                    "b2": BackendSettings(
                        url="http://localhost:2", models=[{"name": "dup"}]
                    ),
                },
            ),
            vllm_settings=VLLMSettings(url="http://localhost:8000"),
        )
        empty = {"object": "list", "data": []}
        for name in ("b1", "b2"):
            router.backends[name].list_models = AsyncMock(return_value=empty)

        listing = await router.list_all_models()
        ids = {m["id"] for m in listing["data"]}
        assert "solo" in ids
        assert "dup" not in ids, (
            "'dup' is declared by b1 and b2, so the router 404s it. A "
            f"declared name is not the same as a routable one. Got: {sorted(ids)}"
        )
        for model_id in ids:
            router.get_backend_for_model(model_id)


# --------------------------------------------------------------------------
# R-7 — /v1/models attributes a row to the backend that declares it
# --------------------------------------------------------------------------


class TestListingAttributionMatchesDeclaration:
    """R-7: ``backend`` on a concrete row names the declaring backend.

    W-3's filter asks whether a name is routable *anywhere*. That is the
    right question for "should this row exist" and the wrong one for
    "whose row is this". When two backends proxy one upstream, both
    upstreams report both names, so the first backend in iteration order
    passes the global check for a name the *second* one declares, stamps
    its own name onto the row, and the dedupe guard then drops the
    authoritative row as a repeat. The listing ends up asserting a
    routing decision the router does not make.

    Narrowing that check to "does this backend declare it" answers both
    questions with one test, and makes the dedupe redundant across
    backends rather than load-bearing.

    The invariant is written as a sweep over every concrete row rather
    than as an assertion about a chosen id: a single-row assertion goes
    stale the moment the fixture or the iteration order is rearranged,
    while "each row agrees with the router" keeps measuring the same
    thing whatever the listing contains.
    """

    @staticmethod
    def _shared_upstream_router() -> BackendRouter:
        """Two backends at one URL, each declaring a different model.

        This is the shape the shipped config already has (three vLLM
        backends pointing at one server), minus the ambiguity that hides
        the defect there by dropping both names before attribution runs.
        """
        router = BackendRouter(
            routing_settings=RoutingSettings(
                enabled=True,
                default_backend="alpha",
                backends={
                    "alpha": BackendSettings(
                        url="http://localhost:1", models=[{"name": "ModelA"}]
                    ),
                    "beta": BackendSettings(
                        url="http://localhost:1", models=[{"name": "ModelB"}]
                    ),
                },
                tiers={
                    "fast": TierSettings(backend="alpha"),
                    "slow": TierSettings(backend="beta"),
                },
            ),
            vllm_settings=VLLMSettings(url="http://localhost:8000"),
        )
        # One upstream, so both backends report the whole catalogue.
        upstream = {
            "object": "list",
            "data": [
                {"id": "ModelA", "object": "model", "created": 1, "owned_by": "vllm"},
                {"id": "ModelB", "object": "model", "created": 1, "owned_by": "vllm"},
            ],
        }
        for name in ("alpha", "beta"):
            router.backends[name].list_models = AsyncMock(return_value=upstream)
        return router

    @pytest.mark.asyncio
    async def test_every_concrete_row_agrees_with_the_router(self) -> None:
        router = self._shared_upstream_router()
        listing = await router.list_all_models()
        concrete = [m for m in listing["data"] if m.get("type") != "tier"]
        assert concrete, "fixture produced no concrete rows to check"
        for row in concrete:
            assert row["backend"] == router.get_backend_name_for_model(row["id"]), (
                f"/v1/models says '{row['id']}' is served by "
                f"'{row['backend']}', but the router sends it to "
                f"'{router.get_backend_name_for_model(row['id'])}'."
            )

    @pytest.mark.asyncio
    async def test_declaring_backends_row_is_not_dropped_as_a_duplicate(self) -> None:
        """The dedupe must not be able to keep the wrong copy.

        Collapsing the four rows a shared upstream produces down to two
        is only an improvement if the two that survive are the true ones;
        keeping one arbitrary row per id trades an ambiguous listing for
        a confidently wrong one.
        """
        router = self._shared_upstream_router()
        listing = await router.list_all_models()
        rows = [m for m in listing["data"] if m["id"] == "ModelB"]
        assert len(rows) == 1, f"expected one row for ModelB, got {rows}"
        assert rows[0]["backend"] == "beta"

    @pytest.mark.asyncio
    async def test_unroutable_log_is_not_used_for_a_name_that_is_listed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Skipping another backend's copy is dedupe, not an unroutable filter.

        ``list_models_unroutable_filtered`` means "no backend can serve
        this name". Emitting it for a name that appears in the same
        response would make the log state something about the gateway
        that the response contradicts -- the defect this branch is
        against, relocated from the API surface to the log surface.
        """
        router = self._shared_upstream_router()
        with caplog.at_level(logging.DEBUG):
            listing = await router.list_all_models()
        listed = {m["id"] for m in listing["data"]}
        for record in caplog.records:
            if "list_models_unroutable_filtered" not in record.message:
                continue
            for model_id in listed:
                assert model_id not in record.message, (
                    f"'{model_id}' is in the response and also logged as "
                    f"unroutable: {record.message}"
                )


# --------------------------------------------------------------------------
# 404 response body shape (OpenAI + Anthropic)
# --------------------------------------------------------------------------


@pytest.fixture
def app_with_shared_model_router(monkeypatch: pytest.MonkeyPatch):
    """FastAPI app whose router 404s 'dup-model' and unknown names.

    Kept as a fixture rather than repeating the boilerplate in each 404-
    shape test: the request-side surface is what these tests measure,
    not the setup.
    """
    from lexora import config as config_module

    def _fake_settings() -> "config_module.Settings":  # type: ignore[name-defined]
        return config_module.Settings(
            routing=RoutingSettings(
                enabled=True,
                default_backend="b1",
                backends={
                    "b1": BackendSettings(
                        url="http://localhost:1", models=[{"name": "dup-model"}]
                    ),
                    "b2": BackendSettings(
                        url="http://localhost:2", models=[{"name": "dup-model"}]
                    ),
                },
                tiers={"light": TierSettings(backend="b1", model="dup-model")},
            )
        )

    monkeypatch.setattr(config_module, "create_settings", _fake_settings)
    app = create_app(_fake_settings())
    # The lifespan builds the real router with the fake settings.
    with TestClient(app) as client:
        yield client


class TestNotFoundResponseShape:
    """The API shape for R-1a / R-2 refusals."""

    def test_openai_endpoint_404_shape_for_unknown(
        self, app_with_shared_model_router: TestClient
    ) -> None:
        response = app_with_shared_model_router.post(
            "/v1/chat/completions",
            json={
                "model": "no-such-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert response.status_code == 404, response.text
        body = response.json()
        assert body == {
            "error": {
                # The message is human-readable; we assert its structural
                # position, not the exact wording.
                "message": body["error"]["message"],
                "type": "invalid_request_error",
                "param": "model",
                "code": "model_not_found",
            }
        }
        assert "no-such-model" in body["error"]["message"]

    def test_openai_endpoint_404_shape_for_ambiguous(
        self, app_with_shared_model_router: TestClient
    ) -> None:
        response = app_with_shared_model_router.post(
            "/v1/chat/completions",
            json={
                "model": "dup-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert response.status_code == 404, response.text
        body = response.json()
        # Ambiguous and unknown share ``code: model_not_found`` on
        # purpose — clients have no useful branch to make between them
        # (msg-082 objection B).
        assert body["error"]["code"] == "model_not_found"
        # The message must tell the operator both offenders.
        assert "b1" in body["error"]["message"]
        assert "b2" in body["error"]["message"]

    def test_anthropic_messages_404_uses_anthropic_shape(
        self, app_with_shared_model_router: TestClient
    ) -> None:
        response = app_with_shared_model_router.post(
            "/v1/messages",
            json={
                "model": "no-such-model",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert response.status_code == 404, response.text
        body = response.json()
        # Anthropic-shaped: the SDK parses this envelope natively; an
        # OpenAI-shaped 404 on /v1/messages would be the one endpoint's
        # response the SDK could not typecheck.
        assert body["type"] == "error"
        assert body["error"]["type"] == "not_found_error"
        assert "no-such-model" in body["error"]["message"]


# --------------------------------------------------------------------------
# R-9 — the 404 dialect is decided by the matched route, not the URL string
# --------------------------------------------------------------------------


@pytest.fixture
def app_with_shared_model_router_app(monkeypatch: pytest.MonkeyPatch):
    """The same app as ``app_with_shared_model_router``, un-wrapped.

    The tests below need to build several clients with different ASGI
    ``root_path`` / ``path`` combinations against one app, so the fixture
    hands back the app rather than a single client.
    """
    from lexora import config as config_module

    def _fake_settings() -> "config_module.Settings":  # type: ignore[name-defined]
        return config_module.Settings(
            routing=RoutingSettings(
                enabled=True,
                default_backend="b1",
                backends={
                    "b1": BackendSettings(
                        url="http://localhost:1", models=[{"name": "dup-model"}]
                    ),
                    "b2": BackendSettings(
                        url="http://localhost:2", models=[{"name": "dup-model"}]
                    ),
                },
                tiers={"light": TierSettings(backend="b1", model="dup-model")},
            )
        )

    monkeypatch.setattr(config_module, "create_settings", _fake_settings)
    return create_app(_fake_settings())


def _anthropic_404(client: TestClient, path: str) -> dict:
    response = client.post(
        path,
        json={
            "model": "no-such-model",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 404, response.text
    return response.json()


def _openai_404(client: TestClient, path: str) -> dict:
    response = client.post(
        path,
        json={
            "model": "no-such-model",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 404, response.text
    return response.json()


class TestNotFoundShapeUnderAsgiRootPath:
    """R-9: a prefix in ``scope["path"]`` must not change the body shape.

    Three deployment shapes are reachable without a proxy in the test,
    because the only thing a proxy changes is what the ASGI server puts
    in ``scope``:

    * A — no ``root_path``. ``scope["path"] == "/v1/messages"``.
    * B — ``root_path="/api"`` and a proxy that strips the prefix before
      forwarding. ``scope["path"]`` is still ``"/v1/messages"``.
    * C — ``root_path="/api"`` and a proxy that forwards the prefix.
      ``scope["path"] == "/api/v1/messages"``.

    In all three the router matches the same endpoint, because Starlette
    strips ``root_path`` for routing (``get_route_path``). Only the URL
    string differs, and only in C. Deciding the dialect from the URL
    string therefore answers C wrong — with the one 404 shape the
    ``anthropic`` SDK cannot typecheck.

    ``request.url.path`` is worth naming precisely: it does not prepend
    ``root_path``. Starlette 0.50.0 builds it from ``scope["path"]``
    verbatim. What puts the prefix there is the server, when the proxy
    in front of it forwards the prefix rather than stripping it.
    """

    def test_prefix_forwarding_proxy_still_gets_the_anthropic_shape(
        self, app_with_shared_model_router_app
    ) -> None:
        """Shape C on /v1/messages. The detector."""
        with TestClient(app_with_shared_model_router_app, root_path="/api") as client:
            body = _anthropic_404(client, "/api/v1/messages")
        assert body["type"] == "error"
        assert body["error"]["type"] == "not_found_error"
        assert "no-such-model" in body["error"]["message"]

    def test_prefix_forwarding_proxy_keeps_the_openai_shape_elsewhere(
        self, app_with_shared_model_router_app
    ) -> None:
        """Shape C on /v1/chat/completions.

        The pin against over-matching. "Answer every 404 in the Anthropic
        envelope" satisfies the test above; it fails this one. The two
        together say the predicate must widen to reach /v1/messages under
        a prefix without also reaching the OpenAI endpoints.
        """
        with TestClient(app_with_shared_model_router_app, root_path="/api") as client:
            body = _openai_404(client, "/api/v1/chat/completions")
        assert "type" not in body
        assert body["error"]["code"] == "model_not_found"
        assert body["error"]["type"] == "invalid_request_error"

    def test_prefix_stripping_proxy_is_unchanged(
        self, app_with_shared_model_router_app
    ) -> None:
        """Shape B — the deployment that was already correct stays correct."""
        with TestClient(app_with_shared_model_router_app, root_path="/api") as client:
            assert _anthropic_404(client, "/v1/messages")["type"] == "error"
            assert "type" not in _openai_404(client, "/v1/chat/completions")

    def test_no_root_path_is_unchanged(
        self, app_with_shared_model_router_app
    ) -> None:
        """Shape A — the shipped deployment. ``deploy/lexora.service`` runs
        ``python -m lexora.main``, whose ``uvicorn.run`` takes no
        ``root_path``, so this is the shape in production today.
        """
        with TestClient(app_with_shared_model_router_app) as client:
            assert _anthropic_404(client, "/v1/messages")["type"] == "error"
            assert "type" not in _openai_404(client, "/v1/chat/completions")
