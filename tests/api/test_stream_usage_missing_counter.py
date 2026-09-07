"""T-streaming-ledger-row §5(c): the silence that should have been a row.

An empty `UsageSink` at a terminal exit has FOUR causes and, before this
change, they were indistinguishable after the fact -- no row, no log, no
counter. Three of the four are correct and are documented as correct in
`get_cost_tracker`'s coverage table: a verbatim-relay backend, a client who
hangs up before the upstream's final frame, and a stream where the count
genuinely never arrived. The fourth is the defect: a backend that parses the
count itself, whose parser has stopped matching what the upstream sends.

The separating fact was already in the generator's hand -- whether the stream
reached normal completion, versus arriving in one of the `except` clauses --
so the remedy is an observable and not an alarm. Raising was rejected
explicitly: at the read site it would break every healthy stream (a
non-final event carries no usage block by design), and at the terminal site
it would fire on all three legitimate causes, turning a client disconnect
into a 500.

The shipped shape: one bit on the backend (`Backend.fills_usage_sink`), a
`completed_normally` local set where each `stream_generator` already marks
success, one predicate (`routes._record_missing_stream_usage`) called from
all three sites, and one prometheus `Counter`
(`lexora_stream_usage_missing_total`, labelled `backend` / `endpoint`).

**The bit is about the BACKEND, never about the wire format.** That is what
keeps this from re-introducing the coupling this design refused: a handler
reading `fills_usage_sink` learns that an implementation parses, not what it
parses, so the router still holds no upstream format.

Two discriminators that look right and are not, both measured here at
`bdfc185` rather than assumed:

- `error_passthrough` is NOT "relays bytes verbatim". It is an error-shape
  flag, `config.ERROR_PASSTHROUGH_TYPES` is exactly `{"anthropic"}`, and
  `factory.py` hands it to `AnthropicBackend` alone -- which is one of the
  three backends that DO fill the sink. Reusing it would mark the
  best-instrumented backend as the un-instrumented one: not a different
  proposition, an anti-correlated one. `test_error_passthrough_is_not_the
  _discriminator` holds that measurement so it cannot quietly stop being
  true.
- the `usage_sink` parameter is not a discriminator either: all five backends
  take `usage_sink: UsageSink | None = None`, and only `anthropic`,
  `claude_code` and `gemini` assign to it. Before this change that fact was
  carried only by two docstrings.
  `test_the_declaration_matches_which_backends_write_the_sink` holds it.

Why the positive and each negative is its OWN named test rather than one
case with four branches: the four legitimate causes of an empty sink are the
entire reason this mechanism exists, so each has to be able to red alone. A
single case with four assertions cannot tell you which one stopped working.

Hazard, and it is why nothing here asserts an absolute: a module-level
`Counter` is process-global, so its value depends on what ran before it in
this session. Every assertion below is a delta across one request, read
through `prometheus_client.REGISTRY.get_sample_value`. That reader was an
open question in the spec -- "I have not verified it works under this
suite's fixtures". Measured: it does, it is already how
`test_stream_disconnect_accounting.py` reads `lexora_active_requests`, and a
label set that has never been touched reads `None` rather than raising,
which is why `_missing_count` maps `None` to `0.0`. `BACKEND_NAME` here is
deliberately not the `"frontier"` other files use, so no other test in the
session can move the sample these deltas are taken over.

RED BEFORE GREEN. Reverting `src/` alone to `bdfc185` and leaving this file
in place: **6 red / 15 green here**, and the whole suite in that state is
**6 failed / 693 passed** -- so the 693 is 15 of this file's own cases plus
the 678 that existed before, and NOT ONE of those 678 moves. The 6 are the
three
`test_a_declaring_backend_that_completes_with_an_empty_sink_is_counted`
cases, `test_a_messages_stream_error_with_an_empty_sink_IS_counted`, and
both cases in `TestTheDeclarationIsNotAnExistingFlagInDisguise` that read
the new attribute. The 15 green are fences rather than detectors, and
knowing WHY matters: with no counter registered,
`REGISTRY.get_sample_value` returns `None` for every label set, so "is not
counted" was vacuously true before this change and now has to stay true
against a counter that actually exists. That the other 678 all stay green is
the point of the run: the suite was blind to this whole class.

Full suite measured by me at both ends: **678 at `bdfc185`, 699 with this
change**, i.e. +21, exactly this file's case count, so nothing else moved.

MUTATION TABLE. One conjunct per row, each a single-site edit against the
finished tree, restored and re-verified byte-identical before the next.
Counts are this file alone, 21 cases, win32 / CPython 3.11.

- Drop the `fills_usage_sink` conjunct from `_record_missing_stream_usage`
  (delete its `if not ...: return`): **3 red / 18 green** -- the three
  `test_a_relay_backend_that_writes_nothing_is_not_counted` cases, and
  nothing else. A backend that never claimed to parse would then be counted
  as a defect, which is exactly the false alarm the relays must not produce.
- Drop the `completed_normally` conjunct: **5 red / 16 green** -- the three
  `test_a_client_disconnect_mid_stream_is_not_counted` cases and both
  `test_a_stream_error_with_an_empty_sink_is_not_counted` cases. The row
  that matters most: without it a hang-up becomes an alarm, which is the
  failure "fail loudly" was rejected for.
- Drop the empty-sink conjunct: **4 red / 17 green** -- the three
  `test_a_filled_sink_opens_a_row_and_counts_nothing` cases plus
  `test_a_backend_error_mid_stream_after_a_count_is_not_counted[messages]`.
  Every ordinary billed stream would then be counted as a silence.
- Delete the `_record_missing_stream_usage(...)` call from `chat_completions`
  only (1 of 3 occurrences): **1 red / 20 green** -- `[chat_completions]` of
  `test_a_declaring_backend_that_completes_with_an_empty_sink_is_counted`,
  and nothing else. The three handlers are three wirings, not one assertion
  counted three times.
- Set `completed_normally = True` before the `try` in `chat_completions`
  instead of at the success mark: **2 red / 19 green** -- the
  `[chat_completions]` cases of `test_a_client_disconnect_mid_stream_is_not
  _counted` and `test_a_stream_error_with_an_empty_sink_is_not_counted`. The
  flag has to be set where success is marked, not merely to exist.
- Leave `fills_usage_sink` at its `False` default on `AnthropicBackend`:
  **2 red / 19 green** -- both cases in
  `TestTheDeclarationIsNotAnExistingFlagInDisguise` that read it.

FIVE OF THE SIX PREDICTED COUNTS WERE WRONG, and the misses are worth more
than the hits because each one located a real asymmetry. Predicted-then-
measured: M1 4->3, M2 3->5, M3 4->4 but over a DIFFERENT set, M4 1->1
(right), M5 1->2, M6 1->2.

- M1: I predicted `test_error_passthrough_is_not_the_discriminator` would
  red. It does not, and it should not -- it reads class attributes and never
  drives the predicate. I had conflated the conjunct with the declaration.
- M3: I predicted all three `..._after_a_count_is_not_counted` cases. Only
  `[messages]` reds. On the two OpenAI-family endpoints that case is held
  TWICE -- the sink is filled AND the error escapes to an `except` clause --
  so dropping one conjunct leaves them green. `[messages]` rests on the
  empty-sink conjunct alone, because its converter absorbs the error and the
  stream completes normally there. That is the same measured asymmetry
  `test_a_messages_stream_error_with_an_empty_sink_IS_counted` records, and
  it surfaced twice from two directions before I understood it.
- M2 and M5: undercounted because I wrote the table before adding
  `test_a_stream_error_with_an_empty_sink_is_not_counted`, which is a second
  detector over `completed_normally`. Two detectors over one conjunct is not
  redundancy here: one drives a cancellation, the other an ordinary
  exception, and those reach the flag by different clauses.
- M6: reds both declaration cases, not one, because
  `test_error_passthrough_is_not_the_discriminator` also asserts
  `AnthropicBackend.fills_usage_sink is True` -- the anti-correlation is only
  a statement about anthropic if anthropic really does declare.

RESTORE RECEIPTS. Every mutation was restored by byte comparison
(`filecmp.cmp(..., shallow=False)`) against a snapshot of the SAME worktree
file taken immediately before the edit -- never against `git show HEAD:` --
and all six came back byte-identical. The reason for the snapshot rule, and
a correction to how it was stated:

Measured at the byte level in this clone at `bdfc185`, `src/lexora/api
/routes.py` is LF in the worktree AND LF in the committed blob, while
`tests/api/test_ledger_coverage.py` is CRLF in the worktree and LF in the
blob. So the mismatch is real but it is per-file -- it is not "the blob is
always LF and the worktree always CRLF", and a rule resting on that premise
would be wrong on `routes.py`, which is the file mutated five times here. A
same-worktree snapshot rests on no premise about line endings at all, which
is why it is the rule rather than an application of one.

Two instruments, both used and one discarded: `grep -c $'\\r$'` under Git
Bash on this host reported every file as fully CRLF, including files that
contain no CR byte at all. It is not a line-ending detector here. The counts
above are `bytes.count(b"\\r\\n")` against `bytes.count(b"\\n")`. The
mutation driver builds each pattern against the newline the target file
actually uses, asserts the pattern was found, and asserts the write changed
bytes -- so a pattern that misses is an error and can never come back as a
clean green, which is the failure this whole apparatus exists to prevent.
"""

import asyncio
import inspect
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from lexora.api.routes import (
    get_backend,
    get_backend_router,
    get_cost_tracker,
    get_metrics_collector,
    get_rate_limiter,
    get_retry_handler,
    get_stats_collector,
    is_rate_limit_enabled,
    router,
)
from lexora.backends.anthropic import AnthropicBackend
from lexora.backends.base import Backend, BackendError
from lexora.backends.claude_code import ClaudeCodeBackend
from lexora.backends.gemini import GeminiBackend
from lexora.backends.openai_compatible import OpenAICompatibleBackend
from lexora.backends.vllm import VLLMBackend
from lexora.config import ERROR_PASSTHROUGH_TYPES
from lexora.services.rate_limiter import RateLimiter
from lexora.services.retry_handler import RetryHandler
from lexora.services.stats import StatsCollector

REQUESTED = "heavy"
RESOLVED = "claude-fable-5"

# ★ Deliberately not the `"frontier"` that `test_streaming_ledger_row.py` and
# `test_stream_disconnect_accounting.py` use. The counter is a process
# global; giving this file its own label value means no other test in the
# session can move the sample these deltas are read over, so a delta here is
# this file's request and nothing else.
BACKEND_NAME = "sink-declaring-backend-under-test"
USER_ID = "user-42"

PROMPT_TOKENS = 137
COMPLETION_TOKENS = 42

# One well-formed OpenAI SSE chunk. `/v1/chat/completions` and
# `/v1/completions` forward it verbatim; `/v1/messages` parses it through
# `anthropic_stream_from_openai`, which is why it is a real chunk.
CHUNK = (
    b"data: "
    + json.dumps(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "choices": [
                {"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}
            ],
        }
    ).encode()
    + b"\n\n"
)

CHAT_BODY = {
    "model": REQUESTED,
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": True,
    "user": USER_ID,
}
COMPLETIONS_BODY = {
    "model": REQUESTED,
    "prompt": "Hi",
    "stream": True,
    "user": USER_ID,
}
MESSAGES_BODY = {
    "model": REQUESTED,
    "max_tokens": 16,
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": True,
    "metadata": {"user_id": USER_ID},
}

ROUTES = [
    ("/v1/chat/completions", CHAT_BODY),
    ("/v1/completions", COMPLETIONS_BODY),
    ("/v1/messages", MESSAGES_BODY),
]
ROUTE_IDS = ["chat_completions", "completions", "messages"]


def _missing_count(endpoint: str) -> float:
    """Current `lexora_stream_usage_missing_total` for this file's labels.

    Read through the public registry and only ever compared as a delta: a
    module-level `Counter` is a process global and an absolute would depend
    on what ran before this test in the same session. `None` -- the label set
    has never been incremented -- is the zero.
    """
    value = REGISTRY.get_sample_value(
        "lexora_stream_usage_missing_total",
        {"backend": BACKEND_NAME, "endpoint": endpoint},
    )
    return 0.0 if value is None else value


def _backend(
    *,
    declares: bool,
    fills: bool,
    passthrough: bool = False,
    raise_after_first_chunk: bool = False,
) -> MagicMock:
    """A backend double whose declaration and behaviour are set separately.

    `declares` and `fills` are two knobs on purpose: the whole mechanism is
    about the case where they DISAGREE, so a double that could not express
    "claims to parse, wrote nothing" could not drive it.

    ★ `fills_usage_sink` is always set explicitly, never left off. An unset
    attribute on a `MagicMock` is a truthy `MagicMock`, so omitting it would
    silently make every double a declaring backend and the non-declaring
    cases would be testing nothing. Same reason `error_passthrough` is set
    explicitly in `test_stream_disconnect_accounting.py`.

    The factory takes `usage_sink=None` by default so a wiring mistake reds
    on the assertion rather than on a `TypeError`, which would say nothing
    about what was counted.
    """

    def _factory(_request: dict, usage_sink: Any = None) -> AsyncIterator[bytes]:
        async def gen() -> AsyncIterator[bytes]:
            yield CHUNK
            if fills and usage_sink is not None:
                usage_sink.prompt_tokens = PROMPT_TOKENS
                usage_sink.completion_tokens = COMPLETION_TOKENS
            if raise_after_first_chunk:
                raise BackendError("upstream went away mid-stream")

        return gen()

    backend = MagicMock()
    backend.chat_completions_stream = MagicMock(side_effect=_factory)
    backend.completions_stream = MagicMock(side_effect=_factory)
    backend.error_passthrough = passthrough
    backend.fills_usage_sink = declares
    return backend


def _app(backend: MagicMock, cost_tracker: Any) -> FastAPI:
    backend_router = MagicMock()
    backend_router.get_backend_for_model = MagicMock(return_value=backend)
    backend_router.resolve_model = MagicMock(return_value=RESOLVED)
    backend_router.get_backend_name_for_model = MagicMock(return_value=BACKEND_NAME)
    backend_router.is_tier = MagicMock(return_value=True)

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_backend] = lambda: backend
    app.dependency_overrides[get_backend_router] = lambda: backend_router
    app.dependency_overrides[get_stats_collector] = lambda: StatsCollector()
    app.dependency_overrides[get_retry_handler] = lambda: RetryHandler(
        max_retries=1, base_delay=0.01, max_delay=0.1, jitter=False
    )
    app.dependency_overrides[get_rate_limiter] = lambda: RateLimiter(
        default_rate=1000.0, default_burst=1000
    )
    app.dependency_overrides[is_rate_limit_enabled] = lambda: False
    app.dependency_overrides[get_metrics_collector] = lambda: None
    app.dependency_overrides[get_cost_tracker] = lambda: cost_tracker
    return app


def _client(backend: MagicMock, cost_tracker: Any) -> TestClient:
    return TestClient(_app(backend, cost_tracker))


class TestTheDefectIsCounted:
    """The one case out of four that the counter exists to separate."""

    @pytest.mark.parametrize(("endpoint", "body"), ROUTES, ids=ROUTE_IDS)
    def test_a_declaring_backend_that_completes_with_an_empty_sink_is_counted(
        self, endpoint: str, body: dict
    ) -> None:
        """Declares it parses, finished cleanly, wrote nothing => exactly one.

        This is the shape a moved upstream wire format produces: the parser
        matches nothing, the sink stays 0/0, the row guard opens no row, and
        before this change the request left no trace at all.
        """
        tracker = MagicMock()
        before = _missing_count(endpoint)

        with _client(_backend(declares=True, fills=False), tracker) as client:
            response = client.post(endpoint, json=body)

        assert response.status_code == 200
        # Exactly one, not "at least one": a `finally` runs once per
        # generator, and a double-count would be a second defect.
        assert _missing_count(endpoint) - before == 1.0
        # ...and no row was invented to go with it. Nothing here estimates.
        assert tracker.record.call_count == 0


class TestTheThreeLegitimateSilencesAreNotCounted:
    """Each legitimate cause of an empty sink, one detector each."""

    @pytest.mark.parametrize(("endpoint", "body"), ROUTES, ids=ROUTE_IDS)
    def test_a_relay_backend_that_writes_nothing_is_not_counted(
        self, endpoint: str, body: dict
    ) -> None:
        """`openai_compatible` / `vllm` in miniature: never declared, so never counted.

        Cause 1 of the three correct ones. The relays hold no parsed object,
        which is why they need no special case anywhere else either -- and
        why the discriminator has to be the declaration and not the fact of
        an empty sink.
        """
        tracker = MagicMock()
        before = _missing_count(endpoint)

        with _client(_backend(declares=False, fills=False), tracker) as client:
            response = client.post(endpoint, json=body)

        assert response.status_code == 200
        assert _missing_count(endpoint) - before == 0.0
        assert tracker.record.call_count == 0

    @pytest.mark.parametrize(("endpoint", "body"), ROUTES, ids=ROUTE_IDS)
    async def test_a_client_disconnect_mid_stream_is_not_counted(
        self, endpoint: str, body: dict
    ) -> None:
        """Cause 2, and the one that matters most.

        Usage arrives in the upstream's final frame, so a client who hangs up
        first leaves the sink empty on a declaring backend -- the same
        observable state as the defect. Only `completed_normally` separates
        them, and getting this wrong turns every hang-up into an alarm, which
        is the failure mode "fail loudly" was rejected for.

        Driven over raw ASGI rather than `TestClient` for the reason
        `test_stream_disconnect_accounting.py` gives: `TestClient` drives the
        app to completion and cannot express "hang up after the first chunk".
        """
        started = asyncio.Event()
        backend = _hanging_backend(started)
        app = _app(backend, None)
        before = _missing_count(endpoint)

        sent = await _post_and_disconnect(app, endpoint, body)

        # The disconnect really landed mid-stream rather than racing the end.
        assert sent[0]["type"] == "http.response.start"
        assert sent[0]["status"] == 200
        assert any(m["type"] == "http.response.body" and m.get("body") for m in sent)
        assert started.is_set()

        assert _missing_count(endpoint) - before == 0.0

    @pytest.mark.parametrize(
        ("endpoint", "body", "propagates"),
        [
            ("/v1/chat/completions", CHAT_BODY, True),
            ("/v1/completions", COMPLETIONS_BODY, True),
            # ★ False, measured, not assumed -- see
            # `test_a_messages_stream_error_with_an_empty_sink_IS_counted`.
            ("/v1/messages", MESSAGES_BODY, False),
        ],
        ids=ROUTE_IDS,
    )
    def test_a_backend_error_mid_stream_after_a_count_is_not_counted(
        self, endpoint: str, body: dict, propagates: bool
    ) -> None:
        """Cause 3: the stream failed after the upstream had sent a count.

        The sink is FILLED here and the row is still owed -- billing covers
        every terminal exit -- so this case rests on the empty-sink conjunct:
        drop it and a perfectly ordinary billed stream gets counted as a
        silence.

        `propagates` is a measurement about the endpoint, not a preference:
        `/v1/messages` runs its bytes through
        `anthropic_stream_from_openai`, whose `except Exception` yields an
        `error` SSE event and returns, so the failure never reaches the
        caller as an exception. Writing that into the parametrisation rather
        than into a `try` keeps it visible.
        """
        tracker = MagicMock()
        before = _missing_count(endpoint)

        with _client(
            _backend(declares=True, fills=True, raise_after_first_chunk=True),
            tracker,
        ) as client:
            if propagates:
                with pytest.raises(BackendError):
                    client.post(endpoint, json=body)
            else:
                assert client.post(endpoint, json=body).status_code == 200

        assert _missing_count(endpoint) - before == 0.0
        # The row is still opened: an error does not un-spend the tokens.
        assert tracker.record.call_count == 1

    @pytest.mark.parametrize(
        ("endpoint", "body"),
        [("/v1/chat/completions", CHAT_BODY), ("/v1/completions", COMPLETIONS_BODY)],
        ids=["chat_completions", "completions"],
    )
    def test_a_stream_error_with_an_empty_sink_is_not_counted(
        self, endpoint: str, body: dict
    ) -> None:
        """The harder half of cause 3: it failed AND it billed nothing.

        This is the case the empty-sink conjunct cannot hold, so it is
        `completed_normally` alone that separates a failed stream from the
        defect. The two OpenAI-family endpoints let the `BackendError` reach
        their `except` clauses, so the success mark is never run and the flag
        stays False.
        """
        tracker = MagicMock()
        before = _missing_count(endpoint)

        with _client(
            _backend(declares=True, fills=False, raise_after_first_chunk=True),
            tracker,
        ) as client:
            with pytest.raises(BackendError):
                client.post(endpoint, json=body)

        assert _missing_count(endpoint) - before == 0.0
        assert tracker.record.call_count == 0

    def test_a_messages_stream_error_with_an_empty_sink_IS_counted(self) -> None:
        """★ A KNOWN GAP, measured this turn and returned to the thread.

        `/v1/messages` is the one site where the spec's fourth negative --
        "declaring backend + `BackendError` => no increment" -- does NOT
        hold, and the reason is in that endpoint's own converter rather than
        in this mechanism. `anthropic_stream_from_openai` catches every
        `Exception` mid-stream, emits an Anthropic `error` event and
        `return`s, because that is how an Anthropic stream reports a
        mid-stream failure to its client. From `stream_generator`'s side the
        iteration therefore ends *normally*: the success mark runs,
        `completed_normally` becomes True, and a failure that billed nothing
        is counted as the defect.

        Not repaired here, and deliberately so. Closing it means giving the
        converter a way to tell its caller "I absorbed a failure" -- a fifth
        piece, in a file this increment does not otherwise touch, changing
        what `/v1/messages` reports on a path unrelated to accounting. That
        is a design decision about the compat layer, not implementation of
        this one, and it is the thread's to take.

        The case is written as an assertion of what actually happens rather
        than left unwritten: an unwritten gap is one nobody can see has
        widened, and if the converter is ever changed to re-raise, this reds
        and says exactly which decision was taken.

        Bound: the false positive needs a mid-stream failure on a declaring
        backend that had sent no usage at all, on `/v1/messages` only. It
        over-counts; it can never suppress a real defect.
        """
        tracker = MagicMock()
        before = _missing_count("/v1/messages")

        with _client(
            _backend(declares=True, fills=False, raise_after_first_chunk=True),
            tracker,
        ) as client:
            assert client.post("/v1/messages", json=MESSAGES_BODY).status_code == 200

        assert _missing_count("/v1/messages") - before == 1.0
        assert tracker.record.call_count == 0


class TestABilledStreamCountsNothing:
    """The ordinary case: a row opens, and the silence counter does not move."""

    @pytest.mark.parametrize(("endpoint", "body"), ROUTES, ids=ROUTE_IDS)
    def test_a_filled_sink_opens_a_row_and_counts_nothing(
        self, endpoint: str, body: dict
    ) -> None:
        """The counter and the row are mutually exclusive by construction.

        The emit predicate is the exact negation of the row-opening guard on
        its third conjunct, so exactly one of the two can fire per request.
        Asserting both halves in one case is what makes that a measurement
        rather than a claim.
        """
        tracker = MagicMock()
        before = _missing_count(endpoint)

        with _client(_backend(declares=True, fills=True), tracker) as client:
            response = client.post(endpoint, json=body)

        assert response.status_code == 200
        assert _missing_count(endpoint) - before == 0.0
        assert tracker.record.call_count == 1
        assert tracker.record.call_args.kwargs["tokens_input"] == PROMPT_TOKENS
        assert tracker.record.call_args.kwargs["tokens_output"] == COMPLETION_TOKENS


class TestTheDeclarationIsNotAnExistingFlagInDisguise:
    """The two discriminators that look right and are not, held as tests.

    Both were re-measured this turn rather than taken from the spec. They are
    fences and not detectors -- they were already true before this change --
    but the whole point of the declaration is that neither existing fact can
    stand in for it, and a fence is how that stops being an argument someone
    has to re-derive.
    """

    #: The three that assign to the sink, measured at `bdfc185`:
    #: `anthropic.py:528/575`, `gemini.py:619/623`, `claude_code.py:485-488`.
    WRITERS = (AnthropicBackend, ClaudeCodeBackend, GeminiBackend)
    NON_WRITERS = (OpenAICompatibleBackend, VLLMBackend)

    def test_the_declaration_matches_which_backends_write_the_sink(self) -> None:
        """Three declare, two do not -- and the default is False.

        The default matters as much as the three: a sixth backend added
        without thinking about the sink is silently correct rather than
        silently alarming.
        """
        assert Backend.fills_usage_sink is False
        for cls in self.WRITERS:
            assert cls.fills_usage_sink is True, cls.__name__
        for cls in self.NON_WRITERS:
            assert cls.fills_usage_sink is False, cls.__name__

    def test_error_passthrough_is_not_the_discriminator(self) -> None:
        """It is anti-correlated, not merely different.

        `error_passthrough` is an error-shape flag about forwarding upstream
        4xx/5xx, it is implemented on exactly one backend, and that backend
        is `anthropic` -- one of the three that DO fill the sink. Using it as
        a stand-in would mark the best-instrumented backend as the
        un-instrumented one. Written as a test so the coincidence cannot be
        reached for again by someone reading the two flags side by side.
        """
        assert ERROR_PASSTHROUGH_TYPES == frozenset({"anthropic"})
        assert AnthropicBackend.fills_usage_sink is True

    def test_taking_the_sink_argument_is_not_the_discriminator(self) -> None:
        """All five accept it; only three assign to it.

        This is why the fact had to become a value: the signature carries no
        information about it, so before this change the only carrier was two
        docstrings saying "accepted and never filled".
        """
        for cls in self.WRITERS + self.NON_WRITERS:
            params = inspect.signature(cls.chat_completions_stream).parameters
            assert "usage_sink" in params, cls.__name__


def _hanging_backend(started: asyncio.Event) -> MagicMock:
    """A declaring backend that emits one chunk and then never finishes.

    The hang is what makes the disconnect land *mid*-stream: the body pump is
    parked on this `await` when the cancellation arrives, so `CancelledError`
    is thrown into `stream_generator()` at its `yield` and the success mark
    is never reached. It declares and never fills, i.e. it is the defect's
    own shape apart from how the stream ends -- which is exactly what makes
    it the test of the `completed_normally` conjunct and nothing else.
    """

    def factory(_request: dict, usage_sink: Any = None) -> AsyncIterator[bytes]:
        async def gen() -> AsyncIterator[bytes]:
            yield CHUNK
            started.set()
            await asyncio.Event().wait()  # never set: park until cancelled

        return gen()

    backend = MagicMock()
    backend.chat_completions_stream = MagicMock(side_effect=factory)
    backend.completions_stream = MagicMock(side_effect=factory)
    backend.error_passthrough = False
    backend.fills_usage_sink = True
    return backend


async def _post_and_disconnect(app: FastAPI, path: str, body: dict) -> list[dict]:
    """Drive the ASGI app and hang up after the first non-empty body chunk.

    `spec_version` < 2.4 selects the task-group branch of
    `StreamingResponse`, which is the one uvicorn's HTTP protocols actually
    run. Lifted from `test_stream_disconnect_accounting.py`, which measured
    that fact about the pinned stack.
    """
    payload = json.dumps(body).encode()
    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(payload)).encode()),
        ],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }

    sent: list[dict] = []
    streaming = asyncio.Event()
    request_delivered = False

    async def receive() -> dict:
        nonlocal request_delivered
        if not request_delivered:
            request_delivered = True
            return {"type": "http.request", "body": payload, "more_body": False}
        await streaming.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        sent.append(message)
        if message["type"] == "http.response.body" and message.get("body"):
            streaming.set()

    await asyncio.wait_for(app(scope, receive, send), timeout=10)
    return sent
