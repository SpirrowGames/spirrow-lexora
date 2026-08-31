"""O-3: a streaming pre-flight rejection is still a request.

すべての streaming エンドポイントは、backend を呼ぶ *前* に stats /
metrics へ登録する。PR-B が新設した passthrough の pre-flight は
`stats_collector.start_request()` より **上** で早期 return していたため、
拒否された streaming リクエストが台帳から完全に消えていた — 同じ
エンドポイントの非 streaming 分岐は常に計上していたので、これは方針の
空白ではなく PR-B が持ち込んだ不整合である。要件 6 (frontier の呼び出し
回数を単独で集計する = 後で請求額と突合する) にも効く: **拒否された
呼び出しも呼び出しである**。

対象は 3 箇所。`/v1/chat/completions` と `/v1/completions` は PR-B が
新設した pre-flight、`/v1/messages` は develop に既存の pre-flight で、
そこでは passthrough かどうかに関わらず **すべての** 早期 return
(governance 拒否・総称 BackendError を含む) が不可視だった。

★ ACTIVE_REQUESTS の assert が本ファイルの要である。素朴な修正
(`start_request` を上へ動かすだけで早期 return を閉じない) は
`total_requests` の assert を緑にしたまま **Gauge を永久にリーク**させる
— 数えないより悪い。in-flight が呼び出し前の値に戻ることを見張る検出器は
ここにしか無い。
"""

import asyncio
import concurrent.futures
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

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
from lexora.backends.base import BackendError, BackendUpstreamError
from lexora.backends.gemini import GeminiGovernanceError
from lexora.services.metrics import MetricsCollector
from lexora.services.rate_limiter import RateLimiter
from lexora.services.retry_handler import RetryHandler
from lexora.services.stats import StatsCollector


def _active_requests(endpoint: str) -> float:
    """Current value of the ACTIVE_REQUESTS gauge for one endpoint.

    Read through the public registry rather than the Gauge's private
    `_value`, and always as a *delta* against the value observed before
    the request: the collector is a process-global, so another test in
    the same session may have touched it.
    """
    value = REGISTRY.get_sample_value(
        "lexora_active_requests", {"endpoint": endpoint}
    )
    return 0.0 if value is None else value


def _raising_stream(exc: BaseException):
    """A backend stream whose first `__anext__` raises.

    `BaseException`, not `Exception`: `asyncio.CancelledError` is the
    class the pre-flight must survive and it is deliberately outside the
    `Exception` hierarchy.
    """

    def factory(_request: dict) -> AsyncIterator[bytes]:
        async def gen() -> AsyncIterator[bytes]:
            raise exc
            yield b""  # pragma: no cover - unreachable, makes this a generator

        return gen()

    return MagicMock(side_effect=factory)


@pytest.fixture
def stats_collector() -> StatsCollector:
    return StatsCollector()


@pytest.fixture
def metrics_collector() -> MetricsCollector:
    return MetricsCollector()


@pytest.fixture
def mock_backend() -> MagicMock:
    backend = MagicMock()
    backend.chat_completions = AsyncMock()
    backend.completions = AsyncMock()
    backend.error_passthrough = True
    return backend


@pytest.fixture
def client(
    mock_backend: MagicMock,
    stats_collector: StatsCollector,
    metrics_collector: MetricsCollector,
) -> TestClient:
    backend_router = MagicMock()
    backend_router.get_backend_for_model = MagicMock(return_value=mock_backend)
    backend_router.resolve_model = MagicMock(return_value="claude-fable-5")
    backend_router.get_backend_name_for_model = MagicMock(return_value="frontier")
    backend_router.is_tier = MagicMock(return_value=True)

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_backend] = lambda: mock_backend
    app.dependency_overrides[get_backend_router] = lambda: backend_router
    app.dependency_overrides[get_stats_collector] = lambda: stats_collector
    app.dependency_overrides[get_retry_handler] = lambda: RetryHandler(
        max_retries=1, base_delay=0.01, max_delay=0.1, jitter=False
    )
    app.dependency_overrides[get_rate_limiter] = lambda: RateLimiter(
        default_rate=1000.0, default_burst=1000
    )
    app.dependency_overrides[is_rate_limit_enabled] = lambda: True
    app.dependency_overrides[get_metrics_collector] = lambda: metrics_collector
    app.dependency_overrides[get_cost_tracker] = lambda: None
    return TestClient(app)


UPSTREAM_REFUSAL = BackendUpstreamError(
    "API error (400): declined",
    status_code=400,
    body={"type": "error", "error": {"type": "refusal", "message": "declined"}},
    backend_name="frontier",
)

CHAT_BODY = {
    "model": "frontier",
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": True,
}
COMPLETIONS_BODY = {"model": "frontier", "prompt": "Hi", "stream": True}
MESSAGES_BODY = {
    "model": "frontier",
    "max_tokens": 16,
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": True,
}


class TestStreamingPreflightIsCounted:
    """拒否された streaming pre-flight が stats と metrics に載ること。"""

    @pytest.mark.parametrize(
        ("endpoint", "stream_attr", "body", "expected_status"),
        [
            ("/v1/chat/completions", "chat_completions_stream", CHAT_BODY, 400),
            ("/v1/completions", "completions_stream", COMPLETIONS_BODY, 400),
            ("/v1/messages", "chat_completions_stream", MESSAGES_BODY, 400),
        ],
    )
    def test_upstream_refusal_is_recorded(
        self,
        client: TestClient,
        mock_backend: MagicMock,
        stats_collector: StatsCollector,
        endpoint: str,
        stream_attr: str,
        body: dict,
        expected_status: int,
    ) -> None:
        setattr(mock_backend, stream_attr, _raising_stream(UPSTREAM_REFUSAL))
        before = _active_requests(endpoint)

        response = client.post(endpoint, json=body)

        assert response.status_code == expected_status
        stats = stats_collector.get_stats()
        assert stats["total_requests"] == 1
        assert stats["failed_requests"] == 1
        # ★ the leak detector: in-flight must return to where it started.
        assert _active_requests(endpoint) == before

    @pytest.mark.parametrize(
        ("endpoint", "stream_attr", "body"),
        [
            ("/v1/chat/completions", "chat_completions_stream", CHAT_BODY),
            ("/v1/completions", "completions_stream", COMPLETIONS_BODY),
            ("/v1/messages", "chat_completions_stream", MESSAGES_BODY),
        ],
    )
    def test_generic_backend_error_is_recorded(
        self,
        client: TestClient,
        mock_backend: MagicMock,
        stats_collector: StatsCollector,
        endpoint: str,
        stream_attr: str,
        body: dict,
    ) -> None:
        """The other early-return class out of the same pre-flight."""
        setattr(mock_backend, stream_attr, _raising_stream(BackendError("boom")))
        before = _active_requests(endpoint)

        response = client.post(endpoint, json=body)

        assert response.status_code == 502
        stats = stats_collector.get_stats()
        assert stats["total_requests"] == 1
        assert stats["failed_requests"] == 1
        assert _active_requests(endpoint) == before

    def test_messages_governance_refusal_is_recorded(
        self,
        client: TestClient,
        mock_backend: MagicMock,
        stats_collector: StatsCollector,
    ) -> None:
        """/v1/messages pre-flights every backend, not only passthrough ones.

        The gate named one call site; this is the return the gate did not
        look at — a naysayer governance refusal, which is the loop's own
        traffic.
        """
        mock_backend.chat_completions_stream = _raising_stream(
            GeminiGovernanceError("tools are not allowed")
        )
        before = _active_requests("/v1/messages")

        response = client.post("/v1/messages", json=MESSAGES_BODY)

        assert response.status_code == 400
        stats = stats_collector.get_stats()
        assert stats["total_requests"] == 1
        assert stats["failed_requests"] == 1
        assert _active_requests("/v1/messages") == before


PREFLIGHT_ENDPOINTS = [
    ("/v1/chat/completions", "chat_completions_stream", CHAT_BODY),
    ("/v1/completions", "completions_stream", COMPLETIONS_BODY),
    ("/v1/messages", "chat_completions_stream", MESSAGES_BODY),
]


class TestPreflightUnlistedExceptionDoesNotLeak:
    """O-6: 列挙されていない例外で抜けても台帳は閉じること。

    上の TestStreamingPreflightIsCounted は pre-flight が **列挙した**
    例外クラス (`BackendUpstreamError` / `BackendError` /
    `GeminiGovernanceError`) について早期 return が `_fail_preflight` を
    通ることを見張る。それは実装の変異は捕まえるが、**列挙の漏れ**は
    捕まえない — `except` 節の列挙は、そこに書かれていない例外について
    何も言わないからである。

    リクエストは pre-flight より **前** に登録されるので、列挙外の例外が
    route handler ごと抜けると `_fail_preflight` にも下流の
    `stream_generator` にも入らず、ACTIVE_REQUESTS が恒久的にリークする。
    しかも `total_requests` は 0 のまま — 拒否された呼び出しが台帳から
    消えるという、このファイルが閉じたはずの穴が別の入口で開く。

    列挙外の例外は仮定ではなく実在する: `_map_model` /
    `_to_anthropic_request` / `_to_gemini_request` /
    `_enforce_governance_gate` はどれも backend 側の `try` の外にあり、
    async generator なので最初の `__anext__()` で走る。httpx も
    `ConnectError` / `TimeoutException` / `HTTPError` しか包まないので、
    閉じた client の再利用が投げる `RuntimeError` は素通りする。

    `asyncio.CancelledError` を並べて測るのは、それが `Exception` ですら
    ない = `except Exception` を足す修正では塞がらないことを固定するため。
    ただし本テストが実証するのは「列挙外の例外一般」であって、
    「クライアント切断が CancelledError を送出する」ことではない
    (その因果はこのリポジトリでは未検証)。
    """

    @pytest.mark.parametrize(("endpoint", "stream_attr", "body"), PREFLIGHT_ENDPOINTS)
    @pytest.mark.parametrize(
        ("exc_factory", "escapes_as"),
        [
            (lambda: RuntimeError("boom"), RuntimeError),
            # `asyncio.CancelledError` reaches the caller as
            # `concurrent.futures.CancelledError`: TestClient drives the app
            # through a blocking portal, and a cancelled portal future
            # re-raises the *futures* class (measured: the two are distinct
            # classes on this interpreter, and the futures one is even an
            # `Exception` while the asyncio one is not). That translation is
            # the harness's, not lexora's — what this test pins is that the
            # exception is re-raised at all rather than swallowed into an
            # HTTP answer, so either class satisfies it.
            (
                lambda: asyncio.CancelledError(),
                (asyncio.CancelledError, concurrent.futures.CancelledError),
            ),
        ],
        ids=["runtime-error", "cancelled-error"],
    )
    def test_unlisted_exception_closes_the_ledger(
        self,
        client: TestClient,
        mock_backend: MagicMock,
        stats_collector: StatsCollector,
        endpoint: str,
        stream_attr: str,
        body: dict,
        exc_factory,
        escapes_as: type[BaseException] | tuple[type[BaseException], ...],
    ) -> None:
        setattr(mock_backend, stream_attr, _raising_stream(exc_factory()))
        before = _active_requests(endpoint)

        # (c) the exception is re-raised, not swallowed into an HTTP answer.
        with pytest.raises(escapes_as):
            client.post(endpoint, json=body)

        # (b) a rejected call is still a call.
        stats = stats_collector.get_stats()
        assert stats["total_requests"] == 1
        assert stats["failed_requests"] == 1
        # (a) ★ the leak detector.
        assert _active_requests(endpoint) == before


def test_preflight_unhandled_log_names_the_exception_type(
    client: TestClient, mock_backend: MagicMock
) -> None:
    """B-14: the log line for a `BaseException` exit must say what left.

    `str(e)` is the empty string for every `BaseException` that is not an
    `Exception` — `CancelledError`, `KeyboardInterrupt`, `SystemExit` all
    measure as `''`. Those are precisely the class of exception the
    `except BaseException` clause was added for, so `error=str(e)` alone
    produced a log record that named the endpoint and nothing else.

    This asserts a property of the shipped code (an empty `str()`), not the
    gate's unverified claim that a client disconnect is what raises it. The
    level is deliberately unchanged: demoting `CancelledError` would encode
    "a cancel here is normal traffic, not a fault", which nothing in this
    repository has measured. That question belongs to F-3.
    """
    from lexora.api import routes as routes_module

    mock_backend.chat_completions_stream = _raising_stream(asyncio.CancelledError())

    with patch.object(routes_module.logger, "error") as logged:
        with pytest.raises((asyncio.CancelledError, concurrent.futures.CancelledError)):
            client.post("/v1/chat/completions", json=CHAT_BODY)

    call = next(
        c
        for c in logged.call_args_list
        if c.args and c.args[0] == "chat_completion_stream_preflight_unhandled"
    )
    assert (call.kwargs["error"], call.kwargs["error_type"]) == ("", "CancelledError")
