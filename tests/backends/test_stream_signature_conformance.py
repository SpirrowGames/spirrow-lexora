"""T-streaming-ledger-row PR-A: every backend's `*_stream` accepts `usage_sink`.

`routes.py` passes `usage_sink=` **unconditionally** to whichever backend the
router returned -- five call sites, `:416` `:497` `:863` `:944` `:2025` -- and
all five backends *override* both stream methods with their own `def`. Python
does not enforce an ABC's signature on an override, so the base class's
default reaches nobody. A backend whose signature lost the parameter would
`TypeError` on **every streamed request through it**, in production, at the
call site.

Nothing else in this suite saw that. Reverting `openai_compatible.py` alone to
`c8a97dc`, every test file left in place, leaves the whole suite at **640
passed**; reverting `vllm.py` alone leaves **639 passed** and the one failure
is the pre-existing `test_ledger_coverage.py` duration bracket, which that
file's own comment concedes and which cannot reach either reverted file (zero
references to it). Occurrences of `unexpected keyword argument 'usage_sink'`
across both runs: **0**. So the four accept-and-ignore signatures -- the part
of PR-A that exists *because it is required at runtime* -- were the one part
of it with no detector. This file is that detector.

`bind` is the exact predicate the call site evaluates: it needs no I/O, no
doubles and no network, and it asks the signature the question `routes.py`
asks it rather than a proxy for that question.

**One case per (backend, method), never one case over all ten**, so a
single-site revert reds exactly the site it reverted. Measured on the finished
tree, win32 / CPython 3.12, one single-site signature revert at a time and
restored before the next: each of the ten reverts reds **1 of 10, and it is
always the case named for that exact (backend, method) pair** -- the other
nine stay green, including the sibling method on the same class. No mutation
reds a pair it does not name.
"""

import inspect
from typing import Any

import pytest

from lexora.backends.anthropic import AnthropicBackend
from lexora.backends.base import UsageSink
from lexora.backends.claude_code import ClaudeCodeBackend
from lexora.backends.gemini import GeminiBackend
from lexora.backends.openai_compatible import OpenAICompatibleBackend
from lexora.backends.vllm import VLLMBackend

# The five the factory can build (`backends/factory.py`), each paired with both
# streaming methods `routes.py` hands a sink to.
BACKENDS = [
    ("anthropic", AnthropicBackend),
    ("claude_code", ClaudeCodeBackend),
    ("gemini", GeminiBackend),
    ("openai_compatible", OpenAICompatibleBackend),
    ("vllm", VLLMBackend),
]
STREAM_METHODS = ("chat_completions_stream", "completions_stream")

CASES = [
    pytest.param(cls, method, id=f"{name}-{method}")
    for name, cls in BACKENDS
    for method in STREAM_METHODS
]


@pytest.mark.parametrize(("backend_cls", "method_name"), CASES)
def test_stream_method_accepts_usage_sink(
    backend_cls: type[Any], method_name: str
) -> None:
    """The signature binds the call `routes.py` makes, or this reds."""
    method = getattr(backend_cls, method_name)
    # `None` stands in for `self`: this asks the signature, not an instance,
    # so no backend is constructed and no upstream is contacted.
    inspect.signature(method).bind(None, {"model": "x"}, usage_sink=UsageSink())
