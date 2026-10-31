"""Identifier- and path-shaped exclusions from D-1 and D-2 (ADR-2026-05-31-14
§7.3, T-redactor-over-redacts-identifiers-and-paths msg-921, plus the four
points approved on 2026-10-31).

Three groups:

* must stay: the output equals the input (false BLOCKING is what this fixes);
* must redact: the leak count is zero;
* leak record: what the exclusion gives up, measured with a fixed seed. These
  print and do not judge a threshold (msg-917, msg-919). The ADR points here
  as the record: ``pytest tests/test_ingress_redact_exclusions.py -s -k record``.

Measured at the commit that added this file (500 tokens each, entropy >= 4.0):

* base64url, all: 41/500 = 8.2%; with ``_`` and no ``-``: 152/500 = 30.4%
* standard base64, all: 52/500 = 10.4%; with ``/`` and no ``+``: 244/500 = 48.8%
* base64url leaking only through digit-only segments: 3/500 = 0.6%
* a word plus short digit groups: 40 of 465 reach entropy 4.0, and all 40 leak
"""

from __future__ import annotations

import base64
import random
import re
import string
from collections.abc import Callable

import pytest

from lexora.ingress.redact import (
    ENTROPY_THRESHOLD,
    KIND_ASSIGNMENT,
    _is_word_or_path_shaped,
    redact,
    shannon_entropy,
)

N = 500
ALNUM = string.ascii_letters + string.digits

_EXAMPLE_1 = "test_verify_refuses_a_declaration_without_matching_scope_hashes"
_EXAMPLE_2 = "verdict_task_prompt=_PR_REVIEW_SYSTEM_PROMPT"
_EXAMPLE_3 = "C:/workspace/unreal/VoxelWorldPlayProofHost-v3"

_D2_MUST_STAY = [
    # The three incidents of msg-897.
    _EXAMPLE_1,
    _EXAMPLE_2,
    _EXAMPLE_3,
    # Paths.
    "docs/adr/ADR-2026-05-31-14-gawa-method-withdrawal.md",
    "C:\\workspace\\unreal\\VoxelWorldPlayProofHost-v3\\Source\\Subsystem.cpp",
    # SCREAMING_SNAKE.
    "PR_REVIEW_SYSTEM_PROMPT_WITH_MAXIMUM_CONTEXT_BUDGET_AND_LIMITS",
    # Einstein's seven (msg-912, msg-914, msg-916).
    "OAuth2Client_connection_settings_dict",
    "Base64UrlEncode_payload_bytes_for_jwt",
    "calculateXAndY_coordinate_list_for_view",
    "convertIPv4ToIPv6_address_mapping",
    "convertBase64ToBase32AndBase16_dict",
    "src/ui/js/lib/db/id_to_ip_map_for_eu",
    "i18n_l10n_a11y_utf8_to_utf16_converter",
    # The incidents behind the 2026-10-31 digit-only segments (msg-7702).
    "test_verify_sha_256_digest_of_uploaded_bundle_matches_manifest",
    "test_verify_refuses_a_declaration_with_2_matching_scope_hashes",
    # Underscores at both ends are ignored.
    "__test_verify_refuses_a_declaration_without_matching_scope__",
]


class TestD2MustStay:
    @pytest.mark.parametrize("text", _D2_MUST_STAY)
    def test_is_unchanged(self, text: str) -> None:
        assert redact(f"def {text}():") == (f"def {text}():", {})

    @pytest.mark.parametrize(
        "token",
        [t for t in _D2_MUST_STAY if ":" not in t and "\\" not in t and "." not in t],
    )
    def test_reaches_the_exclusion_rather_than_missing_the_entropy_bar(self, token: str) -> None:
        """The exclusion is what keeps these, whatever their entropy is."""
        assert _is_word_or_path_shaped(token)

    def test_the_incidents_are_above_the_entropy_bar(self) -> None:
        """Without the exclusion, these would be redacted: the cases are real."""
        for token in (
            _EXAMPLE_1,
            _EXAMPLE_2,
            "/workspace/unreal/VoxelWorldPlayProofHost-v3",
            "test_verify_sha_256_digest_of_uploaded_bundle_matches_manifest",
            "test_verify_refuses_a_declaration_with_2_matching_scope_hashes",
        ):
            assert shannon_entropy(token) >= ENTROPY_THRESHOLD, token


_HIGH_RUN = "aB3dE5gH7jK9mN1pQ3sT5vW7yZ9bC1dF3hJ5"  # 36 characters, no separator


class TestD2MustRedact:
    @pytest.mark.parametrize(
        "token",
        [
            "abcdefghijklmnopqrstuv/wxyz+123456==",
            # C wins over A and B.
            f"prefix_{_HIGH_RUN}",
            f"path/to/{_HIGH_RUN}",
            f"name={_HIGH_RUN}",
            # Long camelCase is C, and stays redacted (2026-10-31, point 3).
            "FVoxelWorldPlayProofHostSubsystemInitializer",
            # A segment that starts with a digit and holds letters.
            "render_3d_volumetric_lighting_pipeline_config_table",
            "enable_2fa_verify_block_mixup_for_every_job_queue_worker",
            # Not the shape of A: an empty side, an empty segment, a third side.
            "abcdefghijklm_nopqrstuvwxyz_ABCDEFGH==",
            "abcdefghijklm__nopqrstuvwxyz_ABCDEFGH",
            "abcdefghijklm=nopqrstuvwxyz=ABCDEFGH",
            # Hyphen-only tokens: the known leftover, pinned (msg-911 (b)).
            "T-redactor-over-redacts-identifiers-and-paths-v2-xjqz",
            # Hyphen and underscore together are neither A nor B.
            "aBcDeFgHiJkLmNoP_qRsTuVwXyZ-0123456789",
        ],
    )
    def test_is_redacted(self, token: str) -> None:
        assert shannon_entropy(token) >= ENTROPY_THRESHOLD, token
        assert redact(f"v {token} e")[0] == "v [REDACTED:entropy#1] e"

    @pytest.mark.parametrize(
        "secret",
        [
            "ghp_" + _HIGH_RUN,
            "github_pat_" + "abcdefghij_klmnopqrst_uvwxyz012345",
            "sk-proj-abcdefghij_klmnopqrstuvwx",
        ],
    )
    def test_known_prefixes_are_not_touched_by_the_exclusion(self, secret: str) -> None:
        assert redact(f"v {secret} e")[0] == "v [REDACTED:key#1] e"

    def test_bearer_is_not_touched_by_the_exclusion(self) -> None:
        out = redact("Authorization: Bearer abcdefghij_klmnopqrst_uvwxyz_012345")[0]
        assert "abcdefghij" not in out

    def test_a_multiline_pem_body_is_not_touched_by_the_exclusion(self) -> None:
        rng = random.Random(20261101)
        alphabet = string.ascii_letters + string.digits + "/"
        rows = ["".join(rng.choice(alphabet) for _ in range(64)) for _ in range(4)]
        out = redact("\n".join(rows))[0]
        assert all(row not in out for row in rows)


def _random_tokens(alphabet: str, keep: Callable[[str], bool], seed: int, n: int = N) -> list[str]:
    """``n`` tokens of 32-64 characters from ``alphabet`` passing ``keep``."""
    rng = random.Random(seed)
    out: list[str] = []
    while len(out) < n:
        token = "".join(rng.choice(alphabet) for _ in range(rng.randint(32, 64)))
        if keep(token) and shannon_entropy(token) >= ENTROPY_THRESHOLD:
            out.append(token)
    return out


def _leaks(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t in redact(f"v {t} e")[0]]


def _true(_: str) -> bool:
    return True


class TestLeakZero:
    """Property tests: fixed seed, 500 tokens each, 32-64 characters."""

    @pytest.mark.parametrize(
        ("alphabet", "keep"),
        [
            (ALNUM, _true),
            (ALNUM + "+", _true),
            (ALNUM + "+/", lambda t: "+" in t and "/" in t),
        ],
        ids=["alnum", "base64-no-slash", "base64-slash-and-plus"],
    )
    def test_nothing_leaks(self, alphabet: str, keep: Callable[[str], bool]) -> None:
        assert _leaks(_random_tokens(alphabet, keep, seed=20261101)) == []

    def test_hex_is_outside_the_new_exclusion(self) -> None:
        """Hex stays out through the older hash-shape rule (commit SHAs, §7.3,
        ``TestD2Entropy.test_hash_shapes_are_excluded``), never through this
        one: 32+ hex characters are a separator-free run (C)."""
        rng = random.Random(20261101)
        for _ in range(N):
            assert not _is_word_or_path_shaped("".join(rng.choice("0123456789abcdef") for _ in range(rng.randint(32, 64))))


#: The first version of A (msg-921), to see what digit-only segments add.
_A_WITHOUT_DIGIT_SEGMENTS = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z][A-Za-z0-9]*)*")


def _encoded(rng: random.Random, encode: Callable[[bytes], bytes]) -> str:
    """Real encoder output: 24-48 random bytes, so 32-64 characters."""
    return encode(rng.randbytes(rng.randint(24, 48))).decode()


def _encoded_tokens(encode: Callable[[bytes], bytes], keep: Callable[[str], bool], seed: int) -> list[str]:
    rng = random.Random(seed)
    out: list[str] = []
    while len(out) < N:
        token = _encoded(rng, encode)
        if keep(token) and shannon_entropy(token) >= ENTROPY_THRESHOLD:
            out.append(token)
    return out


def _b64url(raw: bytes) -> bytes:
    return base64.urlsafe_b64encode(raw).rstrip(b"=")


def _rate(tokens: list[str]) -> str:
    return f"{len(_leaks(tokens))}/{len(tokens)} = {len(_leaks(tokens)) / len(tokens):.1%}"


class TestLeakRecord:
    """Records, not gates. Each prints one line; see the module docstring."""

    def test_record_base64url(self) -> None:
        every = _encoded_tokens(_b64url, _true, seed=1)
        in_class = _encoded_tokens(_b64url, lambda t: "_" in t and "-" not in t, seed=2)
        print(f"\nrecord base64url, all: {_rate(every)}")
        print(f"record base64url, has _ and no -: {_rate(in_class)}")

    def test_record_standard_base64(self) -> None:
        every = _encoded_tokens(base64.b64encode, _true, seed=3)
        in_class = _encoded_tokens(base64.b64encode, lambda t: "/" in t and "+" not in t, seed=4)
        print(f"\nrecord base64, all: {_rate(every)}")
        print(f"record base64, has / and no +: {_rate(in_class)}")

    def test_record_digit_only_segments(self) -> None:
        """What ``[0-9]+`` segments (2026-10-31, point 2) add on their own."""
        in_class = _encoded_tokens(_b64url, lambda t: "_" in t and "-" not in t, seed=2)
        leaked = _leaks(in_class)
        only_through_digits = [t for t in leaked if not _A_WITHOUT_DIGIT_SEGMENTS.fullmatch(t.strip("_"))]
        print(
            f"\nrecord base64url, leaks only through digit-only segments: "
            f"{len(only_through_digits)}/{len(in_class)} = {len(only_through_digits) / len(in_class):.1%}"
        )

    def test_record_short_digit_segments_after_a_word(self) -> None:
        """The case digit-only segments open: a credential in short groups
        behind one word. Digits alone rarely reach entropy 4.0, so a mixed word
        and digit tail is what could have been caught before."""
        rng = random.Random(5)
        tokens: list[str] = []
        tried = 0
        while tried < N:
            tried += 1
            word = "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(4, 12)))
            groups = "_".join(
                "".join(rng.choice(string.digits) for _ in range(rng.randint(2, 6))) for _ in range(rng.randint(5, 10))
            )
            token = f"{word}_{groups}"
            if len(token) >= 32:
                tokens.append(token)
        above = [t for t in tokens if shannon_entropy(t) >= ENTROPY_THRESHOLD]
        print(
            f"\nrecord word + digit groups: {len(tokens)} tokens of 32+, "
            f"{len(above)} at entropy >= 4.0 (redacted before this change), "
            f"{len(_leaks(above))} of those leak now"
        )


class TestD1MustStay:
    """msg-904's five, and the 2026-10-31 camelCase names."""

    @pytest.mark.parametrize(
        "text",
        [
            'TOKEN_RE = re.compile(r"[a-z]+")',
            "max_tokens = 4096\na = 1\nb = 2\nc = 3\nd = 4\ne = 5\nf = 6\ng = 7",
            "token_count = 12",
            'TOKEN_ENV = "GITHUB_TOKEN"',
            "tokenizer = load()",
            "secretName: my-kubernetes-secret-object\nnext_line_here",
            'passwordRegex = "^.{8,}$"\nnext_line_here',
            "apiKeyFile: /etc/service/credentials\nnext_line_here",
            "tokens: abcdefghijkl\nnext_line_here",
            "client_secret_id: abcdefghijkl\nnext_line_here",
            "api-key.header: abcdefghijkl",
        ],
    )
    def test_is_unchanged(self, text: str) -> None:
        assert redact(text) == (text, {})


class TestD1MustRedact:
    @pytest.mark.parametrize(
        "text",
        [
            # msg-904.
            "secrets:\n  abcdefghijkl",
            'token_value_re = "abcdefghijkl"',
            # Plural secrets and passwords stay key names.
            "passwords: abcdefghijkl",
            # The word must end there (the known leftover, msg-911).
            "token_count_max = abcdefghijkl",
            "secretNameValue: abcdefghijkl",
            # The camelCase list is matched with only the first letter lowered.
            "SECRETNAME = abcdefghijkl",
            # One candidate left over makes a key line.
            "max_tokens_secret = abcdefghijkl",
            "max_tokens = 1; api_key = abcdefghijkl",
            "token_count = 1, password = abcdefghijkl",
        ],
    )
    def test_is_redacted(self, text: str) -> None:
        out, counts = redact(text)
        assert "abcdefghijkl" not in out
        assert counts[KIND_ASSIGNMENT] == 1

    def test_an_excluded_name_inside_a_window_does_not_restart_it(self) -> None:
        lines = ["token = None", "a", "b", "max_tokens = 1", "c", "d", "e", "f", "g"]
        out = redact("\n".join(lines))[0].split("\n")
        assert out[-2:] == ["f", "g"] and out[-3] != "e"
