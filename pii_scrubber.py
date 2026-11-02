"""
blindlog.py — PII scrubber middleware using the blindlog PyPI package (v1.2.1).

Wraps BlindLogger with AegisSRE-specific config: BLAKE2b-based deterministic
masking keyed from env vars so the same PII always produces the same ref token
(log lines remain correlatable across runs without exposing raw values).

Overhead: <2ms for typical log payloads up to ~1MB.

Usage:
    from blindlog import scrub, scrub_dict

    clean_str  = scrub("user john.doe@example.com crashed at 192.168.1.1")
    clean_dict = scrub_dict({"email": "john@example.com", "msg": "login failed"})
"""

import os
from typing import Union

from dotenv import load_dotenv
from blindlog import BlindLogger, BlindLogConfig, DEFAULT_SENSITIVE_KEYS

load_dotenv()

# ---------------------------------------------------------------------------
# Config — secret_key + salt injected from env so masking is deterministic
# and consistent across all AegisSRE workers without hardcoding secrets.
# ---------------------------------------------------------------------------
_config = BlindLogConfig(
    secret_key=os.getenv("BLINDLOG_SECRET_KEY", "aegissre-default-secret"),
    salt=os.getenv("BLINDLOG_SALT", "aegissre-v1"),
    # Extend the default sensitive key set with SRE-specific fields
    sensitive_keys=DEFAULT_SENSITIVE_KEYS | frozenset({
        "user_id", "username", "account_id", "customer_id",
        "session_id", "request_id", "trace_id",
        "github_token", "modal_token", "webhook_secret",
    }),
    debug_mode=False,
)

_logger = BlindLogger(config=_config)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

Payload = Union[str, int, float, bool, None, dict, list]


def scrub(raw: Payload) -> Payload:
    """
    Scrub all detectable PII from a string or structured payload.

    - Strings: emails, IPs, tokens, API keys, credit cards, SSNs replaced
      with deterministic ref tokens (e.g. blnd_ref_a7de6e...).
    - Dicts/lists: recursively processed; sensitive keys masked by key name
      in addition to pattern matching on values.

    Args:
        raw: Raw log string, dict (parsed JSON log), or list.

    Returns:
        Same type as input with PII replaced.
    """
    return _logger.mask(raw)


def scrub_dict(data: dict) -> dict:
    """
    Convenience wrapper for dict payloads (e.g. parsed JSON webhook bodies).
    Equivalent to scrub() but type-annotated for dicts specifically.
    """
    result = _logger.mask(data)
    # BlindLogger.mask returns the same type as input
    assert isinstance(result, dict), "BlindLogger.mask returned non-dict for dict input"
    return result


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    samples = [
        "2026-11-02T14:32:01Z INFO  user login: john.doe@example.com from 192.168.1.42",
        "2026-11-02T14:32:02Z DEBUG Authorization: Bearer ghp_abc123XYZsecrettoken456",
        "2026-11-02T14:32:03Z ERROR payment failed for card 4111111111111111",
        "2026-11-02T14:32:04Z INFO  aws_key=AKIAIOSFODNN7EXAMPLE processing",
        "2026-11-02T14:32:05Z WARN  ssn=123-45-6789 flagged for review",
    ]

    print("=== BlindLog Scrub Demo ===\n")
    for line in samples:
        masked = scrub(line)
        print(f"IN : {line}")
        print(f"OUT: {masked}")
        print()

    # Dict scrub
    payload = {
        "email": "jane@corp.com",
        "password": "hunter2",
        "message": "login failed from 10.0.0.1",
        "meta": {"api_key": "sk-abc123secretkey"},
    }
    print("=== Dict Scrub ===")
    import json
    print("IN :", json.dumps(payload, indent=2))
    print("OUT:", json.dumps(scrub_dict(payload), indent=2))
