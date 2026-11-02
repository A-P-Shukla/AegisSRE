"""
modal_reasoning.py — Multimodal root cause analysis via Muse-Spark-816B on Modal.

Receives: sanitized logs (str) + segmented image crop (base64 PNG) + git diff (str)
Returns:  strict JSON RootCauseSpec or a fallback signal for the circuit breaker.

The model is called via its OpenAI-compatible Modal inference endpoint.
No model is self-hosted here — we use the existing Modal-hosted vLLM endpoint.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

import modal

# ---------------------------------------------------------------------------
# Container image — lightweight, just needs the OpenAI client
# ---------------------------------------------------------------------------
reasoning_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "openai==1.35.7",
        "python-dotenv==1.0.1",
    )
)

app = modal.App("aegissre-reasoning")

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------
FAULT_TYPES = frozenset({
    "DOM_ELEMENT_COLLISION",
    "API_TIMEOUT",
    "DB_DEADLOCK",
    "CSS_REGRESSION",
    "NULL_POINTER",
    "AUTH_FAILURE",
    "NETWORK_ERROR",
    "UNKNOWN",
})

CONFIDENCE_THRESHOLD = 0.80

SYSTEM_PROMPT = """You are AegisSRE, an expert Site Reliability Engineer specializing in
root cause analysis. You will receive:
1. Sanitized backend server logs (PII has been replaced with ref tokens)
2. A base64-encoded PNG screenshot crop showing the visual anomaly from a failed UI test
3. A Git commit diff showing recent code changes

Your task is to correlate the visual anomaly with the backend trace and produce a
precise root cause analysis. You MUST respond with ONLY a valid JSON object — no
markdown, no explanation, no preamble.

Required JSON schema:
{
  "fault_type": "<one of: DOM_ELEMENT_COLLISION | API_TIMEOUT | DB_DEADLOCK | CSS_REGRESSION | NULL_POINTER | AUTH_FAILURE | NETWORK_ERROR | UNKNOWN>",
  "root_cause_explanation": "<clear, actionable explanation in 2-3 sentences>",
  "target_file": "<relative path to the file that needs patching, e.g. src/components/CheckoutButton.tsx>",
  "remediation_strategy": "<concrete description of the fix — specific enough for a code model to implement>",
  "confidence": <float between 0.0 and 1.0>
}

Rules:
- If you cannot determine target_file with high confidence, use the most likely candidate.
- confidence must reflect your actual certainty. Do not inflate it.
- remediation_strategy must be specific: name the function, CSS property, or API call to change.
- Do NOT include any text outside the JSON object.
"""


@dataclass
class RootCauseSpec:
    fault_type: str
    root_cause_explanation: str
    target_file: str
    remediation_strategy: str
    confidence: float

    def is_actionable(self) -> bool:
        return (
            self.confidence >= CONFIDENCE_THRESHOLD
            and self.fault_type in FAULT_TYPES
            and bool(self.target_file)
            and self.fault_type != "UNKNOWN"
        )

    def to_dict(self) -> dict:
        return {
            "fault_type": self.fault_type,
            "root_cause_explanation": self.root_cause_explanation,
            "target_file": self.target_file,
            "remediation_strategy": self.remediation_strategy,
            "confidence": self.confidence,
        }


@dataclass
class ReasoningResult:
    success: bool
    spec: Optional[RootCauseSpec]
    raw_response: str
    error: Optional[str]
    fallback: bool
    fallback_reason: Optional[str]


# ---------------------------------------------------------------------------
# Modal function
# ---------------------------------------------------------------------------
@app.function(
    image=reasoning_image,
    secrets=[modal.Secret.from_name("aegis-modal-secrets")],
    timeout=300,
    scaledown_window=60,
)
def analyze_failure(
    sanitized_logs: str,
    cropped_image_b64: str,
    git_diff: str,
    is_visual_fallback: bool = False,
) -> dict:
    """
    Run multimodal root cause analysis.

    Args:
        sanitized_logs:     BlindLog-scrubbed server logs.
        cropped_image_b64:  Base64 PNG — either SAM crop or full-page fallback.
        git_diff:           Output of `git diff HEAD~1 HEAD`.
        is_visual_fallback: True if SAM fell back to full-page screenshot.

    Returns dict representation of ReasoningResult.
    """
    from openai import OpenAI

    client = OpenAI(
        base_url="https://a-p-shukla--ep-muse-spark-open-816b-a42b-server.us-west.modal.direct/v1",
        api_key=f"{os.environ['MODAL_PROXY_TOKEN_ID']}.{os.environ['MODAL_PROXY_TOKEN_SECRET']}",
    )

    # Build user message — multimodal: text + image
    visual_note = (
        "[Note: SAM 2.1 segmentation fell back to full-page screenshot — "
        "no high-confidence anomaly region was isolated. Analyze the full page.]"
        if is_visual_fallback
        else "[SAM 2.1 successfully isolated the anomaly region shown in the image.]"
    )

    user_content = [
        {
            "type": "text",
            "text": (
                f"{visual_note}\n\n"
                f"=== SANITIZED BACKEND LOGS ===\n{sanitized_logs}\n\n"
                f"=== GIT DIFF (recent changes) ===\n{git_diff}"
            ),
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{cropped_image_b64}",
                "detail": "high",
            },
        },
    ]

    try:
        response = client.chat.completions.create(
            model="meta-models/Muse-Spark-Open-816B-A42B",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,   # Low temp for deterministic structured output
            max_tokens=1024,
            top_p=0.95,
            stream=False,
            extra_body={"reasoning_effort": "high"},
        )

        raw = response.choices[0].message.content.strip()

    except Exception as exc:
        return ReasoningResult(
            success=False,
            spec=None,
            raw_response="",
            error=str(exc),
            fallback=True,
            fallback_reason=f"Model API error: {exc}",
        ).__dict__

    # --- Parse and validate JSON output ---
    spec, parse_error = _parse_spec(raw)

    if parse_error or spec is None:
        return ReasoningResult(
            success=False,
            spec=None,
            raw_response=raw,
            error=parse_error,
            fallback=True,
            fallback_reason=f"JSON parse failure: {parse_error}",
        ).__dict__

    if not spec.is_actionable():
        reason = (
            f"confidence {spec.confidence:.2f} < threshold {CONFIDENCE_THRESHOLD}"
            if spec.confidence < CONFIDENCE_THRESHOLD
            else f"fault_type is UNKNOWN or target_file is empty"
        )
        return ReasoningResult(
            success=False,
            spec=spec,
            raw_response=raw,
            error=None,
            fallback=True,
            fallback_reason=f"Not actionable: {reason}",
        ).__dict__

    return ReasoningResult(
        success=True,
        spec=spec,
        raw_response=raw,
        error=None,
        fallback=False,
        fallback_reason=None,
    ).__dict__


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_spec(raw: str) -> tuple[Optional[RootCauseSpec], Optional[str]]:
    """Extract and validate JSON from model output."""
    # Strip any accidental markdown fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return None, f"JSONDecodeError: {e}"

    required = {"fault_type", "root_cause_explanation", "target_file",
                "remediation_strategy", "confidence"}
    missing = required - data.keys()
    if missing:
        return None, f"Missing fields: {missing}"

    if data["fault_type"] not in FAULT_TYPES:
        # Coerce unknown types rather than hard-failing
        data["fault_type"] = "UNKNOWN"

    try:
        confidence = float(data["confidence"])
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        return None, "confidence is not a valid float"

    return RootCauseSpec(
        fault_type=data["fault_type"],
        root_cause_explanation=str(data["root_cause_explanation"]),
        target_file=str(data["target_file"]),
        remediation_strategy=str(data["remediation_strategy"]),
        confidence=confidence,
    ), None


# ---------------------------------------------------------------------------
# Local smoke-test
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    import base64
    from pathlib import Path

    # Minimal synthetic test
    fake_logs = (
        "2026-11-02T14:32:01Z ERROR  NullPointerException at CheckoutButton.tsx:42\n"
        "2026-11-02T14:32:01Z DEBUG  user=ref_8x92a cart_id=ref_1a2b3c\n"
    )
    fake_diff = (
        "diff --git a/src/components/CheckoutButton.tsx b/src/components/CheckoutButton.tsx\n"
        "--- a/src/components/CheckoutButton.tsx\n"
        "+++ b/src/components/CheckoutButton.tsx\n"
        "@@ -40,7 +40,7 @@\n"
        "-  const label = cart.items[0].name;\n"
        "+  const label = cart?.items?.[0]?.name ?? 'Checkout';\n"
    )
    # 1×1 white pixel PNG as placeholder
    tiny_png = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
    )

    result = analyze_failure.remote(fake_logs, tiny_png, fake_diff)
    print(json.dumps(result, indent=2, default=str))
