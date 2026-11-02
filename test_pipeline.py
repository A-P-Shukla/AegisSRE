"""
test_pipeline.py — End-to-end integration test for AegisSRE.

Tests every stage of the pipeline in sequence using synthetic CI failure data.
No real GitHub repo required.

Modal stages are called via modal.Function.lookup() which connects to already-
deployed apps. Run `modal deploy` on each worker first, then run this test.

Run with:
    modal run test_pipeline.py

Stages tested:
    1. BlindLog PII scrub          (local)
    2. SAM 2.1 vision isolation    (Modal — aegissre-sam)
    3. Muse-Spark root cause RCA   (Modal — aegissre-reasoning)
    4. Muse-Spark code synthesis   (Modal — aegissre-coder)
    5. Circuit breaker logic       (local)
"""

import base64
import io
import json
import sys
import textwrap
import traceback

import modal
from dotenv import load_dotenv

load_dotenv()

# ── Helpers ───────────────────────────────────────────────────────────────────

PASS = "✅ PASS"
FAIL = "❌ FAIL"
SKIP = "⚠️  SKIP"

results: list[dict] = []


def section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def record(stage: str, passed: bool, detail: str = "", skip: bool = False):
    icon = SKIP if skip else (PASS if passed else FAIL)
    print(f"  {icon}  {stage}")
    if detail:
        for line in detail.splitlines():
            print(f"       {line}")
    results.append({"stage": stage, "passed": passed or skip, "skip": skip})


def make_test_png() -> str:
    """Return a base64-encoded 100×60 synthetic 'broken UI' PNG."""
    from PIL import Image as PILImage, ImageDraw
    img = PILImage.new("RGB", (100, 60), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 90, 50], outline=(255, 0, 0), width=2)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Synthetic test data ───────────────────────────────────────────────────────

SYNTHETIC_LOGS = textwrap.dedent("""\
    2026-11-02T14:32:00Z INFO  Server started on port 3000
    2026-11-02T14:32:01Z ERROR TypeError: Cannot read properties of undefined (reading 'name')
        at CheckoutButton (src/components/CheckoutButton.tsx:42:28)
    2026-11-02T14:32:01Z DEBUG user=john.doe@example.com cart_id=cart_abc123 session=tok_xyz789
    2026-11-02T14:32:01Z WARN  Cart items array is empty for user john.doe@example.com
    2026-11-02T14:32:02Z ERROR React ErrorBoundary: cart.items[0] is undefined
""")

SYNTHETIC_DIFF = textwrap.dedent("""\
    diff --git a/src/components/CheckoutButton.tsx b/src/components/CheckoutButton.tsx
    --- a/src/components/CheckoutButton.tsx
    +++ b/src/components/CheckoutButton.tsx
    @@ -39,7 +39,7 @@
       const router = useRouter();
       const handleClick = () => router.push('/checkout');
    -  const label = cart.items[0].name;
    +  const label = cart.items[0]?.name ?? 'Checkout';
       return (
""")

SYNTHETIC_FILE = textwrap.dedent("""\
    import React from 'react';
    import { useRouter } from 'next/router';

    interface CartItem { name: string; price: number; }
    interface Cart { items: CartItem[]; }
    interface Props { cart: Cart; }

    export function CheckoutButton({ cart }: Props) {
      const router = useRouter();
      const handleClick = () => router.push('/checkout');
      const label = cart.items[0].name;
      return (
        <button onClick={handleClick} className="checkout-btn">
          {label}
        </button>
      );
    }
""")


# ── Modal app entry point ─────────────────────────────────────────────────────

app = modal.App("aegissre-test")


@app.local_entrypoint()
def main():
    sanitized_logs = run_all()
    total   = len(results)
    passed  = sum(1 for r in results if r["passed"])
    failed  = sum(1 for r in results if not r["passed"])
    skipped = sum(1 for r in results if r.get("skip"))

    section("Test Summary")
    print(f"  Total:   {total}")
    print(f"  Passed:  {passed}")
    print(f"  Failed:  {failed}")
    print(f"  Skipped: {skipped}")
    print()

    if failed == 0:
        print("  ✅  All stages passed — AegisSRE pipeline is operational.")
    else:
        print("  ❌  Some stages failed:")
        for r in results:
            if not r["passed"]:
                print(f"       → {r['stage']}")


def run_all() -> str:
    """Run all test stages. Returns sanitized logs string."""

    # ── Stage 1: BlindLog PII Scrub ───────────────────────────────────────────
    section("Stage 1 — BlindLog PII Scrub (local)")
    sanitized_logs = SYNTHETIC_LOGS

    try:
        from pii_scrubber import scrub
        sanitized_logs = str(scrub(SYNTHETIC_LOGS))

        email_gone   = "john.doe@example.com" not in sanitized_logs
        session_gone = "tok_xyz789" not in sanitized_logs

        record("Email masked",    email_gone,
               next((l for l in sanitized_logs.splitlines() if "blnd" in l or "masked" in l), ""))
        record("Session token masked", session_gone,
               next((l for l in sanitized_logs.splitlines() if "session" in l), ""))
        record("PII scrub stage", email_gone and session_gone)

    except Exception:
        record("PII scrub stage", False, traceback.format_exc())

    # ── Stage 2: SAM 2.1 ─────────────────────────────────────────────────────
    section("Stage 2 — SAM 2.1 Vision Isolation (Modal)")
    visual_crop_b64 = make_test_png()
    is_visual_fallback = True

    try:
        segment_image = modal.Function.from_name("aegissre-sam", "segment_image")
        print("  Calling segment_image.remote() ...")
        sam_result = segment_image.remote(make_test_png())

        record("SAM returned response",   isinstance(sam_result, dict))
        record("cropped_png_b64 present", bool(sam_result.get("cropped_png_b64")))

        if sam_result.get("fallback"):
            record("SAM confidence gate", True, skip=True,
                   detail=f"fallback: {sam_result.get('fallback_reason')}")
        else:
            record("SAM confidence gate",
                   sam_result.get("confidence", 0) >= 0.70,
                   f"confidence={sam_result.get('confidence', 0):.2f}  bbox={sam_result.get('bbox')}")

        visual_crop_b64    = sam_result["cropped_png_b64"]
        is_visual_fallback = sam_result["fallback"]
        print(f"  masks={sam_result.get('mask_count', 0)}  fallback={is_visual_fallback}")

    except Exception:
        record("SAM stage", False, traceback.format_exc())

    # ── Stage 3: Root Cause Analysis ─────────────────────────────────────────
    section("Stage 3 — Muse-Spark Root Cause Analysis (Modal)")
    spec = None

    try:
        analyze_failure = modal.Function.from_name("aegissre-reasoning", "analyze_failure")
        print("  Calling analyze_failure.remote() ...")
        reasoning_result = analyze_failure.remote(
            sanitized_logs=sanitized_logs,
            cropped_image_b64=visual_crop_b64,
            git_diff=SYNTHETIC_DIFF,
            is_visual_fallback=is_visual_fallback,
        )

        record("Reasoning returned response", isinstance(reasoning_result, dict))

        if reasoning_result.get("fallback"):
            record("Reasoning confidence gate", False,
                   f"fallback: {reasoning_result.get('fallback_reason')}\n"
                   f"raw ({len(reasoning_result.get('raw_response',''))} chars): '{reasoning_result.get('raw_response', '')[:500]}'\n"
                   f"error: {reasoning_result.get('error')}")
        else:
            spec = reasoning_result.get("spec", {})
            # spec comes back as a dict from the Modal remote call
            if hasattr(spec, "__getitem__"):
                get = lambda k: spec.get(k)      # dict
            else:
                get = lambda k: getattr(spec, k, None)  # dataclass

            record("fault_type present",           bool(get("fault_type")),
                   f"fault_type: {get('fault_type')}")
            record("target_file present",          bool(get("target_file")),
                   f"target_file: {get('target_file')}")
            record("remediation_strategy present", bool(get("remediation_strategy")),
                   f"strategy: {str(get('remediation_strategy') or '')[:80]}")
            record("confidence >= 0.80",           (get("confidence") or 0) >= 0.80,
                   f"confidence: {get('confidence')}")
            print(f"\n  Root Cause Spec:\n{json.dumps(spec if isinstance(spec, dict) else spec.__dict__, indent=4)}")

    except Exception:
        record("Reasoning stage", False, traceback.format_exc())

    # ── Stage 4: Code Patch Synthesis ────────────────────────────────────────
    section("Stage 4 — Code Patch Synthesis (Modal)")
    patch = None

    try:
        generate_patch = modal.Function.from_name("aegissre-coder", "generate_patch")

        if spec and (spec.get("target_file") if isinstance(spec, dict) else getattr(spec, "target_file", None)):
            target_file = spec.get("target_file") if isinstance(spec, dict) else spec.target_file
            remediation = spec.get("remediation_strategy") if isinstance(spec, dict) else spec.remediation_strategy
            root_cause  = spec.get("root_cause_explanation") if isinstance(spec, dict) else spec.root_cause_explanation
        else:
            print("  (using synthetic remediation — Stage 3 spec unavailable)")
            target_file = "src/components/CheckoutButton.tsx"
            remediation = "Use optional chaining cart?.items?.[0]?.name with fallback 'Checkout'"
            root_cause  = "cart.items[0] throws TypeError when cart.items is empty"

        print(f"  Calling generate_patch.remote() for {target_file} ...")
        coder_result = generate_patch.remote(
            target_file=target_file,
            file_content=SYNTHETIC_FILE,
            remediation_strategy=remediation,
            root_cause_explanation=root_cause,
        )

        record("Coder returned response", isinstance(coder_result, dict))
        record("Patch generated",         coder_result.get("success", False),
               f"error: {coder_result.get('error')}" if not coder_result.get("success") else "")

        if coder_result.get("patch"):
            patch = coder_result["patch"]
            lines = patch.splitlines()
            record("Patch has @@ hunk header",    any("@@" in l for l in lines))
            record("Patch has removal lines (-)",  any(l.startswith("-") and not l.startswith("---") for l in lines))
            record("Patch has addition lines (+)", any(l.startswith("+") and not l.startswith("+++") for l in lines))
            print(f"\n  Patch ({len(patch)} chars):")
            print("  " + "\n  ".join(lines[:25]))

    except Exception:
        record("Coder stage", False, traceback.format_exc())

    # ── Stage 5: Circuit Breaker Logic ───────────────────────────────────────
    section("Stage 5 — Circuit Breaker Logic (local)")

    try:
        from pipeline import _extract_fault_type, _run_test_gate

        record("Test gate skips when no command",
               _run_test_gate(patch or "", test_command=None) is True)
        record("Fault type extracted (good result)",
               _extract_fault_type({"fallback": False, "spec": {"fault_type": "NULL_POINTER"}}) == "NULL_POINTER")
        record("Fault type fallback (bad result)",
               _extract_fault_type({"fallback": True, "spec": None}) == "UNKNOWN")
        record("No raw PII reaches model",
               "john.doe@example.com" not in sanitized_logs and
               "tok_xyz789" not in sanitized_logs)

    except Exception:
        record("Circuit breaker stage", False, traceback.format_exc())

    return sanitized_logs
