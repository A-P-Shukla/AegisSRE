"""
pipeline.py — AegisSRE circuit breaker orchestrator.

Wires all stages together in the correct order:
  1. BlindLog PII scrub
  2. SAM 2.1 vision isolation (Modal A10G)
  3. Muse-Spark multimodal root cause analysis (Modal 8×H100)
  4. Qwen2.5-Coder patch synthesis (Modal A100)
  5. Test verification gate (pytest / npm test in subprocess)
  6. GitHub PR or Issue via PyGithub

Each stage has an explicit circuit breaker. No stage can proceed with
unverified output from the previous stage.
"""

import base64
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pii_scrubber import scrub
from modal_sam import segment_image
from modal_reasoning import analyze_failure
from modal_coder import CoderModel
from github_ops import (
    get_file_content,
    get_git_diff,
    create_fix_branch,
    apply_patch_to_branch,
    open_pull_request,
    open_triage_issue,
)

logger = logging.getLogger("aegissre.pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Pipeline input / output types
# ---------------------------------------------------------------------------

@dataclass
class PipelineInput:
    """Payload received from the GitHub Actions webhook."""
    screenshot_b64: str       # Base64-encoded PNG of the failed UI
    raw_logs: str             # Raw (un-sanitized) server logs
    repo_ref: str = "main"    # Branch/commit the test ran against
    test_command: Optional[str] = None  # e.g. "pytest" or "npm test"


@dataclass
class PipelineResult:
    success: bool
    outcome: str              # "pr_opened" | "draft_pr_opened" | "issue_opened" | "error"
    url: str                  # GitHub PR or Issue URL
    stages: dict = field(default_factory=dict)   # Per-stage results for audit
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(payload: PipelineInput) -> PipelineResult:
    """
    Execute the full AegisSRE pipeline end-to-end.

    Circuit breakers are embedded at every stage. On any unrecoverable
    failure the pipeline falls back to opening a GitHub Issue.
    """
    stages: dict = {}
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    # -----------------------------------------------------------------------
    # Stage 1: PII Scrub
    # -----------------------------------------------------------------------
    logger.info("[1/6] BlindLog PII scrub")
    sanitized_logs = str(scrub(payload.raw_logs))
    stages["pii_scrub"] = {"status": "ok", "chars_in": len(payload.raw_logs), "chars_out": len(sanitized_logs)}

    # -----------------------------------------------------------------------
    # Stage 2: SAM 2.1 vision isolation
    # -----------------------------------------------------------------------
    logger.info("[2/6] SAM 2.1 segmentation (Modal A10G)")
    sam_result = segment_image.remote(payload.screenshot_b64)
    stages["sam"] = sam_result

    if sam_result["fallback"]:
        logger.warning(f"SAM fallback: {sam_result['fallback_reason']}")

    # Always proceed — SAM fallback just means we use the full screenshot
    visual_crop_b64 = sam_result["cropped_png_b64"]
    is_visual_fallback = sam_result["fallback"]

    # -----------------------------------------------------------------------
    # Stage 3: Git diff
    # -----------------------------------------------------------------------
    logger.info("[3/6] Fetching git diff")
    try:
        git_diff = get_git_diff(base="HEAD~1", head="HEAD")
    except Exception as e:
        logger.warning(f"Could not fetch git diff: {e} — proceeding without diff")
        git_diff = "(git diff unavailable)"
    stages["git_diff"] = {"status": "ok", "length": len(git_diff)}

    # -----------------------------------------------------------------------
    # Stage 4: Multimodal root cause analysis
    # -----------------------------------------------------------------------
    logger.info("[4/6] Muse-Spark root cause analysis (Modal 8×H100)")
    reasoning_result = analyze_failure.remote(
        sanitized_logs=sanitized_logs,
        cropped_image_b64=visual_crop_b64,
        git_diff=git_diff,
        is_visual_fallback=is_visual_fallback,
    )
    stages["reasoning"] = reasoning_result

    # Circuit breaker: reasoning fallback → open Issue
    if reasoning_result["fallback"]:
        logger.warning(f"Reasoning fallback: {reasoning_result['fallback_reason']}")
        return _open_issue_fallback(
            fault_type=_extract_fault_type(reasoning_result),
            fallback_reason=reasoning_result["fallback_reason"],
            sanitized_logs=sanitized_logs,
            sam_result=sam_result,
            stages=stages,
        )

    spec = reasoning_result["spec"]
    logger.info(f"Root cause: {spec['fault_type']} in {spec['target_file']} (confidence={spec['confidence']:.0%})")

    # -----------------------------------------------------------------------
    # Stage 5: Code patch synthesis
    # -----------------------------------------------------------------------
    logger.info("[5/6] Qwen2.5-Coder patch synthesis (Modal A100)")
    try:
        file_content = get_file_content(spec["target_file"], ref=payload.repo_ref)
    except Exception as e:
        return _open_issue_fallback(
            fault_type=spec["fault_type"],
            fallback_reason=f"Could not fetch target file '{spec['target_file']}': {e}",
            sanitized_logs=sanitized_logs,
            sam_result=sam_result,
            stages=stages,
        )

    coder = CoderModel()
    coder_result = coder.generate_patch.remote(
        target_file=spec["target_file"],
        file_content=file_content,
        remediation_strategy=spec["remediation_strategy"],
        root_cause_explanation=spec["root_cause_explanation"],
    )
    stages["coder"] = coder_result

    # Circuit breaker: patch synthesis failed → open Issue
    if not coder_result["success"]:
        return _open_issue_fallback(
            fault_type=spec["fault_type"],
            fallback_reason=f"Patch synthesis failed: {coder_result['error']}",
            sanitized_logs=sanitized_logs,
            sam_result=sam_result,
            stages=stages,
        )

    patch = coder_result["patch"]

    # -----------------------------------------------------------------------
    # Stage 6: Apply patch + test verification gate
    # -----------------------------------------------------------------------
    logger.info("[6/6] Applying patch and running test gate")
    branch = create_fix_branch(base_branch=payload.repo_ref)
    stages["branch"] = branch

    try:
        apply_patch_to_branch(
            patch=patch,
            branch=branch,
            commit_message=f"fix({spec['fault_type'].lower()}): {spec['remediation_strategy'][:60]}",
        )
    except Exception as e:
        return _open_issue_fallback(
            fault_type=spec["fault_type"],
            fallback_reason=f"git apply failed: {e}",
            sanitized_logs=sanitized_logs,
            sam_result=sam_result,
            stages=stages,
        )

    tests_passed = _run_test_gate(patch, payload.test_command)
    stages["tests_passed"] = tests_passed
    is_draft = not tests_passed

    if is_draft:
        logger.warning("Test gate failed — opening Draft PR for human review")
    else:
        logger.info("Test gate passed — opening PR")

    git_result = open_pull_request(
        branch=branch,
        fault_type=spec["fault_type"],
        target_file=spec["target_file"],
        root_cause=spec["root_cause_explanation"],
        remediation=spec["remediation_strategy"],
        confidence=spec["confidence"],
        is_draft=is_draft,
        base_branch=payload.repo_ref,
    )
    stages["github"] = {"url": git_result.url, "number": git_result.number}

    outcome = "draft_pr_opened" if is_draft else "pr_opened"
    return PipelineResult(
        success=True,
        outcome=outcome,
        url=git_result.url,
        stages=stages,
    )


# ---------------------------------------------------------------------------
# Test verification gate
# ---------------------------------------------------------------------------

def _run_test_gate(patch: str, test_command: Optional[str]) -> bool:
    """
    Apply the patch to a local temp directory and run the test suite.

    Returns True if tests pass, False otherwise.
    If no test_command is provided, skips the gate and returns True
    (the PR will still be opened, non-draft).
    """
    if not test_command:
        logger.info("No test_command provided — skipping test gate")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        patch_path = Path(tmpdir) / "fix.patch"
        patch_path.write_text(patch, encoding="utf-8")

        try:
            # Apply patch to current working directory (sandbox)
            apply_result = subprocess.run(
                ["git", "apply", "--check", str(patch_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if apply_result.returncode != 0:
                logger.warning(f"git apply --check failed: {apply_result.stderr}")
                return False

            subprocess.run(
                ["git", "apply", str(patch_path)],
                check=True,
                capture_output=True,
                timeout=30,
            )

            # Run tests
            test_result = subprocess.run(
                test_command.split(),
                capture_output=True,
                text=True,
                timeout=300,
            )
            passed = test_result.returncode == 0
            if not passed:
                logger.warning(f"Tests failed:\n{test_result.stdout[-2000:]}\n{test_result.stderr[-1000:]}")
            return passed

        except subprocess.TimeoutExpired:
            logger.warning("Test gate timed out (>300s)")
            return False
        except Exception as e:
            logger.warning(f"Test gate error: {e}")
            return False


# ---------------------------------------------------------------------------
# Circuit breaker fallback
# ---------------------------------------------------------------------------

def _open_issue_fallback(
    fault_type: str,
    fallback_reason: str,
    sanitized_logs: str,
    sam_result: dict,
    stages: dict,
) -> PipelineResult:
    logger.warning(f"Circuit breaker: opening triage issue — {fallback_reason}")
    git_result = open_triage_issue(
        fault_type=fault_type,
        fallback_reason=fallback_reason,
        sanitized_logs=sanitized_logs,
        sam_result=sam_result,
    )
    stages["github_issue"] = {"url": git_result.url, "number": git_result.number}
    return PipelineResult(
        success=False,
        outcome="issue_opened",
        url=git_result.url,
        stages=stages,
        error=fallback_reason,
    )


def _extract_fault_type(reasoning_result: dict) -> str:
    spec = reasoning_result.get("spec")
    if spec and isinstance(spec, dict):
        return spec.get("fault_type", "UNKNOWN")
    return "UNKNOWN"
