"""
github_ops.py — GitHub Git operations for AegisSRE.

Handles all GitHub REST API interactions:
- Fetching file content from a repo
- Creating a fix branch
- Applying a unified diff patch to a branch
- Opening a Pull Request
- Opening a GitHub Issue (circuit breaker fallback)

Auth: Fine-grained PAT injected via Modal Secret "aegis-github-secrets"
      Environment variables: GITHUB_TOKEN, GITHUB_REPO (owner/repo format)

Scopes required on the PAT:
  - contents: write   (read files, push branches)
  - pull_requests: write (open PRs)
  - issues: write     (open fallback issues)
"""

import base64
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from github import Github, GithubException
from github.Repository import Repository


# ---------------------------------------------------------------------------
# Auth helper — works both locally (from .env) and inside Modal (from Secret)
# ---------------------------------------------------------------------------

def _get_github_client() -> tuple[Github, Repository]:
    """
    Return an authenticated Github client and the target repository.
    Reads GITHUB_TOKEN and GITHUB_REPO from environment.
    """
    token = os.environ.get("GITHUB_TOKEN")
    repo_name = os.environ.get("GITHUB_REPO")

    if not token:
        raise EnvironmentError(
            "GITHUB_TOKEN not set. Inject via Modal Secret 'aegis-github-secrets' "
            "or add to .env for local development."
        )
    if not repo_name:
        raise EnvironmentError(
            "GITHUB_REPO not set. Expected format: 'owner/repository-name'."
        )

    client = Github(token)
    repo = client.get_repo(repo_name)
    return client, repo


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class GitOpsResult:
    success: bool
    url: str          # PR or Issue URL
    number: int       # PR or Issue number
    branch: str       # branch name (empty for Issues)
    error: Optional[str]


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

def get_file_content(file_path: str, ref: str = "main") -> str:
    """
    Fetch the raw content of a file from the repository.

    Args:
        file_path:  Relative path within the repo, e.g. "src/components/Button.tsx"
        ref:        Branch or commit SHA to read from. Defaults to "main".

    Returns:
        Decoded file content as a string.
    """
    _, repo = _get_github_client()
    content_file = repo.get_contents(file_path, ref=ref)
    # get_contents returns ContentFile; decode base64
    return base64.b64decode(content_file.content).decode("utf-8")


def get_git_diff(base: str = "HEAD~1", head: str = "HEAD") -> str:
    """
    Return the git diff between two refs using the GitHub Compare API.

    Args:
        base: Base commit/branch (older).
        head: Head commit/branch (newer).

    Returns:
        Unified diff string.
    """
    _, repo = _get_github_client()
    comparison = repo.compare(base, head)
    # Build a unified diff string from the file patches
    diff_lines = []
    for f in comparison.files:
        if f.patch:
            diff_lines.append(
                f"diff --git a/{f.filename} b/{f.filename}\n"
                f"--- a/{f.filename}\n"
                f"+++ b/{f.filename}\n"
                f"{f.patch}"
            )
    return "\n".join(diff_lines)


def create_fix_branch(base_branch: str = "main") -> str:
    """
    Create a new fix branch from base_branch.

    Returns:
        The new branch name, e.g. "fix/aegis-hotfix-20261102-143201"
    """
    _, repo = _get_github_client()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    branch_name = f"fix/aegis-hotfix-{timestamp}"

    base_ref = repo.get_branch(base_branch)
    repo.create_git_ref(
        ref=f"refs/heads/{branch_name}",
        sha=base_ref.commit.sha,
    )
    return branch_name


def apply_patch_to_branch(
    patch: str,
    branch: str,
    commit_message: str = "fix: automated patch by AegisSRE",
) -> bool:
    """
    Apply a unified diff patch to a branch via the GitHub Contents API.

    Parses each file in the diff and updates it with the patched content.

    Args:
        patch:          Unified diff string (output of modal_coder).
        branch:         Target branch name.
        commit_message: Git commit message.

    Returns:
        True if all files were patched successfully.
    """
    _, repo = _get_github_client()

    # Parse the diff to extract (filename, patched_content) pairs
    patched_files = _apply_diff_in_memory(patch, repo, branch)
    if not patched_files:
        raise ValueError("Could not parse any file changes from the patch.")

    for file_path, new_content in patched_files.items():
        content_file = repo.get_contents(file_path, ref=branch)
        repo.update_file(
            path=file_path,
            message=commit_message,
            content=new_content,
            sha=content_file.sha,
            branch=branch,
        )

    return True


def open_pull_request(
    branch: str,
    fault_type: str,
    target_file: str,
    root_cause: str,
    remediation: str,
    confidence: float,
    is_draft: bool = False,
    base_branch: str = "main",
) -> GitOpsResult:
    """
    Open a Pull Request for a generated fix.

    Args:
        branch:       The fix branch to merge from.
        fault_type:   e.g. "DOM_ELEMENT_COLLISION"
        target_file:  File that was patched.
        root_cause:   Human-readable root cause explanation.
        remediation:  Remediation strategy that was applied.
        confidence:   Model confidence score.
        is_draft:     True if the test gate failed — opened as draft.
        base_branch:  Target branch to merge into.

    Returns:
        GitOpsResult with PR URL and number.
    """
    _, repo = _get_github_client()

    prefix = "[DRAFT] Needs Review" if is_draft else "[AegisSRE Fix]"
    title = f"{prefix}: {fault_type} in {Path(target_file).name}"[:72]

    body = _build_pr_body(
        fault_type=fault_type,
        target_file=target_file,
        root_cause=root_cause,
        remediation=remediation,
        confidence=confidence,
        is_draft=is_draft,
    )

    try:
        pr = repo.create_pull(
            title=title,
            body=body,
            head=branch,
            base=base_branch,
            draft=is_draft,
        )
        return GitOpsResult(
            success=True,
            url=pr.html_url,
            number=pr.number,
            branch=branch,
            error=None,
        )
    except GithubException as e:
        return GitOpsResult(
            success=False,
            url="",
            number=0,
            branch=branch,
            error=str(e),
        )


def open_triage_issue(
    fault_type: str,
    fallback_reason: str,
    sanitized_logs: str,
    sam_result: dict,
    screenshot_b64: Optional[str] = None,
) -> GitOpsResult:
    """
    Circuit breaker fallback: open a GitHub Issue for human triage.

    Called when model confidence < 0.80, JSON parsing fails, or test gate fails.

    Returns:
        GitOpsResult with Issue URL and number.
    """
    _, repo = _get_github_client()

    title = f"[AegisSRE Alert]: Manual Triage Required — {fault_type}"

    body = _build_issue_body(
        fault_type=fault_type,
        fallback_reason=fallback_reason,
        sanitized_logs=sanitized_logs,
        sam_result=sam_result,
    )

    try:
        issue = repo.create_issue(
            title=title,
            body=body,
            labels=["aegissre", "needs-triage", "automated"],
        )
        return GitOpsResult(
            success=True,
            url=issue.html_url,
            number=issue.number,
            branch="",
            error=None,
        )
    except GithubException as e:
        return GitOpsResult(
            success=False,
            url="",
            number=0,
            branch="",
            error=str(e),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_diff_in_memory(
    patch: str,
    repo: Repository,
    branch: str,
) -> dict[str, str]:
    """
    Parse a unified diff and apply each hunk to the current file content.

    Returns a dict mapping file_path → new_content.
    Uses Python's `patch` library approach: apply hunks line by line.
    """
    import patch as patch_lib  # python-patch

    # Write patch to a temp file, apply against temp dir
    patched: dict[str, str] = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        patch_file = Path(tmpdir) / "changes.patch"
        patch_file.write_text(patch, encoding="utf-8")

        # Find all target files referenced in the diff
        file_pattern = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)
        target_files = file_pattern.findall(patch)

        for file_path in target_files:
            # Fetch current content from GitHub
            try:
                original = base64.b64decode(
                    repo.get_contents(file_path, ref=branch).content
                ).decode("utf-8")
            except GithubException:
                continue  # File may not exist yet (new file patch)

            # Write original to tmpdir, apply patch
            orig_path = Path(tmpdir) / file_path
            orig_path.parent.mkdir(parents=True, exist_ok=True)
            orig_path.write_text(original, encoding="utf-8")

            pset = patch_lib.fromfile(str(patch_file))
            if pset and pset.apply(root=tmpdir):
                patched[file_path] = orig_path.read_text(encoding="utf-8")

    return patched


def _build_pr_body(
    fault_type: str,
    target_file: str,
    root_cause: str,
    remediation: str,
    confidence: float,
    is_draft: bool,
) -> str:
    status = "⚠️ DRAFT — test gate failed, requires manual review" if is_draft else "✅ Tests passed"
    return f"""## 🛡️ AegisSRE Automated Fix

| Field | Value |
|---|---|
| **Fault Type** | `{fault_type}` |
| **Target File** | `{target_file}` |
| **Model Confidence** | `{confidence:.0%}` |
| **Status** | {status} |

### Root Cause
{root_cause}

### Remediation Applied
{remediation}

---
*Generated autonomously by [AegisSRE](https://github.com). \
All logs were sanitized via BlindLog before model inference. \
Review the diff carefully before merging.*
"""


def _build_issue_body(
    fault_type: str,
    fallback_reason: str,
    sanitized_logs: str,
    sam_result: dict,
) -> str:
    sam_info = (
        f"Fallback active — {sam_result.get('fallback_reason', 'unknown')}"
        if sam_result.get("fallback")
        else f"Confidence: {sam_result.get('confidence', 'N/A'):.0%}, "
             f"BBox: {sam_result.get('bbox', 'N/A')}"
    )
    # Truncate logs to 3000 chars to stay within GitHub Issue limits
    log_preview = sanitized_logs[:3000] + ("\n... [truncated]" if len(sanitized_logs) > 3000 else "")

    return f"""## 🚨 AegisSRE Circuit Breaker Triggered

AegisSRE was unable to autonomously generate a fix for this failure.
**Human triage is required.**

| Field | Value |
|---|---|
| **Fault Type** | `{fault_type}` |
| **Fallback Reason** | {fallback_reason} |
| **SAM 2.1 Vision** | {sam_info} |

### Sanitized Log Trace
*(All PII has been replaced with deterministic ref tokens via BlindLog)*

```
{log_preview}
```

---
*AegisSRE aborted automated remediation to prevent pushing unverified code. \
Attach findings and close this issue once resolved.*
"""
