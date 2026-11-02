"""
modal_coder.py — Code patch synthesis via Muse-Spark on Modal.

Reuses the same OpenAI-compatible Modal inference endpoint as modal_reasoning.py.
No separate GPU worker needed — the 816B model handles both root cause analysis
and code synthesis via two focused prompts.

Input:  target_file path + file content + remediation strategy + root cause
Output: a valid Git Unified Diff (.patch format)
"""

import os
import re
import textwrap

import modal

coder_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "openai==1.57.0",
        "python-dotenv==1.0.1",
        "httpx==0.27.0",   # pin httpx to avoid proxies kwarg conflict
    )
)

app = modal.App("aegissre-coder")

SYSTEM_PROMPT = """\
You are an expert software engineer. Your only job is to produce a minimal, \
correct Git Unified Diff that implements the requested fix.

Output format — ONLY a valid unified diff, nothing else:
--- a/<filepath>
+++ b/<filepath>
@@ -<line>,<count> +<line>,<count> @@
 context line
-removed line
+added line
 context line

Rules:
- Include exactly 3 lines of unchanged context before and after each change.
- Do NOT include any explanation, markdown fences, or text outside the diff.
- Make the smallest possible change that implements the remediation strategy.
- Preserve all existing indentation, formatting, and code style.
- Do NOT rewrite the whole file.
"""


@app.function(
    image=coder_image,
    secrets=[modal.Secret.from_name("aegis-modal-secrets")],
    timeout=120,
    scaledown_window=60,
)
def generate_patch(
    target_file: str,
    file_content: str,
    remediation_strategy: str,
    root_cause_explanation: str,
) -> dict:
    """
    Generate a unified diff patch for the target file.

    Args:
        target_file:             Relative path, e.g. "src/components/Button.tsx"
        file_content:            Raw content of the file to patch.
        remediation_strategy:    From RootCauseSpec — what to change.
        root_cause_explanation:  From RootCauseSpec — why it broke.

    Returns dict:
        success (bool)
        patch   (str | None)  — unified diff string
        error   (str | None)
    """
    from openai import OpenAI

    client = OpenAI(
        base_url="https://a-p-shukla--ep-muse-spark-open-816b-a42b-server.us-west.modal.direct/v1",
        api_key=f"{os.environ['MODAL_PROXY_TOKEN_ID']}.{os.environ['MODAL_PROXY_TOKEN_SECRET']}",
    )

    user_message = textwrap.dedent(f"""\
        FILE: {target_file}

        ROOT CAUSE: {root_cause_explanation}

        REMEDIATION STRATEGY: {remediation_strategy}

        ORIGINAL FILE CONTENT:
        {file_content}

        Produce the unified diff now.
    """)

    try:
        response = client.chat.completions.create(
            model="meta-models/Muse-Spark-Open-816B-A42B",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.05,
            max_tokens=2048,
            top_p=0.95,
            stream=False,
            extra_body={"reasoning_effort": "medium"},
        )
        raw = response.choices[0].message.content.strip()

    except Exception as exc:
        return {"success": False, "patch": None, "error": str(exc)}

    patch, error = _validate_patch(raw, target_file)
    if error:
        return {"success": False, "patch": None, "error": error}

    return {"success": True, "patch": patch, "error": None}


def _validate_patch(raw: str, target_file: str) -> tuple[str | None, str | None]:
    """Strip markdown fences and validate the diff has at least one hunk."""
    cleaned = re.sub(r"^```(?:diff|patch)?\s*", "", raw, flags=re.MULTILINE)
    cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE).strip()

    if not cleaned:
        return None, "Model returned empty output"
    if not re.search(r"^@@\s+-\d+", cleaned, re.MULTILINE):
        return None, "Output missing a valid diff hunk (@@ marker not found)"
    if "---" not in cleaned or "+++" not in cleaned:
        return None, "Output missing --- or +++ diff headers"

    return cleaned, None


@app.local_entrypoint()
def main():
    import json, textwrap

    sample = textwrap.dedent("""\
        import React from 'react';
        export function CheckoutButton({ cart }) {
          const label = cart.items[0].name;
          return <button>{label}</button>;
        }
    """)

    result = generate_patch.remote(
        target_file="src/components/CheckoutButton.tsx",
        file_content=sample,
        remediation_strategy="Use optional chaining cart?.items?.[0]?.name with fallback 'Checkout'",
        root_cause_explanation="cart.items[0] throws when cart is empty",
    )
    print(json.dumps(result, indent=2))
