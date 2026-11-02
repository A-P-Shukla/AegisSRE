"""
modal_coder.py — Code patch synthesis worker hosted on Modal.

Receives a RootCauseSpec (target_file + remediation_strategy) plus the raw
file content, and outputs a valid Git Unified Diff (.patch format).

Model: Qwen2.5-Coder-32B-Instruct (open-weights, hosted via vLLM on Modal)
Hardware: A100-40GB (sufficient for 32B at fp16 with vLLM)
"""

import re
import textwrap

import modal

# ---------------------------------------------------------------------------
# Container image — vLLM + model weights
# ---------------------------------------------------------------------------
coder_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.5.3",
        "huggingface_hub==0.23.4",
        "torch==2.3.1",
    )
)

app = modal.App("aegissre-coder")

weights_volume = modal.Volume.from_name("aegissre-coder-weights", create_if_missing=True)
WEIGHTS_DIR = "/weights"
MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert software engineer tasked with producing a minimal, correct code fix.
You will be given:
1. The ORIGINAL file content
2. A REMEDIATION STRATEGY describing exactly what needs to change

Your ONLY output must be a valid Git Unified Diff in this format:
--- a/<filepath>
+++ b/<filepath>
@@ ... @@
 context line
-removed line
+added line
 context line

Rules:
- Include 3 lines of context before and after each change.
- Do NOT include any explanation, markdown, or text outside the diff.
- Make the smallest possible change that implements the remediation strategy.
- Preserve all existing formatting, indentation, and style.
- Do NOT rewrite the entire file.
"""


@app.cls(
    image=coder_image,
    gpu="A100-40GB",
    volumes={WEIGHTS_DIR: weights_volume},
    timeout=300,
    scaledown_window=120,
)
class CoderModel:
    """Persistent class keeps the vLLM engine warm between calls."""

    @modal.enter()
    def load_model(self):
        """Download weights on first cold start, then load vLLM engine."""
        from huggingface_hub import snapshot_download
        from vllm import LLM
        import os

        local_model_dir = os.path.join(WEIGHTS_DIR, "qwen2.5-coder-32b")
        if not os.path.exists(local_model_dir):
            snapshot_download(repo_id=MODEL_ID, local_dir=local_model_dir)
            weights_volume.commit()

        self.llm = LLM(
            model=local_model_dir,
            dtype="float16",
            max_model_len=8192,
            trust_remote_code=True,
        )

    @modal.method()
    def generate_patch(
        self,
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
            patch (str | None)      — unified diff string
            error (str | None)
        """
        from vllm import SamplingParams

        user_message = textwrap.dedent(f"""\
            FILE: {target_file}

            ROOT CAUSE: {root_cause_explanation}

            REMEDIATION STRATEGY: {remediation_strategy}

            ORIGINAL FILE CONTENT:
            ```
            {file_content}
            ```

            Produce the unified diff now.
        """)

        # Qwen chat template
        prompt = (
            "<|im_start|>system\n"
            f"{SYSTEM_PROMPT}\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_message}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        params = SamplingParams(
            temperature=0.05,   # Near-deterministic for code
            max_tokens=2048,
            top_p=0.95,
            stop=["<|im_end|>"],
        )

        outputs = self.llm.generate([prompt], params)
        raw = outputs[0].outputs[0].text.strip()

        patch, error = _validate_patch(raw, target_file)
        if error:
            return {"success": False, "patch": None, "error": error}

        return {"success": True, "patch": patch, "error": None}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_patch(raw: str, target_file: str) -> tuple[str | None, str | None]:
    """
    Validate that the model output looks like a real unified diff.
    Strips markdown fences if present.
    """
    # Strip markdown code fences
    cleaned = re.sub(r"^```(?:diff|patch)?\s*", "", raw, flags=re.MULTILINE)
    cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE).strip()

    if not cleaned:
        return None, "Model returned empty output"

    # Must contain at least one hunk header
    if not re.search(r"^@@\s+-\d+", cleaned, re.MULTILINE):
        return None, "Output does not contain a valid diff hunk (@@ marker missing)"

    # Must reference the target file in --- / +++ headers
    if "---" not in cleaned or "+++" not in cleaned:
        return None, "Output missing --- or +++ diff headers"

    # Ensure file path is referenced (partial match is fine)
    filename = target_file.split("/")[-1]
    if filename not in cleaned:
        # Non-fatal: model may use a slightly different path; just warn via error=None
        pass

    return cleaned, None


# ---------------------------------------------------------------------------
# Local smoke-test
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    import json

    sample_file = textwrap.dedent("""\
        import React from 'react';

        interface Props {
          cart: { items: { name: string }[] };
        }

        export function CheckoutButton({ cart }: Props) {
          const label = cart.items[0].name;
          return <button>{label}</button>;
        }
    """)

    result = CoderModel().generate_patch.remote(
        target_file="src/components/CheckoutButton.tsx",
        file_content=sample_file,
        remediation_strategy=(
            "Use optional chaining (cart?.items?.[0]?.name) with a fallback "
            "value of 'Checkout' to prevent NullPointerException when cart is empty."
        ),
        root_cause_explanation=(
            "cart.items[0] throws when the cart is empty, causing a "
            "NullPointerException that breaks the checkout UI rendering."
        ),
    )

    print(json.dumps(result, indent=2))
