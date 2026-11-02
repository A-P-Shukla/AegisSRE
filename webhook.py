"""
webhook.py — FastAPI webhook receiver for AegisSRE.

Listens for POST /webhook from a GitHub Actions workflow on CI/CD failure.
Validates the HMAC-SHA256 signature, queues the pipeline run, and returns
immediately so the GitHub Action does not time out.

Endpoints:
  POST /webhook          — receive CI failure payload
  GET  /health           — liveness probe
  GET  /status/{run_id}  — poll async run status

Deploy locally:  uvicorn webhook:web_app --host 0.0.0.0 --port 8000
Deploy on Modal: modal deploy webhook.py
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import modal
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field

from pipeline import PipelineInput, PipelineResult, run_pipeline

load_dotenv()

logger = logging.getLogger("aegissre.webhook")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# In-memory run status store
# (replace with Redis or Modal Dict for multi-instance deployments)
# ---------------------------------------------------------------------------
_run_store: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Pydantic request model
# ---------------------------------------------------------------------------

class WebhookPayload(BaseModel):
    """
    Expected JSON body from the GitHub Actions step.

    The CI workflow should POST this payload when a test fails:

      - name: Notify AegisSRE
        if: failure()
        run: |
          curl -s -X POST ${{ secrets.AEGISSRE_WEBHOOK_URL }}/webhook \
            -H "Content-Type: application/json" \
            -H "X-AegisSRE-Signature: ${{ steps.sign.outputs.sig }}" \
            -d '{
              "screenshot_b64": "${{ steps.screenshot.outputs.b64 }}",
              "raw_logs": "${{ steps.logs.outputs.text }}",
              "repo_ref": "${{ github.sha }}",
              "test_command": "pytest"
            }'
    """
    screenshot_b64: str = Field(..., description="Base64-encoded PNG screenshot of the failed UI")
    raw_logs: str = Field(..., description="Raw (un-sanitized) server/test logs")
    repo_ref: str = Field("main", description="Branch name or commit SHA the test ran against")
    test_command: Optional[str] = Field(None, description="Test command for the verification gate, e.g. 'pytest'")


# ---------------------------------------------------------------------------
# FastAPI application  (named web_app to avoid clashing with Modal's app)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(web_app: FastAPI):
    logger.info("AegisSRE webhook receiver started")
    yield
    logger.info("AegisSRE webhook receiver shutting down")


web_app = FastAPI(
    title="AegisSRE Webhook",
    description="Autonomous SRE pipeline trigger endpoint",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@web_app.get("/health")
async def health():
    """Liveness probe for load balancers and GitHub Action connectivity checks."""
    return {"status": "ok", "service": "aegissre-webhook"}


@web_app.get("/status/{run_id}")
async def get_run_status(run_id: str):
    """Poll the status of an async pipeline run."""
    if run_id not in _run_store:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return _run_store[run_id]


@web_app.post("/webhook", status_code=202)
async def receive_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_aegissre_signature: Optional[str] = Header(None),
):
    """
    Receive a CI/CD failure notification and trigger the AegisSRE pipeline.

    Returns 202 Accepted immediately. The pipeline runs asynchronously.
    Poll /status/{run_id} for results.
    """
    raw_body = await request.body()

    # --- HMAC signature validation ---
    _verify_signature(raw_body, x_aegissre_signature)

    # --- Parse body ---
    try:
        body = json.loads(raw_body)
        payload = WebhookPayload(**body)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid payload: {e}")

    # --- Generate run ID and register ---
    run_id = str(uuid.uuid4())
    _run_store[run_id] = {"run_id": run_id, "status": "queued", "result": None}

    # --- Queue pipeline as background task ---
    background_tasks.add_task(_execute_pipeline, run_id, payload)

    logger.info(f"Pipeline queued — run_id={run_id}")
    return {"run_id": run_id, "status": "queued", "poll_url": f"/status/{run_id}"}


# ---------------------------------------------------------------------------
# Background execution
# ---------------------------------------------------------------------------

async def _execute_pipeline(run_id: str, payload: WebhookPayload):
    """Run the pipeline in a background thread (Modal calls are blocking)."""
    _run_store[run_id]["status"] = "running"
    try:
        pipeline_input = PipelineInput(
            screenshot_b64=payload.screenshot_b64,
            raw_logs=payload.raw_logs,
            repo_ref=payload.repo_ref,
            test_command=payload.test_command,
        )
        # run_pipeline makes Modal remote calls — run in thread to avoid blocking event loop
        result: PipelineResult = await asyncio.get_event_loop().run_in_executor(
            None, run_pipeline, pipeline_input
        )
        _run_store[run_id] = {
            "run_id": run_id,
            "status": "completed",
            "outcome": result.outcome,
            "url": result.url,
            "stages": result.stages,
            "error": result.error,
        }
        logger.info(f"Pipeline completed — run_id={run_id} outcome={result.outcome} url={result.url}")

    except Exception as e:
        logger.exception(f"Pipeline error — run_id={run_id}: {e}")
        _run_store[run_id] = {
            "run_id": run_id,
            "status": "error",
            "outcome": "error",
            "url": "",
            "stages": {},
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Security: HMAC-SHA256 signature verification
# ---------------------------------------------------------------------------

def _verify_signature(body: bytes, signature_header: Optional[str]):
    """
    Verify the HMAC-SHA256 signature on the incoming webhook.

    The GitHub Action should sign the payload with AEGISSRE_WEBHOOK_SECRET:
      sig = "sha256=" + hmac.new(secret, body, sha256).hexdigest()

    Raises HTTP 401 if verification fails.
    Set AEGISSRE_SKIP_SIGNATURE_VERIFICATION=true to bypass during local dev.
    """
    if os.getenv("AEGISSRE_SKIP_SIGNATURE_VERIFICATION", "").lower() == "true":
        logger.warning("Signature verification SKIPPED — development mode only")
        return

    secret = os.getenv("AEGISSRE_WEBHOOK_SECRET")
    if not secret:
        raise HTTPException(
            status_code=500,
            detail="AEGISSRE_WEBHOOK_SECRET not configured on server",
        )

    if not signature_header:
        raise HTTPException(
            status_code=401,
            detail="Missing X-AegisSRE-Signature header",
        )

    expected = "sha256=" + hmac.new(
        secret.encode("utf-8"), body, hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(expected, signature_header):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")


# ---------------------------------------------------------------------------
# Modal ASGI deployment
# The Modal App must be named 'app' at module level so 'modal deploy webhook.py'
# finds it automatically without needing an explicit --app flag.
# ---------------------------------------------------------------------------

modal_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi==0.111.0",
        "uvicorn==0.30.1",
        "python-dotenv==1.0.1",
        "blindlog==1.2.1",
        "PyGithub==2.3.0",
        "patch==1.16",
    )
)

app = modal.App("aegissre-webhook")


@app.function(
    image=modal_image,
    secrets=[
        modal.Secret.from_name("aegis-modal-secrets"),
        modal.Secret.from_name("aegis-github-secrets"),
        modal.Secret.from_name("aegis-webhook-secrets"),
    ],
    min_containers=1,
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def serve():
    return web_app
