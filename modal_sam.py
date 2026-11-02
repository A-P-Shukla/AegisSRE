"""
modal_sam.py — SAM 2.1 vision worker hosted on Modal (A10G GPU).

Receives a base64-encoded PNG screenshot, runs SAM 2.1 segmentation,
and returns the highest-confidence anomaly bounding box + a cropped PNG.

Deploy:  modal deploy modal_sam.py
Call:    segment_image.remote(image_b64: str) -> SAMResult dict
"""

import base64
import io
from dataclasses import dataclass
from typing import Optional

import modal

# ---------------------------------------------------------------------------
# Container image — installs SAM 2.1 and its dependencies
# ---------------------------------------------------------------------------
sam_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        "pillow==10.3.0",
        "numpy==1.26.4",
        "huggingface_hub==0.23.4",
    )
    # Install SAM 2 directly from Meta's repo
    .run_commands(
        "pip install git+https://github.com/facebookresearch/sam2.git"
    )
)

app = modal.App("aegissre-sam")

# ---------------------------------------------------------------------------
# Model weights volume — downloaded once, reused across cold starts
# ---------------------------------------------------------------------------
weights_volume = modal.Volume.from_name("aegissre-sam-weights", create_if_missing=True)
WEIGHTS_DIR = "/weights"
MODEL_ID = "facebook/sam2.1-hiera-tiny"      # ~38MB — practical on CPU, no GPU needed
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
CHECKPOINT = "sam2.1_hiera_tiny.pt"


@app.function(
    image=sam_image,

    volumes={WEIGHTS_DIR: weights_volume},
    timeout=600,   # SAM weights download + CPU inference can take several minutes
    scaledown_window=60,
)
def segment_image(image_b64: str, prompt_points: Optional[list] = None) -> dict:
    """
    Segment a screenshot to isolate visual anomalies.

    Args:
        image_b64:      Base64-encoded PNG screenshot from CI/CD failure.
        prompt_points:  Optional list of [[x, y]] point prompts. If None,
                        runs automatic grid-based mask generation.

    Returns a dict with keys:
        success (bool)
        confidence (float)          — highest-scoring mask confidence
        bbox (list[int])            — [x1, y1, x2, y2] of best mask
        cropped_png_b64 (str)       — base64 PNG of the cropped anomaly region
        mask_count (int)            — total masks generated
        fallback (bool)             — True if confidence < 0.70 or 0 masks found
        fallback_reason (str | None)
    """
    import os
    import numpy as np
    from PIL import Image
    from huggingface_hub import snapshot_download
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    CONFIDENCE_THRESHOLD = 0.70

    # --- 1. Download / load weights (cached in volume) ---
    local_model_dir = os.path.join(WEIGHTS_DIR, "sam2.1-hiera-tiny")
    if not os.path.exists(local_model_dir):
        snapshot_download(repo_id=MODEL_ID, local_dir=local_model_dir)
        weights_volume.commit()

    checkpoint = os.path.join(local_model_dir, CHECKPOINT)

    sam2_model = build_sam2(MODEL_CFG, checkpoint, device="cpu")

    # --- 2. Decode input image ---
    image_bytes = base64.b64decode(image_b64)
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    np_image = np.array(pil_image)

    # --- 3. Run segmentation ---
    if prompt_points:
        # Prompted mode: use provided click points
        predictor = SAM2ImagePredictor(sam2_model)
        predictor.set_image(np_image)
        points = np.array(prompt_points, dtype=np.float32)
        labels = np.ones(len(points), dtype=np.int32)  # all foreground
        masks, scores, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
        # Build mask list matching auto-generator output shape
        mask_data = [
            {"segmentation": masks[i], "stability_score": float(scores[i])}
            for i in range(len(masks))
        ]
    else:
        # Automatic grid-based generation — no prompt needed
        generator = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=32,
            pred_iou_thresh=0.60,
            stability_score_thresh=0.85,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
        )
        mask_data = generator.generate(np_image)

    # --- 4. Guard: zero masks ---
    if not mask_data:
        return _fallback_result(pil_image, "SAM returned 0 masks")

    # --- 5. Pick highest-confidence mask ---
    best_mask = max(mask_data, key=lambda m: m["stability_score"])
    confidence = float(best_mask["stability_score"])

    if confidence < CONFIDENCE_THRESHOLD:
        return _fallback_result(
            pil_image,
            f"Best mask confidence {confidence:.2f} < threshold {CONFIDENCE_THRESHOLD}",
        )

    # --- 6. Derive bounding box from mask ---
    seg = best_mask["segmentation"]  # bool H×W array
    rows = np.any(seg, axis=1)
    cols = np.any(seg, axis=0)
    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]
    y1, y2 = int(row_indices[0]),  int(row_indices[-1])
    x1, x2 = int(col_indices[0]),  int(col_indices[-1])
    bbox = [x1, y1, x2, y2]

    # --- 7. Crop anomaly region (add 8px padding) ---
    pad = 8
    h, w = np_image.shape[:2]
    crop = pil_image.crop((
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(w, x2 + pad),
        min(h, y2 + pad),
    ))
    cropped_b64 = _pil_to_b64(crop)

    return {
        "success": True,
        "confidence": confidence,
        "bbox": bbox,
        "cropped_png_b64": cropped_b64,
        "mask_count": len(mask_data),
        "fallback": False,
        "fallback_reason": None,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil_to_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _fallback_result(full_image, reason: str) -> dict:
    """Return the full screenshot when SAM cannot isolate an anomaly."""
    return {
        "success": False,
        "confidence": 0.0,
        "bbox": None,
        "cropped_png_b64": _pil_to_b64(full_image),  # full page as fallback
        "mask_count": 0,
        "fallback": True,
        "fallback_reason": reason,
    }


# ---------------------------------------------------------------------------
# Local test entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(image: str = "test_screenshot.png"):
    """Quick smoke-test: encode a local PNG and call the remote function."""
    with open(image, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    print(f"Sending {image} to SAM 2.1 ...")
    result = segment_image.remote(b64)
    print(result)
