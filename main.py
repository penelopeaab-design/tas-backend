# main.py
import os
import subprocess
import json
import asyncio
import tempfile
from pathlib import Path
import logging
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Image processing
from PIL import Image
import numpy as np
import cv2

# ---------------------------
# Logging setup
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# FastAPI app with CORS
# ---------------------------
app = FastAPI(title="TAS Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Detector: C2PA provenance using c2patool (cross‑platform)
# ---------------------------
def check_c2pa(file_path: str) -> dict:
    """
    Uses c2patool (official C2PA tool) to verify provenance.
    Works on Windows and Linux.
    """
    # Find c2patool in PATH or current directory
    c2pa_tool = shutil.which("c2patool")
    if not c2pa_tool:
        # On Windows, also try with .exe
        if os.name == 'nt':
            c2pa_tool = shutil.which("c2patool.exe")
    if not c2pa_tool:
        # Look in current directory as fallback
        local_path = os.path.join(os.path.dirname(__file__), "c2patool")
        if os.name == 'nt':
            local_path += ".exe"
        if os.path.exists(local_path):
            c2pa_tool = local_path
        else:
            return {"present": False, "details": "c2patool not found - please install it"}

    # Ensure the binary is executable (fix for permission denied on Render)
    try:
        # Only try to chmod on Unix-like systems (not Windows)
        if os.name != 'nt':
            os.chmod(c2pa_tool, 0o755)  # rwxr-xr-x permissions
    except Exception as e:
        # If we can't set permissions, log it but continue
        logger.warning(f"Could not set executable permission on {c2pa_tool}: {e}")

    try:
        result = subprocess.run(
            [c2pa_tool, file_path, '--output', '-'],
            capture_output=True, text=True, timeout=10
        )

        if result.returncode == 0 and result.stdout:
            data = json.loads(result.stdout)
            active_manifest = data.get('active_manifest')
            if active_manifest:
                return {
                    "present": True,
                    "details": "C2PA signature found",
                    "manifest": active_manifest
                }
        return {"present": False, "details": "No C2PA data"}
    except subprocess.TimeoutExpired:
        return {"present": False, "details": "c2patool timed out"}
    except json.JSONDecodeError:
        return {"present": False, "details": "Invalid c2patool output"}
    except Exception as e:
        return {"present": False, "details": f"Error: {str(e)}"}

# ---------------------------
# Detector: Forensic heuristics with image resizing
# ---------------------------
def forensic_analysis(image_path: str) -> dict:
    try:
        # Open image with PIL first to check size and resize if needed
        pil_img = Image.open(image_path)
        
        # Define maximum dimension (e.g., 1024 pixels)
        max_dim = 1024
        if max(pil_img.size) > max_dim:
            # Resize while keeping aspect ratio
            pil_img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            # Save the resized image to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_resized:
                pil_img.save(tmp_resized, format='JPEG', quality=85)
                resized_path = tmp_resized.name
        else:
            resized_path = image_path  # use original if already small

        # Now use OpenCV on the (possibly resized) image
        img = cv2.imread(resized_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        noise_score = min(laplacian_var / 200.0, 1.0)

        # Re-open with PIL to get EXIF and dimensions (using original or resized)
        analysis_img = Image.open(resized_path)
        exif = analysis_img.info.get('exif')
        exif_present = exif is not None

        stats = os.stat(resized_path)
        file_size_kb = stats.st_size / 1024.0
        width, height = analysis_img.size
        pixels = width * height
        compression_ratio = file_size_kb / (pixels / 1000.0) if pixels > 0 else 0

        # Clean up temporary file if we created one
        if resized_path != image_path:
            os.unlink(resized_path)

        ai_prob = (
            (1.0 - noise_score) * 0.5 +
            (0.3 if not exif_present else 0.0) +
            (0.2 if compression_ratio < 0.1 else 0.0)
        )
        ai_prob = min(ai_prob, 1.0)

        return {
            "ai_probability": ai_prob,
            "noise_variance": float(laplacian_var),
            "exif_present": exif_present,
            "compression_ratio": float(compression_ratio),
            "details": "Forensic heuristics (resized if needed)"
        }
    except Exception as e:
        return {"ai_probability": 0.5, "details": f"Error: {str(e)}"}

# ---------------------------
# Ensemble
# ---------------------------
def ensemble(forensic: dict, c2pa: dict) -> dict:
    ai_score = 0.5

    if c2pa.get('present', False):
        ai_score -= 0.4   # strong RAW signal
    else:
        ai_score += 0.1   # slight AI bias

    forensic_prob = forensic.get('ai_probability', 0.5)
    ai_score += (forensic_prob - 0.5) * 0.4

    ai_score = max(0.0, min(1.0, ai_score))

    if ai_score < 0.3:
        verdict = "RAW"
        confidence = int((1 - ai_score) * 100)
    elif ai_score > 0.7:
        verdict = "AI"
        confidence = int(ai_score * 100)
    else:
        verdict = "Uncertain"
        confidence = int((1 - abs(ai_score - 0.5) * 2) * 100)

    return {
        "verdict": verdict,
        "confidence": confidence,
        "details": {
            "c2pa": c2pa,
            "forensic": forensic
        }
    }

# ---------------------------
# API Endpoint
# ---------------------------
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "Only image files are supported")

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        c2pa_result = await asyncio.to_thread(check_c2pa, tmp_path)
        forensic_result = await asyncio.to_thread(forensic_analysis, tmp_path)

        final = ensemble(forensic_result, c2pa_result)
        return JSONResponse(content=final)
    finally:
        os.unlink(tmp_path)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))