# main.py
import os
import subprocess
import json
import asyncio
import tempfile
import shutil
from pathlib import Path
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Image processing
from PIL import Image
import numpy as np
import cv2

# ---------------------------
# Tunable Ensemble Parameters
# ---------------------------
NEUTRAL_SCORE = 0.5
C2PA_PRESENT_RAW_SHIFT = -0.4
C2PA_ABSENT_AI_SHIFT = 0.1
FORENSIC_WEIGHT = 0.5
ELA_WEIGHT = 1.0
FREQ_WEIGHT = 0.5
COLOR_WEIGHT = 0.2
META_WEIGHT = 0.5           # new weight for metadata analysis
RAW_THRESHOLD = 0.3
AI_THRESHOLD = 0.7

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
# Detector: C2PA provenance using c2patool (copied to temp dir)
# ---------------------------
def check_c2pa(file_path: str) -> dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_tool = os.path.join(script_dir, "c2patool")
    if os.name == 'nt':
        src_tool += ".exe"

    if not os.path.exists(src_tool):
        logger.error(f"c2patool not found at {src_tool}")
        return {"present": False, "details": "c2patool binary missing"}

    temp_dir = tempfile.mkdtemp()
    dest_tool = os.path.join(temp_dir, "c2patool")
    if os.name == 'nt':
        dest_tool += ".exe"

    try:
        shutil.copy2(src_tool, dest_tool)
        os.chmod(dest_tool, 0o755)
        logger.info(f"Copied c2patool to {dest_tool} and set permissions")

        result = subprocess.run(
            [dest_tool, file_path, '--output', '-'],
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
        logger.exception("C2PA check failed")
        return {"present": False, "details": f"Error: {str(e)}"}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# ---------------------------
# Detector: Forensic heuristics (noise, EXIF, compression) with resizing
# ---------------------------
def forensic_analysis(image_path: str) -> dict:
    try:
        pil_img = Image.open(image_path)
        max_dim = 1024
        if max(pil_img.size) > max_dim:
            pil_img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_resized:
                pil_img.save(tmp_resized, format='JPEG', quality=85)
                resized_path = tmp_resized.name
            pil_img.close()
        else:
            resized_path = image_path

        img = cv2.imread(resized_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        noise_score = min(laplacian_var / 200.0, 1.0)

        analysis_img = Image.open(resized_path)
        exif = analysis_img.info.get('exif')
        exif_present = exif is not None

        stats = os.stat(resized_path)
        file_size_kb = stats.st_size / 1024.0
        width, height = analysis_img.size
        pixels = width * height
        compression_ratio = file_size_kb / (pixels / 1000.0) if pixels > 0 else 0

        analysis_img.close()

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
            "details": "Forensic heuristics"
        }
    except Exception as e:
        logger.exception("Forensic analysis failed")
        return {"ai_probability": 0.5, "details": f"Error: {str(e)}"}

# ---------------------------
# Detector: Error Level Analysis (ELA) with resizing and error handling
# ---------------------------
def error_level_analysis(image_path: str) -> dict:
    try:
        pil_img = Image.open(image_path).convert('RGB')
        max_dim = 1024
        if max(pil_img.size) > max_dim:
            pil_img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_resized:
                pil_img.save(tmp_resized, format='JPEG', quality=85)
                resized_path = tmp_resized.name
        else:
            resized_path = image_path

        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_jpeg:
            pil_img.save(tmp_jpeg.name, 'JPEG', quality=90)
            temp_jpeg = tmp_jpeg.name

        resaved = Image.open(temp_jpeg).convert('RGB')
        original_np = np.array(pil_img, dtype=np.int16)
        resaved_np = np.array(resaved, dtype=np.int16)
        diff = np.abs(original_np - resaved_np)
        diff_gray = np.mean(diff, axis=2)

        if diff_gray.max() > 0:
            diff_gray = (diff_gray * 255 / diff_gray.max()).astype(np.uint8)
        else:
            diff_gray = diff_gray.astype(np.uint8)

        mean_diff = np.mean(diff_gray)
        std_diff = np.std(diff_gray)
        score = min(1.0, (mean_diff / 50.0) + (std_diff / 50.0))
        score = max(0.0, min(1.0, score))

        os.unlink(temp_jpeg)
        if resized_path != image_path:
            os.unlink(resized_path)

        pil_img.close()
        resaved.close()

        return {
            "ela_score": float(score),
            "details": "Error Level Analysis (resized if needed)"
        }
    except Exception as e:
        logger.exception("ELA failed for this image")
        return {"ela_score": 0.5, "details": f"Error: {str(e)}"}

# ---------------------------
# Detector: Frequency Analysis using FFT
# ---------------------------
def frequency_analysis(image_path: str) -> dict:
    try:
        pil_img = Image.open(image_path).convert('L')
        max_dim = 1024
        if max(pil_img.size) > max_dim:
            pil_img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        img = np.array(pil_img, dtype=np.float32)
        pil_img.close()

        fft = np.fft.fft2(img)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)

        rows, cols = img.shape
        crow, ccol = rows // 2, cols // 2
        radius = int(min(rows, cols) * 0.1)
        mask = np.ones((rows, cols), dtype=bool)
        mask[crow-radius:crow+radius, ccol-radius:ccol+radius] = False

        total_energy = np.sum(magnitude**2)
        high_energy = np.sum(magnitude[mask]**2)
        ratio = high_energy / total_energy if total_energy > 0 else 0.5
        score = min(1.0, ratio * 2.0)

        return {
            "fft_score": float(score),
            "details": "Frequency analysis (FFT)"
        }
    except Exception as e:
        logger.exception("Frequency analysis failed")
        return {"fft_score": 0.5, "details": f"Error: {str(e)}"}

# ---------------------------
# Detector: Colour Histogram Entropy
# ---------------------------
def color_entropy_analysis(image_path: str) -> dict:
    try:
        pil_img = Image.open(image_path).convert('RGB')
        max_dim = 1024
        if max(pil_img.size) > max_dim:
            pil_img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        img = np.array(pil_img)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()
        sat_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256]).flatten()
        hue_hist = hue_hist / (hue_hist.sum() + 1e-8)
        sat_hist = sat_hist / (sat_hist.sum() + 1e-8)
        hue_entropy = -np.sum([p * np.log2(p + 1e-12) for p in hue_hist if p > 0])
        sat_entropy = -np.sum([p * np.log2(p + 1e-12) for p in sat_hist if p > 0])
        hue_score = min(1.0, hue_entropy / 8.0)
        sat_score = min(1.0, sat_entropy / 8.0)
        color_score = (hue_score + sat_score) / 2
        return {
            "color_entropy": float(color_score),
            "hue_entropy": float(hue_entropy),
            "sat_entropy": float(sat_entropy),
            "details": "Colour histogram entropy"
        }
    except Exception as e:
        logger.exception("Colour entropy analysis failed")
        return {"color_entropy": 0.5, "details": f"Error: {str(e)}"}

# ---------------------------
# NEW Detector: Metadata Analysis (EXIF, etc.)
# ---------------------------
def metadata_analysis(image_path: str) -> dict:
    """
    Examines metadata (EXIF, software tags) to estimate AI probability.
    Returns a score 0-1 where higher = more likely AI.
    """
    try:
        img = Image.open(image_path)
        # Get EXIF data (if any)
        exif = img._getexif()
        info = img.info

        score = 0.5  # neutral

        if exif:
            # Check for camera make/model (tags 271 = Make, 272 = Model)
            has_camera = (271 in exif) or (272 in exif)
            if has_camera:
                # Strong real signal
                score -= 0.3
            # Check for software tag (305 = Software)
            if 305 in exif:
                software = str(exif[305]).lower()
                # If software suggests AI generation, increase score
                ai_keywords = ['ai', 'generator', 'midjourney', 'dalle', 'stable diffusion']
                if any(k in software for k in ai_keywords):
                    score += 0.4
                # Editing software is neutral
            # If any EXIF but no camera info, slight real signal
            elif not has_camera:
                score -= 0.1
        else:
            # No EXIF at all – moderate AI signal
            score += 0.2

        # Clamp
        score = max(0.0, min(1.0, score))

        return {
            "metadata_score": float(score),
            "has_exif": exif is not None,
            "has_camera_info": (exif and (271 in exif or 272 in exif)) if exif else False,
            "details": "Metadata analysis"
        }
    except Exception as e:
        logger.exception("Metadata analysis failed")
        return {"metadata_score": 0.5, "details": f"Error: {str(e)}"}

# ---------------------------
# Ensemble (all detectors)
# ---------------------------
def ensemble(forensic: dict, ela: dict, freq: dict, color: dict, meta: dict, c2pa: dict) -> dict:
    ai_score = NEUTRAL_SCORE

    if c2pa.get('present', False):
        ai_score += C2PA_PRESENT_RAW_SHIFT
    else:
        ai_score += C2PA_ABSENT_AI_SHIFT

    ai_score += (forensic.get('ai_probability', 0.5) - NEUTRAL_SCORE) * FORENSIC_WEIGHT

    # Invert ELA
    ela_ai = 1 - ela.get('ela_score', 0.5)
    ai_score += (ela_ai - NEUTRAL_SCORE) * ELA_WEIGHT

    # Invert frequency
    freq_ai = 1 - freq.get('fft_score', 0.5)
    ai_score += (freq_ai - NEUTRAL_SCORE) * FREQ_WEIGHT

    # Invert colour entropy
    color_ai = 1 - color.get('color_entropy', 0.5)
    ai_score += (color_ai - NEUTRAL_SCORE) * COLOR_WEIGHT

    # Metadata score (already an AI probability)
    meta_score = meta.get('metadata_score', 0.5)
    ai_score += (meta_score - NEUTRAL_SCORE) * META_WEIGHT

    ai_score = max(0.0, min(1.0, ai_score))

    if ai_score < RAW_THRESHOLD:
        verdict = "RAW"
        confidence = int((1 - ai_score) * 100)
    elif ai_score > AI_THRESHOLD:
        verdict = "AI"
        confidence = int(ai_score * 100)
    else:
        verdict = "Uncertain"
        dist_to_raw = ai_score - RAW_THRESHOLD
        dist_to_ai = AI_THRESHOLD - ai_score
        confidence = int((1 - min(dist_to_raw, dist_to_ai) / (AI_THRESHOLD - RAW_THRESHOLD)) * 100)
        confidence = max(0, min(100, confidence))

    return {
        "verdict": verdict,
        "confidence": confidence,
        "details": {
            "c2pa": c2pa,
            "forensic": forensic,
            "ela": ela,
            "frequency": freq,
            "color_entropy": color,
            "metadata": meta
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
        # Run all detectors in parallel
        c2pa_result = await asyncio.to_thread(check_c2pa, tmp_path)
        forensic_result = await asyncio.to_thread(forensic_analysis, tmp_path)
        ela_result = await asyncio.to_thread(error_level_analysis, tmp_path)
        freq_result = await asyncio.to_thread(frequency_analysis, tmp_path)
        color_result = await asyncio.to_thread(color_entropy_analysis, tmp_path)
        meta_result = await asyncio.to_thread(metadata_analysis, tmp_path)

        final = ensemble(forensic_result, ela_result, freq_result, color_result, meta_result, c2pa_result)
        return JSONResponse(content=final)
    finally:
        os.unlink(tmp_path)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))