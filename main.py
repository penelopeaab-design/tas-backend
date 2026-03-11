# main.py
import os
import subprocess
import json
import asyncio
import tempfile
import shutil
from pathlib import Path
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
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
META_WEIGHT = 0.5

COPYMOVE_WEIGHT = 0.5
JPEGGHOST_WEIGHT = 0.5
CFA_WEIGHT = 0.5
LIGHTING_WEIGHT = 0.5
NOISE_INCONSISTENCY_WEIGHT = 0.5
GRRE_WEIGHT = 0.5          # premium detector weight

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
# Existing detectors (unchanged)
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

def metadata_analysis(image_path: str) -> dict:
    try:
        img = Image.open(image_path)
        exif = img._getexif()
        score = 0.5
        if exif:
            has_camera = (271 in exif) or (272 in exif)
            if has_camera:
                score -= 0.3
            if 305 in exif:
                software = str(exif[305]).lower()
                ai_keywords = ['ai', 'generator', 'midjourney', 'dalle', 'stable diffusion']
                if any(k in software for k in ai_keywords):
                    score += 0.4
            elif not has_camera:
                score -= 0.1
        else:
            score += 0.2
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
# Fixed detectors
# ---------------------------

def copy_move_analysis(image_path: str) -> dict:
    try:
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        max_dim = 1024
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        if des is None or len(kp) < 10:
            return {"copy_move_score": 0.0, "details": "Insufficient features"}
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(des, des, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                if m.queryIdx != m.trainIdx:
                    good.append(m)
        if len(good) < 5:
            return {"copy_move_score": 0.0, "details": "No copy-move detected"}
        area = gray.shape[0] * gray.shape[1]
        raw_score = len(good) / (area * 1e-5)
        score = min(1.0, raw_score)
        return {
            "copy_move_score": float(score),
            "matches_found": len(good),
            "details": "Copy-move detection (SIFT, resized)"
        }
    except Exception as e:
        logger.exception("Copy-move analysis failed")
        return {"copy_move_score": 0.5, "details": f"Error: {str(e)}"}

def jpeg_ghost_analysis(image_path: str) -> dict:
    """
    Detects multiple JPEG compressions by analyzing differences at various qualities.
    Properly cleans up temp files.
    """
    try:
        pil_img = Image.open(image_path).convert('RGB')
        max_dim = 1024
        if max(pil_img.size) > max_dim:
            pil_img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

        qualities = [95, 85, 75, 65]
        diff_std = []
        prev_img = None
        temp_files = []

        for q in qualities:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                pil_img.save(tmp.name, 'JPEG', quality=q)
                temp_files.append(tmp.name)
                tmp_img = cv2.imread(tmp.name, cv2.IMREAD_GRAYSCALE)

            if prev_img is not None:
                diff = cv2.absdiff(prev_img, tmp_img)
                diff_std.append(np.std(diff))
            prev_img = tmp_img

        for f in temp_files:
            try:
                os.unlink(f)
            except Exception as e:
                logger.warning(f"Could not delete temp file {f}: {e}")

        pil_img.close()

        if len(diff_std) < 2:
            return {"jpeg_ghost_score": 0.5, "details": "Insufficient data"}

        ghost_strength = np.std(diff_std) / 10.0
        score = min(1.0, ghost_strength)
        return {
            "jpeg_ghost_score": float(score),
            "diff_std": diff_std,
            "details": "JPEG ghost analysis"
        }
    except Exception as e:
        logger.exception("JPEG ghost analysis failed")
        return {"jpeg_ghost_score": 0.5, "details": f"Error: {str(e)}"}

def cfa_analysis(image_path: str) -> dict:
    try:
        img = cv2.imread(image_path)
        b, g, r = cv2.split(img)
        g_blur = cv2.GaussianBlur(g, (5,5), 0)
        g_diff = cv2.absdiff(g, g_blur)
        cfa_error = np.mean(g_diff) / 255.0
        score = 1.0 - min(1.0, cfa_error * 5)
        return {
            "cfa_score": float(score),
            "details": "CFA analysis"
        }
    except Exception as e:
        logger.exception("CFA analysis failed")
        return {"cfa_score": 0.5, "details": f"Error: {str(e)}"}

def lighting_consistency_analysis(image_path: str) -> dict:
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        quadrants = [
            gray[0:h//2, 0:w//2],
            gray[0:h//2, w//2:w],
            gray[h//2:h, 0:w//2],
            gray[h//2:h, w//2:w]
        ]
        means = [np.mean(q) for q in quadrants]
        lighting_var = np.var(means) / 255.0
        score = min(1.0, lighting_var * 5)
        return {
            "lighting_score": float(score),
            "quadrant_means": means,
            "details": "Lighting consistency analysis"
        }
    except Exception as e:
        logger.exception("Lighting analysis failed")
        return {"lighting_score": 0.5, "details": f"Error: {str(e)}"}

def noise_inconsistency_analysis(image_path: str) -> dict:
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        block_size = 32
        h, w = img.shape
        noise_map = []
        for y in range(0, h-block_size, block_size):
            for x in range(0, w-block_size, block_size):
                block = img[y:y+block_size, x:x+block_size]
                lap = cv2.Laplacian(block, cv2.CV_64F)
                noise_map.append(np.var(lap))
        if not noise_map:
            return {"noise_inconsistency_score": 0.5, "details": "Image too small"}
        noise_std = np.std(noise_map)
        score = min(1.0, noise_std / 20.0)
        return {
            "noise_inconsistency_score": float(score),
            "noise_std": float(noise_std),
            "details": "Noise inconsistency mapping"
        }
    except Exception as e:
        logger.exception("Noise inconsistency analysis failed")
        return {"noise_inconsistency_score": 0.5, "details": f"Error: {str(e)}"}

# ---------------------------
# NEW Premium Detector: GRRE
# ---------------------------
def grre_analysis(image_path: str) -> dict:
    """
    G-Channel Removal Reconstruction Error.
    Removes green channel, reconstructs it, and measures error.
    Higher score = more likely AI.
    """
    try:
        img = cv2.imread(image_path)
        b, g, r = cv2.split(img)
        # Remove green channel (set to zero)
        b_zero = b.astype(np.float32)
        r_zero = r.astype(np.float32)
        # Simple interpolation: average of blue and red
        g_reconstructed = (b_zero + r_zero) / 2
        g_original = g.astype(np.float32)
        # Compute error metrics
        mse = np.mean((g_original - g_reconstructed) ** 2)
        # Normalise (typical MSE range may be 0-10000, adjust)
        score = min(1.0, mse / 5000.0)
        return {
            "grre_score": float(score),
            "mse": float(mse),
            "details": "G-Channel removal reconstruction error"
        }
    except Exception as e:
        logger.exception("GRRE analysis failed")
        return {"grre_score": 0.5, "details": f"Error: {str(e)}"}

# ---------------------------
# Ensemble (all detectors)
# ---------------------------
def ensemble(forensic: dict, ela: dict, freq: dict, color: dict, meta: dict,
             copymove: dict, jpegghost: dict, cfa: dict, lighting: dict, noise: dict,
             grre: dict, c2pa: dict) -> dict:
    ai_score = NEUTRAL_SCORE

    if c2pa.get('present', False):
        ai_score += C2PA_PRESENT_RAW_SHIFT
    else:
        ai_score += C2PA_ABSENT_AI_SHIFT

    ai_score += (forensic.get('ai_probability', 0.5) - NEUTRAL_SCORE) * FORENSIC_WEIGHT

    ela_ai = 1 - ela.get('ela_score', 0.5)
    ai_score += (ela_ai - NEUTRAL_SCORE) * ELA_WEIGHT

    freq_ai = 1 - freq.get('fft_score', 0.5)
    ai_score += (freq_ai - NEUTRAL_SCORE) * FREQ_WEIGHT

    color_ai = 1 - color.get('color_entropy', 0.5)
    ai_score += (color_ai - NEUTRAL_SCORE) * COLOR_WEIGHT

    ai_score += (meta.get('metadata_score', 0.5) - NEUTRAL_SCORE) * META_WEIGHT

    ai_score += (copymove.get('copy_move_score', 0.5) - NEUTRAL_SCORE) * COPYMOVE_WEIGHT
    ai_score += (jpegghost.get('jpeg_ghost_score', 0.5) - NEUTRAL_SCORE) * JPEGGHOST_WEIGHT
    ai_score += (cfa.get('cfa_score', 0.5) - NEUTRAL_SCORE) * CFA_WEIGHT
    ai_score += (lighting.get('lighting_score', 0.5) - NEUTRAL_SCORE) * LIGHTING_WEIGHT
    ai_score += (noise.get('noise_inconsistency_score', 0.5) - NEUTRAL_SCORE) * NOISE_INCONSISTENCY_WEIGHT
    ai_score += (grre.get('grre_score', 0.5) - NEUTRAL_SCORE) * GRRE_WEIGHT

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
            "metadata": meta,
            "copy_move": copymove,
            "jpeg_ghost": jpegghost,
            "cfa": cfa,
            "lighting": lighting,
            "noise_inconsistency": noise,
            "grre": grre
        }
    }

# ---------------------------
# API Endpoint with timeout and premium flag
# ---------------------------
@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    premium: bool = Query(False, description="Enable premium detectors")
):
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "Only image files are supported")

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Create tasks for all detectors
        c2pa_task = asyncio.create_task(asyncio.to_thread(check_c2pa, tmp_path))
        forensic_task = asyncio.create_task(asyncio.to_thread(forensic_analysis, tmp_path))
        ela_task = asyncio.create_task(asyncio.to_thread(error_level_analysis, tmp_path))
        freq_task = asyncio.create_task(asyncio.to_thread(frequency_analysis, tmp_path))
        color_task = asyncio.create_task(asyncio.to_thread(color_entropy_analysis, tmp_path))
        meta_task = asyncio.create_task(asyncio.to_thread(metadata_analysis, tmp_path))
        copymove_task = asyncio.create_task(asyncio.to_thread(copy_move_analysis, tmp_path))
        jpegghost_task = asyncio.create_task(asyncio.to_thread(jpeg_ghost_analysis, tmp_path))
        cfa_task = asyncio.create_task(asyncio.to_thread(cfa_analysis, tmp_path))
        lighting_task = asyncio.create_task(asyncio.to_thread(lighting_consistency_analysis, tmp_path))
        noise_task = asyncio.create_task(asyncio.to_thread(noise_inconsistency_analysis, tmp_path))
        
        tasks = [
            c2pa_task, forensic_task, ela_task, freq_task, color_task,
            meta_task, copymove_task, jpegghost_task, cfa_task,
            lighting_task, noise_task
        ]

        # Conditionally add premium detector
        if premium:
            grre_task = asyncio.create_task(asyncio.to_thread(grre_analysis, tmp_path))
            tasks.append(grre_task)
        else:
            # Provide a placeholder result
            grre_result = {"grre_score": 0.5, "details": "Premium feature disabled"}

        # Wait for all tasks with a timeout (30 seconds)
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=30
        )

        # Unpack results
        (c2pa_result, forensic_result, ela_result, freq_result, color_result,
         meta_result, copymove_result, jpegghost_result, cfa_result,
         lighting_result, noise_result) = results[:11]

        if premium:
            grre_result = results[11]  # the last task result
        # else grre_result already set

        final = ensemble(
            forensic_result, ela_result, freq_result, color_result, meta_result,
            copymove_result, jpegghost_result, cfa_result, lighting_result, noise_result,
            grre_result, c2pa_result
        )
        return JSONResponse(content=final)
    except asyncio.TimeoutError:
        logger.error("Detection timed out")
        return JSONResponse(status_code=504, content={"error": "Detection timed out"})
    finally:
        os.unlink(tmp_path)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))