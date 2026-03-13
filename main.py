# main.py
import os
import re
import subprocess
import json
import asyncio
import tempfile
import shutil
from pathlib import Path
import logging
import filetype
import piexif
from xml.etree import ElementTree as ET

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Image processing
from PIL import Image
import numpy as np
import cv2

# Ensure SIFT is available (requires opencv-contrib-python)
try:
    sift = cv2.SIFT_create()
except AttributeError:
    raise ImportError("OpenCV contrib modules not found. Please install opencv-contrib-python.")

# ---------------------------
# Tunable Ensemble Parameters – normalized sum = 1.0
# ---------------------------
NEUTRAL_SCORE = 0.5
C2PA_PRESENT_RAW_SHIFT = -0.4
C2PA_ABSENT_AI_SHIFT = 0.0

FORENSIC_WEIGHT = 0.04
ELA_WEIGHT = 0.05
FREQ_WEIGHT = 0.01
COLOR_WEIGHT = 0.07
META_WEIGHT = 0.15
COPYMOVE_WEIGHT = 0.05
JPEGGHOST_WEIGHT = 0.05
CFA_WEIGHT = 0.02
LIGHTING_WEIGHT = 0.02
NOISE_INCONSISTENCY_WEIGHT = 0.02
GRRE_WEIGHT = 0.17
COLOR_CORR_WEIGHT = 0.10
SHARPNESS_WEIGHT = 0.10
XMP_WEIGHT = 0.05
EXIF_COMPLETENESS_WEIGHT = 0.05
SENSOR_NOISE_WEIGHT = 0.05

_total = sum([FORENSIC_WEIGHT, ELA_WEIGHT, FREQ_WEIGHT, COLOR_WEIGHT, META_WEIGHT,
              COPYMOVE_WEIGHT, JPEGGHOST_WEIGHT, CFA_WEIGHT, LIGHTING_WEIGHT,
              NOISE_INCONSISTENCY_WEIGHT, GRRE_WEIGHT, COLOR_CORR_WEIGHT,
              SHARPNESS_WEIGHT, XMP_WEIGHT, EXIF_COMPLETENESS_WEIGHT, SENSOR_NOISE_WEIGHT])
assert abs(_total - 1.0) < 0.001, f"Weights must sum to 1.0, got {_total}"

RAW_THRESHOLD = 0.35
AI_THRESHOLD = 0.65

# ---------------------------
# Logging setup
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# FastAPI app with CORS and rate limiting
# ---------------------------
app = FastAPI(title="TAS Detection API")

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Constants
# ---------------------------
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
C2PA_TOOL_PATH = os.environ.get("C2PA_TOOL_PATH", os.path.join(os.path.dirname(__file__), "c2patool"))
if os.name == 'nt':
    C2PA_TOOL_PATH += ".exe"

# ---------------------------
# Helper: validate image file using magic bytes
# ---------------------------
def validate_image(file_bytes: bytes) -> bool:
    kind = filetype.guess(file_bytes)
    if kind is None:
        return False
    return kind.mime.startswith('image/')

# ---------------------------
# Detector: C2PA provenance using c2patool
# ---------------------------
def check_c2pa(file_path: str) -> dict:
    if not os.path.exists(C2PA_TOOL_PATH):
        logger.error(f"c2patool not found at {C2PA_TOOL_PATH}")
        return {"present": False, "details": "c2patool binary missing"}

    temp_dir = tempfile.mkdtemp()
    dest_tool = os.path.join(temp_dir, "c2patool")
    if os.name == 'nt':
        dest_tool += ".exe"

    try:
        shutil.copy2(C2PA_TOOL_PATH, dest_tool)
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
# Detector: Forensic heuristics (noise, EXIF, compression)
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
        score = 0.5
        has_exif = False
        has_camera_info = False
        software = None

        try:
            exif_dict = piexif.load(image_path)
            if exif_dict:
                has_exif = True
                ifd_0 = exif_dict.get('0th', {})
                if ifd_0.get(piexif.ImageIFD.Make) or ifd_0.get(piexif.ImageIFD.Model):
                    has_camera_info = True
                    score -= 0.3
                software = ifd_0.get(piexif.ImageIFD.Software)
                if software and isinstance(software, bytes):
                    software = software.decode('utf-8', errors='ignore').lower()
                    ai_keywords = ['ai', 'generator', 'midjourney', 'dalle', 'stable diffusion']
                    if any(k in software for k in ai_keywords):
                        score += 0.4
        except Exception:
            pass

        if not has_exif:
            score += 0.2
        elif not has_camera_info:
            score -= 0.1

        score = max(0.0, min(1.0, score))

        return {
            "metadata_score": float(score),
            "has_exif": has_exif,
            "has_camera_info": has_camera_info,
            "software": software,
            "details": "Metadata analysis (EXIF + XMP)"
        }
    except Exception as e:
        logger.exception("Metadata analysis failed")
        return {"metadata_score": 0.5, "details": f"Error: {str(e)}"}

def xmp_analysis(image_path: str) -> dict:
    try:
        with open(image_path, 'rb') as f:
            data = f.read()
        xmp_start = data.find(b'<x:xmpmeta')
        if xmp_start == -1:
            return {"xmp_score": 0.0, "details": "No XMP found"}

        chunk = data[xmp_start:xmp_start+65536]
        try:
            chunk_str = chunk.decode('utf-8', errors='ignore')
        except:
            return {"xmp_score": 0.5, "details": "XMP decoding error"}

        score = 0.0
        if 'dc:creator' in chunk_str:
            score += 0.3
        if 'photoshop:Source' in chunk_str:
            score += 0.3
        m = re.search(r'xmp:CreatorTool="([^"]+)"', chunk_str)
        if m:
            tool = m.group(1).lower()
            ai_keywords = ['midjourney', 'dalle', 'stable diffusion', 'firefly']
            if any(k in tool for k in ai_keywords):
                score += 0.4
        score = min(1.0, score)
        return {
            "xmp_score": float(score),
            "details": "XMP metadata analysis"
        }
    except Exception as e:
        logger.exception("XMP analysis failed")
        return {"xmp_score": 0.5, "details": f"Error: {str(e)}"}

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
    try:
        pil_img = Image.open(image_path).convert('RGB')
        max_dim = 1024
        if max(pil_img.size) > max_dim:
            pil_img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

        qualities = [95, 85, 75, 65]
        temp_files = []
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_orig:
            pil_img.save(tmp_orig.name, 'JPEG', quality=100)
            orig_path = tmp_orig.name
            temp_files.append(orig_path)
            orig_img = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)

        max_diff = 0.0
        for q in qualities:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                pil_img.save(tmp.name, 'JPEG', quality=q)
                temp_files.append(tmp.name)
                q_img = cv2.imread(tmp.name, cv2.IMREAD_GRAYSCALE)
            diff = cv2.absdiff(orig_img, q_img)
            diff_std = np.std(diff)
            if diff_std > max_diff:
                max_diff = diff_std

        for f in temp_files:
            try:
                os.unlink(f)
            except Exception as e:
                logger.warning(f"Could not delete temp file {f}: {e}")

        pil_img.close()
        score = min(1.0, max_diff / 30.0)
        return {
            "jpeg_ghost_score": float(score),
            "details": "JPEG ghost analysis (robust)"
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

def grre_analysis(image_path: str) -> dict:
    try:
        img = cv2.imread(image_path)
        b, g, r = cv2.split(img)
        b_zero = b.astype(np.float32)
        r_zero = r.astype(np.float32)
        g_reconstructed = (b_zero + r_zero) / 2
        g_original = g.astype(np.float32)
        mse = np.mean((g_original - g_reconstructed) ** 2)
        score = min(1.0, mse / 5000.0)
        return {
            "grre_score": float(score),
            "mse": float(mse),
            "details": "G-Channel removal reconstruction error"
        }
    except Exception as e:
        logger.exception("GRRE analysis failed")
        return {"grre_score": 0.5, "details": f"Error: {str(e)}"}

def color_correlation_analysis(image_path: str) -> dict:
    try:
        img = cv2.imread(image_path)
        h, w, _ = img.shape
        max_dim = 1024
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = new_h, new_w

        patch_size = 64
        correlations = []
        for y in range(0, h - patch_size, patch_size):
            for x in range(0, w - patch_size, patch_size):
                patch = img[y:y+patch_size, x:x+patch_size]
                b, g, r = cv2.split(patch)
                b = b.flatten().astype(np.float32)
                g = g.flatten().astype(np.float32)
                r = r.flatten().astype(np.float32)
                corr_rg = np.corrcoef(r, g)[0,1] if np.std(r) > 0 and np.std(g) > 0 else 0
                corr_rb = np.corrcoef(r, b)[0,1] if np.std(r) > 0 and np.std(b) > 0 else 0
                corr_gb = np.corrcoef(g, b)[0,1] if np.std(g) > 0 and np.std(b) > 0 else 0
                avg_corr = (corr_rg + corr_rb + corr_gb) / 3.0
                correlations.append(avg_corr)

        if not correlations:
            return {"color_corr_score": 0.5, "details": "Image too small"}

        global_avg = np.mean(correlations)
        ai_prob = 1.0 - (global_avg + 1) / 2
        ai_prob = max(0.0, min(1.0, ai_prob))
        return {
            "color_corr_score": float(ai_prob),
            "avg_corr": float(global_avg),
            "details": "Local color channel correlation"
        }
    except Exception as e:
        logger.exception("Color correlation analysis failed")
        return {"color_corr_score": 0.5, "details": f"Error: {str(e)}"}

def sharpness_analysis(image_path: str) -> dict:
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        max_dim = 1024
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = new_h, new_w

        block_size = 64
        sharpness_vals = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = img[y:y+block_size, x:x+block_size]
                lap = cv2.Laplacian(block, cv2.CV_64F)
                sharpness_vals.append(np.var(lap))

        if len(sharpness_vals) < 2:
            return {"sharpness_score": 0.5, "details": "Image too small"}

        sharpness_std = np.std(sharpness_vals)
        uniformity = min(1.0, sharpness_std / 500.0)
        ai_prob = 1.0 - uniformity
        return {
            "sharpness_score": float(ai_prob),
            "sharpness_std": float(sharpness_std),
            "details": "Sharpness inconsistency (calibrated)"
        }
    except Exception as e:
        logger.exception("Sharpness analysis failed")
        return {"sharpness_score": 0.5, "details": f"Error: {str(e)}"}

# ---------------------------
# NEW: EXIF completeness detector
# ---------------------------
def exif_completeness_analysis(image_path: str) -> dict:
    try:
        exif_dict = piexif.load(image_path)
        total_fields = 0
        for ifd_name, ifd_data in exif_dict.items():
            if isinstance(ifd_data, dict):
                total_fields += len(ifd_data)
        ai_prob = max(0.0, 1.0 - min(1.0, total_fields / 20.0))
        return {
            "exif_completeness_score": float(ai_prob),
            "field_count": total_fields,
            "details": "EXIF completeness"
        }
    except Exception as e:
        logger.exception("EXIF completeness analysis failed")
        return {"exif_completeness_score": 0.5, "details": f"Error: {str(e)}"}

# ---------------------------
# NEW: Sensor noise pattern detector (PRNU-lite) – corrected inversion
# ---------------------------
def sensor_noise_analysis(image_path: str) -> dict:
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        denoised = cv2.GaussianBlur(img, (5, 5), 0)
        residual = img - denoised
        h, w = residual.shape
        q1 = residual[0:h//2, 0:w//2]
        q2 = residual[0:h//2, w//2:w]
        q3 = residual[h//2:h, 0:w//2]
        q4 = residual[h//2:h, w//2:w]
        variances = [np.var(q) for q in [q1, q2, q3, q4]]
        mean_var = np.mean(variances)
        if mean_var < 1e-6:
            return {"sensor_noise_score": 0.5, "details": "Uniform region"}
        consistency = np.std(variances) / mean_var
        # Low consistency (more uniform) likely AI, high consistency (structured) likely real.
        # So we set ai_prob = consistency (low consistency → low ai_prob)
        ai_prob = max(0.0, min(1.0, consistency))
        return {
            "sensor_noise_score": float(ai_prob),
            "consistency": float(consistency),
            "details": "Sensor noise pattern (PRNU-lite)"
        }
    except Exception as e:
        logger.exception("Sensor noise analysis failed")
        return {"sensor_noise_score": 0.5, "details": f"Error: {str(e)}"}

# ---------------------------
# Ensemble (all detectors)
# ---------------------------
def ensemble(forensic: dict, ela: dict, freq: dict, color_ent: dict, meta: dict,
             copymove: dict, jpegghost: dict, cfa: dict, lighting: dict, noise: dict,
             grre: dict, color_corr: dict, sharpness: dict, xmp: dict,
             exif_comp: dict, sensor_noise: dict, c2pa: dict) -> dict:
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

    color_ent_ai = 1 - color_ent.get('color_entropy', 0.5)
    ai_score += (color_ent_ai - NEUTRAL_SCORE) * COLOR_WEIGHT

    ai_score += (meta.get('metadata_score', 0.5) - NEUTRAL_SCORE) * META_WEIGHT

    ai_score += (copymove.get('copy_move_score', 0.5) - NEUTRAL_SCORE) * COPYMOVE_WEIGHT
    ai_score += (jpegghost.get('jpeg_ghost_score', 0.5) - NEUTRAL_SCORE) * JPEGGHOST_WEIGHT
    ai_score += (cfa.get('cfa_score', 0.5) - NEUTRAL_SCORE) * CFA_WEIGHT
    ai_score += (lighting.get('lighting_score', 0.5) - NEUTRAL_SCORE) * LIGHTING_WEIGHT
    ai_score += (noise.get('noise_inconsistency_score', 0.5) - NEUTRAL_SCORE) * NOISE_INCONSISTENCY_WEIGHT
    ai_score += (grre.get('grre_score', 0.5) - NEUTRAL_SCORE) * GRRE_WEIGHT

    ai_score += (color_corr.get('color_corr_score', 0.5) - NEUTRAL_SCORE) * COLOR_CORR_WEIGHT
    ai_score += (sharpness.get('sharpness_score', 0.5) - NEUTRAL_SCORE) * SHARPNESS_WEIGHT
    ai_score += (xmp.get('xmp_score', 0.5) - NEUTRAL_SCORE) * XMP_WEIGHT

    ai_score += (exif_comp.get('exif_completeness_score', 0.5) - NEUTRAL_SCORE) * EXIF_COMPLETENESS_WEIGHT
    ai_score += (sensor_noise.get('sensor_noise_score', 0.5) - NEUTRAL_SCORE) * SENSOR_NOISE_WEIGHT

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
            "color_entropy": color_ent,
            "metadata": meta,
            "copy_move": copymove,
            "jpeg_ghost": jpegghost,
            "cfa": cfa,
            "lighting": lighting,
            "noise_inconsistency": noise,
            "grre": grre,
            "color_correlation": color_corr,
            "sharpness": sharpness,
            "xmp": xmp,
            "exif_completeness": exif_comp,
            "sensor_noise": sensor_noise
        }
    }

# ---------------------------
# Version endpoint
# ---------------------------
@app.get("/version")
async def version():
    return {
        "detectors": [
            "c2pa", "forensic", "ela", "frequency", "color_entropy",
            "metadata", "copy_move", "jpeg_ghost", "cfa", "lighting",
            "noise_inconsistency", "grre", "color_correlation",
            "sharpness", "xmp", "exif_completeness", "sensor_noise"
        ],
        "weights": {
            "forensic": FORENSIC_WEIGHT,
            "ela": ELA_WEIGHT,
            "frequency": FREQ_WEIGHT,
            "color_entropy": COLOR_WEIGHT,
            "metadata": META_WEIGHT,
            "copy_move": COPYMOVE_WEIGHT,
            "jpeg_ghost": JPEGGHOST_WEIGHT,
            "cfa": CFA_WEIGHT,
            "lighting": LIGHTING_WEIGHT,
            "noise_inconsistency": NOISE_INCONSISTENCY_WEIGHT,
            "grre": GRRE_WEIGHT,
            "color_correlation": COLOR_CORR_WEIGHT,
            "sharpness": SHARPNESS_WEIGHT,
            "xmp": XMP_WEIGHT,
            "exif_completeness": EXIF_COMPLETENESS_WEIGHT,
            "sensor_noise": SENSOR_NOISE_WEIGHT
        },
        "thresholds": {
            "raw": RAW_THRESHOLD,
            "ai": AI_THRESHOLD
        },
        "c2pa_shifts": {
            "present": C2PA_PRESENT_RAW_SHIFT,
            "absent": C2PA_ABSENT_AI_SHIFT
        }
    }

# ---------------------------
# API Endpoint
# ---------------------------
@app.post("/detect")
@limiter.limit("10/minute")
async def detect(
    request: Request,
    file: UploadFile = File(...),
    premium: bool = Query(True, description="All detectors are always enabled")
):
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "Only image files are supported")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB.")

    if not validate_image(content):
        raise HTTPException(400, "Uploaded file is not a valid image (magic bytes mismatch)")

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Run all detectors in parallel
        c2pa_task = asyncio.create_task(asyncio.to_thread(check_c2pa, tmp_path))
        forensic_task = asyncio.create_task(asyncio.to_thread(forensic_analysis, tmp_path))
        ela_task = asyncio.create_task(asyncio.to_thread(error_level_analysis, tmp_path))
        freq_task = asyncio.create_task(asyncio.to_thread(frequency_analysis, tmp_path))
        color_ent_task = asyncio.create_task(asyncio.to_thread(color_entropy_analysis, tmp_path))
        meta_task = asyncio.create_task(asyncio.to_thread(metadata_analysis, tmp_path))
        copymove_task = asyncio.create_task(asyncio.to_thread(copy_move_analysis, tmp_path))
        jpegghost_task = asyncio.create_task(asyncio.to_thread(jpeg_ghost_analysis, tmp_path))
        cfa_task = asyncio.create_task(asyncio.to_thread(cfa_analysis, tmp_path))
        lighting_task = asyncio.create_task(asyncio.to_thread(lighting_consistency_analysis, tmp_path))
        noise_task = asyncio.create_task(asyncio.to_thread(noise_inconsistency_analysis, tmp_path))
        grre_task = asyncio.create_task(asyncio.to_thread(grre_analysis, tmp_path))
        color_corr_task = asyncio.create_task(asyncio.to_thread(color_correlation_analysis, tmp_path))
        sharpness_task = asyncio.create_task(asyncio.to_thread(sharpness_analysis, tmp_path))
        xmp_task = asyncio.create_task(asyncio.to_thread(xmp_analysis, tmp_path))
        exif_comp_task = asyncio.create_task(asyncio.to_thread(exif_completeness_analysis, tmp_path))
        sensor_noise_task = asyncio.create_task(asyncio.to_thread(sensor_noise_analysis, tmp_path))

        tasks = [
            c2pa_task, forensic_task, ela_task, freq_task, color_ent_task,
            meta_task, copymove_task, jpegghost_task, cfa_task,
            lighting_task, noise_task, grre_task, color_corr_task,
            sharpness_task, xmp_task, exif_comp_task, sensor_noise_task
        ]

        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=30
        )

        (c2pa_result, forensic_result, ela_result, freq_result, color_ent_result,
         meta_result, copymove_result, jpegghost_result, cfa_result,
         lighting_result, noise_result, grre_result,
         color_corr_result, sharpness_result, xmp_result,
         exif_comp_result, sensor_noise_result) = results

        final = ensemble(
            forensic_result, ela_result, freq_result, color_ent_result, meta_result,
            copymove_result, jpegghost_result, cfa_result, lighting_result, noise_result,
            grre_result, color_corr_result, sharpness_result, xmp_result,
            exif_comp_result, sensor_noise_result, c2pa_result
        )
        return JSONResponse(content=final)
    except asyncio.TimeoutError:
        logger.error("Detection timed out")
        return JSONResponse(status_code=504, content={"error": "Detection timed out"})
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))