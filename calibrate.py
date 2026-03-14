import requests
import os
import json
from pathlib import Path

API_URL = "http://localhost:8000/detect"          # local or live URL

# Define the three folders and their expected verdicts
FOLDERS = [
    ("calibration/authentic",    "RAW", "Authentic (Unsplash)"),
    ("calibration/camera_roll",  "RAW", "Authentic (Camera Roll)"),
    ("calibration/ai_generated", "AI",  "AI-Generated")
]

def analyse_folder(folder, expected_verdict, label):
    if not os.path.exists(folder):
        print(f"\nFolder not found: {folder} – skipping.")
        return

    files = list(Path(folder).glob("*.jpg")) + list(Path(folder).glob("*.png")) + list(Path(folder).glob("*.jpeg"))
    if not files:
        print(f"\nNo images found in {folder}")
        return

    results = {"RAW": 0, "AI": 0, "Uncertain": 0}
    scores = []

    for img_path in files:
        with open(img_path, "rb") as f:
            r = requests.post(
                API_URL,
                files={"file": (img_path.name, f, "image/jpeg" if img_path.suffix.lower() in ['.jpg','.jpeg'] else "image/png")}
            )
        if r.status_code == 200:
            data = r.json()
            verdict = data["verdict"]
            confidence = data["confidence"]
            results[verdict] += 1
            scores.append(data.get("confidence", 0))

            if verdict != expected_verdict:
                print(f"\n>>> {img_path.name} → {verdict} ({confidence}%)")
                details = data.get('details', {})
                detector_map = {
                    'forensic': ('ai_probability', False),
                    'ela': ('ela_score', True),
                    'frequency': ('fft_score', True),
                    'color_entropy': ('color_entropy', True),
                    'metadata': ('metadata_score', False),
                    'copy_move': ('copy_move_score', False),
                    'jpeg_ghost': ('jpeg_ghost_score', False),
                    'cfa': ('cfa_score', False),
                    'lighting': ('lighting_score', False),
                    'noise_inconsistency': ('noise_inconsistency_score', False),
                    'grre': ('grre_score', False),
                    'color_correlation': ('color_corr_score', False),
                    'sharpness': ('sharpness_score', False),
                    'xmp': ('xmp_score', False),
                    'exif_completeness': ('exif_completeness_score', False),
                    'sensor_noise': ('sensor_noise_score', False),
                    'c2pa': ('present', False),
                }
                for detector, (key, invert) in detector_map.items():
                    det_data = details.get(detector, {})
                    if detector == 'c2pa':
                        raw = 1.0 if det_data.get('present', False) else 0.0
                    else:
                        raw = det_data.get(key, 0.5)
                    ai_signal = (1 - raw) if invert else raw
                    if ai_signal < 0.4:
                        direction = "→ RAW"
                    elif ai_signal > 0.6:
                        direction = "→ AI"
                    else:
                        direction = "≈ neutral"
                    print(f"    {detector:25}: {ai_signal:.2f} {direction}")
        else:
            print(f"  ERROR: {img_path.name} → HTTP {r.status_code}")

    total = len(files)
    print(f"\n{label} ({total} images, expected: {expected_verdict})")
    print(f"  Correct:   {results[expected_verdict]}/{total} "
          f"({results[expected_verdict] / total * 100:.1f}%)")
    print(f"  Uncertain: {results['Uncertain']}/{total} "
          f"({results['Uncertain'] / total * 100:.1f}%)")
    wrong_key = "AI" if expected_verdict == "RAW" else "RAW"
    print(f"  Wrong:     {results[wrong_key]}/{total} "
          f"({results[wrong_key] / total * 100:.1f}%)")
    if scores:
        print(f"  Avg confidence: {sum(scores)/len(scores):.1f}%")

if __name__ == "__main__":
    print("TAS Calibration Run (All Folders)")
    print("=" * 50)
    for folder, expected, label in FOLDERS:
        analyse_folder(folder, expected, label)
        print("-" * 40)