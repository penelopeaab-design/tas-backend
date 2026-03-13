# calibrate.py
import requests
import os
import json
from pathlib import Path

API_URL = "http://localhost:8000/detect"          # local or live URL
AUTHENTIC_DIR = "calibration/authentic"            # folder with authentic images
AI_DIR = "calibration/ai_generated"                # folder with AI-generated images

def analyse_folder(folder, expected_verdict, label):
    files = list(Path(folder).glob("*.jpg")) + list(Path(folder).glob("*.png")) + list(Path(folder).glob("*.jpeg"))
    
    if not files:
        print(f"No images found in {folder}")
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
            
            # Flag wrong verdicts for inspection
            if verdict != expected_verdict and verdict != "Uncertain":
                print(f"  WRONG: {img_path.name} → {verdict} ({confidence}%)")
        else:
            print(f"  ERROR: {img_path.name} → HTTP {r.status_code}")
    
    total = len(files)
    print(f"\n{label} ({total} images, expected: {expected_verdict})")
    print(f"  Correct:   {results[expected_verdict]}/{total} "
          f"({results[expected_verdict]/total*100:.1f}%)")
    print(f"  Uncertain: {results['Uncertain']}/{total} "
          f"({results['Uncertain']/total*100:.1f}%)")
    wrong_key = "AI" if expected_verdict == "RAW" else "RAW"
    print(f"  Wrong:     {results[wrong_key]}/{total} "
          f"({results[wrong_key]/total*100:.1f}%)")
    if scores:
        print(f"  Avg confidence: {sum(scores)/len(scores):.1f}%")

if __name__ == "__main__":
    print("TAS Calibration Run")
    print("=" * 40)
    analyse_folder(AUTHENTIC_DIR, "RAW", "Authentic Photos")
    analyse_folder(AI_DIR, "AI", "AI-Generated Images")