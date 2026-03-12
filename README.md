\# TAS - The All Seeing



\*\*TAS\*\* is a powerful, open‑source image authenticity detection API. It uses \*\*15 independent forensic detectors\*\* to analyse images and classify them as `RAW` (authentic), `AI` (AI‑generated), or `Uncertain`. No machine learning models are required – all detectors are based on explainable, deterministic algorithms.



🔗 \*\*Live API\*\*: \[https://tas-backend-dx2f.onrender.com](https://tas-backend-dx2f.onrender.com)  

📚 \*\*Interactive docs\*\*: \[https://tas-backend-dx2f.onrender.com/docs](https://tas-backend-dx2f.onrender.com/docs)



---



\## ⚠️ Important Disclaimer



TAS is a heuristic tool. \*\*No detector achieves 100% accuracy\*\*. Results should be treated as indicators, not definitive proof. \*\*Do not use TAS verdicts as sole evidence in legal, editorial, or consequential decisions.\*\*



The system is designed to be transparent and auditable – every detector's score is returned in the response. The ensemble weights are exposed via the `/version` endpoint and can be adjusted in `main.py`.



---



\## ✨ Features



\- \*\*15 detectors\*\* covering:

&nbsp; - Cryptographic provenance (C2PA)

&nbsp; - Forensic heuristics (noise, EXIF, compression)

&nbsp; - Error Level Analysis (ELA)

&nbsp; - Frequency analysis (FFT)

&nbsp; - Colour histogram entropy

&nbsp; - Metadata analysis (EXIF with `piexif`)

&nbsp; - XMP metadata scan

&nbsp; - Copy‑move detection (SIFT)

&nbsp; - JPEG ghost detection (robust version)

&nbsp; - Color Filter Array (CFA) analysis

&nbsp; - Lighting consistency

&nbsp; - Noise inconsistency mapping

&nbsp; - G‑Channel Removal Reconstruction Error (GRRE)

&nbsp; - Local color channel correlation

&nbsp; - Sharpness inconsistency (calibrated)

\- \*\*Parallel execution\*\* – all detectors run concurrently for speed.

\- \*\*Explainable results\*\* – each detector returns a score and details.

\- \*\*Tunable ensemble\*\* – weights and thresholds are exposed via `/version` and can be adjusted by editing `main.py`.

\- \*\*CORS enabled\*\* – ready for frontend integration.

\- \*\*Rate limited\*\* – protects against abuse (10 requests per minute per IP).

\- \*\*Deployed on Render\*\* – free tier, auto‑spins down with inactivity.



---



\## 🚀 Quick Start



\### Prerequisites



\- Python 3.14 or later

\- \[Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (required for compiling `numpy` on Windows)

\- `c2patool` binary (for C2PA detection)



\### 1. Install `c2patool`



Download the latest `c2patool` from the \[official releases page](https://github.com/contentauth/c2pa-rs/releases?q=c2patool) and place the executable in the project root directory (next to `main.py`).  

On Windows, ensure the file is named `c2patool.exe`. On Linux/macOS, name it `c2patool` and make it executable (`chmod +x c2patool`).



\### 2. Clone the repository



```bash

git clone https://github.com/yourusername/tas-backend.git

cd tas-backend

