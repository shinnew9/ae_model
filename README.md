# AutoEncoder_model
**DCASE 2024 Baseline Autoencoder Fine-Tuning for Real-World Anomalous Sound Detection (KTX Depot Data)**

> Fine-tuned a DCASE 2024 baseline autoencoder model for anomalous sound detection using real-world audio collected from KTX maintenance environments (e.g., traction motor-related equipment).

---

## TL;DR
- **Task:** Unsupervised anomalous sound detection (ASD)
- **Model:** DCASE-style **baseline autoencoder** fine-tuning
- **Data:** Real-world machine/maintenance audio (windowed segments in `windowed_data/`)
- **Output:** Anomaly scores based on reconstruction error + evaluation logs/results

---

## Background
Anomalous sound detection is commonly framed as an **unsupervised / domain-generalization** problem in industrial monitoring.
This repository follows the general DCASE challenge direction of ASD baselines and adapts them to real recorded machine sounds. :contentReference[oaicite:1]{index=1}

---

## What’s Inside
- `windowed_data/` — windowed audio segments derived from real-world recordings
- `windowing.py` / `window+ing.ipynb` — audio windowing pipeline
- `autoencoder.py` — baseline AE architecture
- `anomaly_detector.py` — inference / anomaly scoring logic
- `models/`, `logs/`, `results/` — checkpoints, run logs, and outputs

---

## Approach
1. **Preprocess audio**
   - Convert raw recordings into fixed-length windows (frames)
   - Normalize/Standardize features for stable training

2. **Fine-tune baseline AE**
   - Train the autoencoder to reconstruct normal patterns
   - Use **reconstruction error** as anomaly score

3. **Inference & scoring**
   - Compute anomaly score per window
   - Aggregate / analyze results (logs + saved outputs)

---

## 🧩 Architecture Diagram

      flowchart LR
        A[Raw machine audio<br/>(KTX depot recordings)] --> B[Windowing pipeline<br/>(fixed-length segments)]
        B --> C[Feature processing<br/>(normalization)]
        C --> D[Baseline Autoencoder<br/>(fine-tuning)]
        D --> E[Reconstruction error]
        E --> F[Anomaly score<br/>(window-level)]
        F --> G[Logs / Results<br/>(thresholding & analysis)]

### Key Points
- Separates data windowing from model training and anomaly scoring
- Fine-tunes a lightweight baseline AE to adapt to real-world audio distributions

## Notes on Data
This repository includes real-world audio-derived windows under windowed_data/.
If you plan to publish or redistribute the dataset, confirm internal/company policy and data usage constraints.

# How to Run (Example)
pip install -r requirements.txt
python main.py

Results (Optional)

Add a small table or bullet summary (e.g., example anomaly score distributions, notable failure cases)

If you have DCASE-style metrics or AUC/pAUC, place them here

Future Work

Explore domain-shift robustness (different machines/conditions)

Add frequency-domain augmentations and calibration

Compare AE baseline vs stronger models (e.g., CNN-based embedding + density estimation)

-->
