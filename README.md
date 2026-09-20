# 🫀 ECG Signal Quality Coach

An educational web app that analyzes ECG recordings and tells you **which parts are safe to analyze — and why the rest are not**. It combines classical signal processing with a trained machine-learning model to detect recording problems like baseline drift, power-line hum, muscle noise, motion artifacts, electrode pops, flatlines, and amplifier clipping.

> ⚠️ **Disclaimer:** This is an educational tool. It checks *signal quality only* and is **not a diagnostic device**.

---

## ✨ Features

- 📤 **Upload any ECG CSV** (works with or without a header row, detects the time column and sampling rate automatically)
- 🟢🟡🔴 **Window-by-window quality scoring** — every 5-second window gets a **Good / Caution / Poor** verdict with a color-coded timeline
- 🤖 **Dual engine:** choose between a trained **Random Forest AI model** (trained on 6,000 labeled synthetic ECG windows) or a **rule-based** scoring engine
- ❤️ **Safe heart-rate & HRV (RMSSD) estimation** — computed from clean windows only, compared against a naive whole-recording analysis to show how noise inflates HRV
- 🔍 **Automatic R-peak detection** (Pan–Tompkins-style) with beats marked as clean or flagged
- 💡 **Practical tips** — turns the most common problems in your recording into concrete recording advice
- 🔌 **Notch filter option** (50/60 Hz) — see how power-line hum affects scoring, and how it disappears with filtering
- ⬇️ **Downloadable CSV report** of every window's metrics
- 📊 **Explainable AI** — view feature importances and per-artifact F1 scores of the trained model

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/ecg-signal-quality-coach.git
cd ecg-signal-quality-coach


How It Works
The AI pipeline
1. 
Generate data —  generate_ecg_data.py  synthesizes thousands of ECG windows with random heart rate, waveform shape, and injected artifacts. Each window is labeled with the artifacts it contains (baseline, powerline, muscle, motion, pop, flatline, clipping).
2. 
Extract features —  ecg_features.py  turns each window into 21 numbers: frequency-band power shares, waveform shape statistics (kurtosis, skew, Hjorth parameters, zero-crossing rate), and beat-regularity measures (R-peak rate, RR variability, template correlation).
3. 
Train —  train_ecg_model.py  fits a multi-output Random Forest (300 trees) to predict the probability of each artifact per window, evaluates per-artifact F1 scores, and saves the model to  ecg_quality_model.joblib .
4. 
Score — each window's score is  100 − (severity-weighted probability of the most likely problem) . Power-line hum counts half because a notch filter can fix it.




Project Structure
├── app.py                # Streamlit web app (UI, plots, report)
├── ecg_quality.py        # Signal processing + rule-based scoring
├── ecg_features.py       # Feature extraction + AI scoring helpers
├── generate_ecg_data.py  # Step 1: build labeled training data
├── train_ecg_model.py    # Step 2: train & evaluate the Random Forest
├── make_demo_data.py     # Generate the 3 demo CSVs
├── requirements.txt
└── ecg_quality_model.joblib   # (generated) trained model




🛠️ Tech Stack
 
Python — numpy, scipy (filters, periodograms, peak detection), pandas
 
scikit-learn — Random Forest multi-label classifier
 
Streamlit + Plotly — interactive web UI and signal visualization
 
joblib — model serialization





🧪 Known Limitations & Future Work
 
The model is trained and tested on synthetic data. Real-world validation (e.g., PhysioNet recordings) is planned future work.
 
The app analyzes one ECG lead at a time.
 
Scores are tuned for resting recordings at typical sampling rates (250–500 Hz).