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
