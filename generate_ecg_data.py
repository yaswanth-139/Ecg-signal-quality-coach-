"""Step 1 of the AI pipeline: generate labeled ECG windows and extract features.

Run:  python generate_ecg_data.py
Output: ecg_training_data.csv  (one row per window: features + 7 issue labels)
"""
import sys
import time

import numpy as np
import pandas as pd
from scipy import signal as sp

from ecg_features import FEATURES, FS, ISSUES, window_features

N_WINDOWS = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
DURATIONS = [4, 5, 6, 8, 10]          # window lengths (seconds) the model learns
rng = np.random.default_rng(7)


# ------------------------------------------------------------ clean ECG
def synth_ecg(dur):
    n = int(dur * FS)
    t = np.arange(n) / FS
    hr = rng.uniform(50, 110)
    r_amp = rng.uniform(0.6, 1.6)
    w = rng.uniform(0.8, 1.3)
    waves = [(-0.20, rng.uniform(0.05, 0.25), 0.025 * w),
             (-0.04, -rng.uniform(0.03, 0.2), 0.010 * w),
             (0.00, 1.0, 0.012 * w),
             (0.04, -rng.uniform(0.1, 0.35), 0.010 * w),
             (rng.uniform(0.22, 0.32), rng.uniform(0.15, 0.5), 0.05 * w)]
    beats, cur = [], rng.uniform(-0.3, 60 / hr)
    while cur < dur + 0.5:
        beats.append(cur)
        cur += 60 / hr + rng.normal(0, rng.uniform(0.01, 0.05))
    sig = np.zeros(n)
    pvc = rng.integers(len(beats)) if rng.random() < 0.06 else -1   # occasional ectopic beat (NOT noise)
    for k, b in enumerate(beats):
        if k == pvc:                       # premature, wide, tall beat with inverted T
            for off, amp, wd in [(0.0, 1.3, 0.035), (0.22, -0.5, 0.07)]:
                sig += r_amp * amp * np.exp(-((t - (b - 0.15) - off) ** 2) / (2 * wd ** 2))
            continue
        jitter = rng.uniform(0.9, 1.1)
        for off, amp, wd in waves:
            sig += r_amp * jitter * amp * np.exp(-((t - b - off) ** 2) / (2 * wd ** 2))
    sig += rng.normal(0, rng.uniform(0.005, 0.03) * r_amp, n)
    if rng.random() < 0.1:                 # inverted lead
        sig = -sig
    return t, sig, r_amp


# ------------------------------------------------------------ artifacts
def add_artifacts(t, sig, r):
    """Randomly inject artifacts. Returns the noisy signal and the label dict."""
    n, dur = len(t), len(t) / FS
    y = dict.fromkeys(ISSUES, 0)

    if rng.random() < 0.22:                                   # baseline wander
        level = rng.uniform(0.05, 1.5)
        if rng.random() < 0.5:
            drift = sum(level * r * rng.uniform(0.4, 1) * np.sin(2 * np.pi * rng.uniform(0.05, 0.5) * t + rng.uniform(0, 6.28))
                        for _ in range(rng.integers(1, 3)))
        else:
            rw = sp.sosfilt(sp.butter(2, 0.4, fs=FS, output="sos"), rng.normal(size=n))
            drift = rw / (rw.std() + 1e-12) * level * r / np.sqrt(2)
        sig = sig + drift
        y["baseline"] = int(level >= 0.30)

    if rng.random() < 0.22:                                   # power-line hum
        level, f0 = rng.uniform(0.02, 0.8), rng.choice([50, 60])
        hum = level * r * np.sin(2 * np.pi * f0 * t + rng.uniform(0, 6.28))
        if rng.random() < 0.3:
            hum += 0.1 * level * r * np.sin(2 * np.pi * 3 * f0 * t)
        sig = sig + hum
        y["powerline"] = int(level >= 0.10)

    if rng.random() < 0.22:                                   # muscle (EMG) noise
        level = rng.uniform(0.02, 0.6)
        noise = rng.normal(0, 1, n)
        if rng.random() < 0.5:
            noise = sp.sosfilt(sp.butter(2, 20, btype="high", fs=FS, output="sos"), noise)
            noise /= noise.std() + 1e-12
        env = np.ones(n)
        if rng.random() < 0.3:                                # only part of the window
            a = rng.integers(0, n // 2)
            env = np.zeros(n)
            env[a:a + int(n * rng.uniform(0.3, 0.7))] = 1
        sig = sig + level * r * noise * env
        y["muscle"] = int(level >= 0.12 and env.mean() >= 0.3)

    if rng.random() < 0.15:                                   # motion artifact
        level = rng.uniform(0.2, 3.0)
        L = int(FS * rng.uniform(0.5, 2.5))
        a = rng.integers(0, max(1, n - L))
        burst = (np.sin(2 * np.pi * rng.uniform(0.7, 6) * np.arange(L) / FS + rng.uniform(0, 6.28))
                 + 0.3 * rng.normal(size=L)) * np.hanning(L)
        sig = sig.copy()
        sig[a:a + L] += level * r * burst
        y["motion"] = int(level >= 0.7)

    if rng.random() < 0.12:                                   # electrode pops
        level = rng.uniform(0.3, 4.0)
        sig = sig.copy()
        for _ in range(rng.integers(1, 3)):
            i = rng.integers(0, n - 30)
            tau = rng.uniform(10, 60) / 1000 * FS
            sign = rng.choice([-1, 1])
            sig[i:i + 30] += sign * level * r * np.exp(-np.arange(30) / tau)
        y["pop"] = int(level >= 1.5)

    if rng.random() < 0.06:                                   # amplifier clipping
        hi = sig.max() * rng.uniform(0.3, 1.0)
        sig = np.minimum(sig, hi)
        at_rail = np.mean(sig >= hi - 1e-12)
        if rng.random() < 0.5:
            lo = sig.min() * rng.uniform(0.3, 1.0)
            sig = np.maximum(sig, lo)
            at_rail = max(at_rail, np.mean(sig <= lo + 1e-12))
        y["clipping"] = int(at_rail >= 0.02)

    if rng.random() < 0.05:                                   # flatline / electrode off
        L = n if rng.random() < 0.4 else int(FS * rng.uniform(0.6, dur))
        a = rng.integers(0, n - L + 1)
        sig = sig.copy()
        sig[a:a + L] = sig[a]
        if L / FS >= 1.0:
            y["flatline"] = 1
            if L > n * 0.5:                                   # window mostly flat: other noise is hidden
                for k in ISSUES:
                    if k != "flatline":
                        y[k] = 0
    return sig, y


# ------------------------------------------------------------ main
if __name__ == "__main__":
    t0 = time.time()
    rows = []
    for i in range(N_WINDOWS):
        dur = float(rng.choice(DURATIONS))
        t, clean, r = synth_ecg(dur)
        sig, y = add_artifacts(t, clean, r)
        rows.append({**dict(zip(FEATURES, window_features(sig, FS))), **y, "duration_s": dur})
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{N_WINDOWS} windows ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv("ecg_training_data.csv", index=False)
    clean_share = (df[ISSUES].sum(axis=1) == 0).mean() * 100
    print(f"\nSaved ecg_training_data.csv: {len(df)} windows, {len(FEATURES)} features")
    print(f"Clean windows: {clean_share:.0f}%")
    print("Windows containing each issue:")
    print(df[ISSUES].sum().to_string())
