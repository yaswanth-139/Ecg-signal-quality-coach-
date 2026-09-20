"""Generate 3 demo ECG CSVs (clean, noisy, very_noisy). Needs only numpy + pandas."""
import numpy as np
import pandas as pd

FS = 250          # sampling rate (Hz)
DURATION = 60     # seconds
rng = np.random.default_rng(42)


def clean_ecg(fs=FS, duration=DURATION, hr=72):
    n = int(fs * duration)
    t = np.arange(n) / fs
    sig = np.zeros(n)
    # beat times with small heart-rate variability
    beats, cur = [], 0.5
    while cur < duration - 0.5:
        beats.append(cur)
        cur += 60 / hr + rng.normal(0, 0.02)
    # (offset s, amplitude mV, width s) for P, Q, R, S, T waves
    waves = [(-0.20, 0.15, 0.025), (-0.04, -0.15, 0.010),
             (0.00, 1.00, 0.012), (0.04, -0.25, 0.010), (0.25, 0.30, 0.050)]
    for b in beats:
        for off, amp, w in waves:
            sig += amp * np.exp(-((t - b - off) ** 2) / (2 * w ** 2))
    sig += rng.normal(0, 0.01, n)  # tiny sensor noise
    return t, sig


def add_baseline_wander(sig, t, amp=0.5, f=0.3, start=0, end=None):
    end = end if end is not None else t[-1]
    m = (t >= start) & (t < end)
    sig[m] += amp * np.sin(2 * np.pi * f * t[m])


def add_powerline(sig, t, amp=0.2, f=50, start=0, end=None):
    end = end if end is not None else t[-1]
    m = (t >= start) & (t < end)
    sig[m] += amp * np.sin(2 * np.pi * f * t[m])


def add_muscle_noise(sig, t, amp=0.3, start=0, end=None):
    end = end if end is not None else t[-1]
    m = (t >= start) & (t < end)
    sig[m] += rng.normal(0, amp, m.sum())


def add_motion_artifact(sig, t, start, end, amp=2.0):
    m = (t >= start) & (t < end)
    x = t[m] - start
    sig[m] += amp * np.sin(2 * np.pi * 1.2 * x) * np.hanning(m.sum()) + rng.normal(0, 0.2, m.sum())


def add_electrode_pop(sig, t, at, amp=3.0):
    i = int(at * FS)
    sig[i:i + 3] += amp
    sig[i + 3:i + 25] += amp * np.exp(-np.arange(22) / 6)


def add_flatline(sig, t, start, end):
    m = (t >= start) & (t < end)
    sig[m] = sig[m][0]  # lead-off: signal stuck at one value


if __name__ == "__main__":
    # 1) clean
    t, s = clean_ecg()
    pd.DataFrame({"time": t, "ecg": s}).to_csv("ecg_clean.csv", index=False)

    # 2) noisy: drift, hum and some muscle noise in parts of the recording
    t, s = clean_ecg()
    add_baseline_wander(s, t, amp=0.6, start=10, end=25)
    add_powerline(s, t, amp=0.25, start=30, end=45)
    add_muscle_noise(s, t, amp=0.25, start=48, end=55)
    pd.DataFrame({"time": t, "ecg": s}).to_csv("ecg_noisy.csv", index=False)

    # 3) very noisy: everything above + motion, pops, lead-off
    t, s = clean_ecg()
    add_baseline_wander(s, t, amp=0.8, start=5, end=30)
    add_powerline(s, t, amp=0.4, start=20, end=50)
    add_muscle_noise(s, t, amp=0.4, start=35, end=60)
    add_motion_artifact(s, t, start=12, end=18, amp=2.5)
    add_electrode_pop(s, t, at=27.0)
    add_electrode_pop(s, t, at=41.5)
    add_flatline(s, t, start=52, end=57)
    pd.DataFrame({"time": t, "ecg": s}).to_csv("ecg_very_noisy.csv", index=False)

    print(f"Saved 3 CSVs (fs = {FS} Hz, {DURATION} s each)")
