"""Feature extraction + AI scoring helpers, shared by training and the app."""
import numpy as np
from scipy import signal as sp
from scipy.stats import kurtosis, skew

from ecg_quality import bandpass, detect_r_peaks, _longest_run

FS = 250  # the model is trained at 250 Hz; other rates are resampled to this

# what the AI predicts for every window (multi-label: a window can have several problems)
ISSUES = ["baseline", "powerline", "muscle", "motion", "pop", "flatline", "clipping"]

# how much each problem hurts the score (power-line hum is fixable with a notch filter)
SEVERITY = {"baseline": 1.0, "powerline": 0.5, "muscle": 1.0, "motion": 1.0,
            "pop": 1.0, "flatline": 1.0, "clipping": 1.0}

# AI issue name -> key used for the readable text/tips in ecg_quality.py
ISSUE_KEY = {"baseline": "bw", "powerline": "mains", "muscle": "hf", "motion": "motion",
             "pop": "amp", "flatline": "flat", "clipping": "sat"}

FEATURES = [
    "bp_0_07", "bp_07_5", "bp_5_15", "bp_15_40", "bp_hf", "bp_50", "bp_60",
    "kurtosis", "skew", "peak_over_std", "zcr", "hjorth_mob", "hjorth_comp",
    "flat_run_s", "zero_diff_frac", "rail_frac",
    "beats_per_s", "rr_cv", "corr_mean", "corr_min", "amp_cv",
]


def window_features(seg, fs):
    """Turn one ECG window into a fixed-length vector of numbers (the AI's input)."""
    seg = np.asarray(seg, float)
    if abs(fs - FS) > 1e-6:
        seg = sp.resample(seg, int(round(len(seg) * FS / fs)))
    n = len(seg)
    dur = n / FS
    rng = float(seg.max() - seg.min()) + 1e-12
    x = seg - seg.mean()
    std = float(x.std()) + 1e-12

    # 1) where is the signal power? (shares of total power per frequency band)
    f, P = sp.periodogram(x, FS, window="hann")
    total = P[f >= 0.05].sum() + 1e-12

    def band(lo, hi):
        return P[(f >= lo) & (f < hi)].sum() / total

    m50, m60 = np.abs(f - 50) <= 1.0, np.abs(f - 60) <= 1.0
    ft = {
        "bp_0_07": band(0.05, 0.7), "bp_07_5": band(0.7, 5.0),
        "bp_5_15": band(5.0, 15.0), "bp_15_40": band(15.0, 40.0),
        "bp_hf": P[(f >= 40) & ~m50 & ~m60].sum() / total,
        "bp_50": P[m50].sum() / total, "bp_60": P[m60].sum() / total,
    }

    # 2) what does the waveform look like? (shape statistics)
    dx = np.diff(x)
    mob = np.sqrt(dx.var() / (x.var() + 1e-12))
    mob2 = np.sqrt(np.diff(dx).var() / (dx.var() + 1e-12))
    flat = np.abs(np.diff(seg)) <= 1e-6 * rng
    ft.update({
        "kurtosis": kurtosis(x) if std > 1e-9 else 0.0, "skew": abs(skew(x)) if std > 1e-9 else 0.0,
        "peak_over_std": np.abs(x).max() / std,
        "zcr": np.mean(np.diff(np.sign(x - np.median(x))) != 0),
        "hjorth_mob": mob, "hjorth_comp": mob2 / (mob + 1e-12),
        "flat_run_s": _longest_run(flat) / FS, "zero_diff_frac": flat.mean(),
        "rail_frac": max(np.mean(seg >= seg.max() - 1e-3 * rng),
                         np.mean(seg <= seg.min() + 1e-3 * rng)),
    })

    # 3) are the heartbeats regular and similar to each other?
    xf = bandpass(seg, FS)
    peaks = detect_r_peaks(xf, FS)
    ft.update({"beats_per_s": len(peaks) / dur, "rr_cv": 1.0, "corr_mean": 0.0,
               "corr_min": 0.0, "amp_cv": 1.0})
    if len(peaks) >= 3:
        rr = np.diff(peaks) / FS
        ft["rr_cv"] = rr.std() / (rr.mean() + 1e-12)
        amps = np.abs(xf[peaks])
        ft["amp_cv"] = amps.std() / (amps.mean() + 1e-12)
    a, b = int(0.25 * FS), int(0.45 * FS)
    beats = [xf[p - a:p + b] for p in peaks if p - a >= 0 and p + b <= n]
    if len(beats) >= 2:
        beats = np.array([(s - s.mean()) / (s.std() + 1e-12) for s in beats])
        tmpl = np.median(beats, axis=0)
        tmpl = (tmpl - tmpl.mean()) / (tmpl.std() + 1e-12)
        c = beats @ tmpl / beats.shape[1]
        ft["corr_mean"], ft["corr_min"] = float(c.mean()), float(c.min())

    vec = np.array([ft[k] for k in FEATURES], float)
    return np.nan_to_num(vec, nan=0.0, posinf=1e3, neginf=-1e3)


def predict_probs(model, X):
    """Probability of every issue for every row of X -> array (n_rows, n_issues)."""
    out = model.predict_proba(np.atleast_2d(X))
    return np.column_stack([o[:, 1] if o.shape[1] > 1 else np.zeros(len(o)) for o in out])


def score_from_probs(p):
    """Turn issue probabilities into (score 0-100, status, list of flagged issues)."""
    p = np.asarray(p, float)
    hurt = np.array([SEVERITY[k] * p[i] for i, k in enumerate(ISSUES)])
    score = 100.0 * (1.0 - hurt.max())
    status = "good" if score >= 70 else ("caution" if score >= 40 else "poor")
    flagged = [ISSUES[i] for i in np.argsort(-hurt) if p[i] >= 0.5]
    if status != "good" and not flagged:                 # unsure: show the most likely cause
        top = int(np.argmax(hurt))
        if p[top] >= 0.3:
            flagged = [ISSUES[top]]
    return score, status, flagged


def truth_status(y):
    """Status implied by the true labels (used to evaluate the model)."""
    y = np.asarray(y)
    poor = any(y[i] for i, k in enumerate(ISSUES) if k != "powerline")
    return "poor" if poor else ("caution" if y[ISSUES.index("powerline")] else "good")
