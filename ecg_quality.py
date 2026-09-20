"""ECG signal-quality analysis. Needs only numpy, scipy, pandas."""
import numpy as np
import pandas as pd
from scipy import signal as sp

# ---------------------------------------------------------------- tuning
# Each metric is turned into a 0..1 sub-score: 1 = clean, 0 = bad.
# (good_limit, bad_limit): metric <= good_limit -> 1.0, metric >= bad_limit -> 0.0
LIMITS = {
    "bw":    (0.10, 0.60),   # baseline-wander power / total power
    "hf":    (0.02, 0.15),   # high-frequency (>40 Hz) power / total power
    "mains": (0.01, 0.10),   # 50/60 Hz power / total power
    "amp":   (1.60, 3.00),   # window peak amplitude / typical window peak amplitude
}
CORR_LIMITS = (0.55, 0.90)   # beat-template correlation: <=0.55 -> 0, >=0.90 -> 1
WEIGHTS = {"bw": 0.20, "hf": 0.15, "mains": 0.10, "amp": 0.20, "corr": 0.35}
GREEN, YELLOW = 70, 40       # score >= 70 good, 40-70 caution, < 40 poor
MIN_FLAT_SECONDS = 1.0       # a constant signal this long = electrode off / flatline
MAX_SAT_FRACTION = 0.05      # >5% of samples stuck at the min/max rail = clipping

ISSUE_TEXT = {
    "bw": "baseline drift / motion",
    "hf": "high-frequency (muscle) noise",
    "mains": "power-line interference",
    "amp": "abnormal amplitude (spike / loose electrode)",
    "corr": "inconsistent beat shape (noise or ectopic beats)",
    "beats": "too few detectable beats",
    "flat": "flatline (electrode off?)",
    "sat": "signal clipping (saturation)",
    "motion": "motion artifact (movement)",
}

TIPS = {
    "bw": "Baseline drift: keep the subject still and relaxed, breathe normally, and re-check electrode adhesion and skin preparation.",
    "hf": "Muscle noise: ask the subject to relax arms and legs and avoid talking or tensing during the recording.",
    "mains": "Power-line hum: move away from mains cables and chargers, keep leads short and untangled, and enable the notch filter.",
    "amp": "Spikes / loose electrode: secure electrodes and cables (strain relief) and avoid touching the leads during recording.",
    "corr": "Irregular beat shape: could be noise or genuine ectopic beats. Inspect these windows manually before analysis.",
    "beats": "Beats are hard to detect: check electrode placement and signal gain.",
    "flat": "Flatline: an electrode or cable is disconnected. Reconnect it and re-record.",
    "sat": "Clipping: the amplifier input range is exceeded. Reduce the gain or check the input range.",
    "motion": "Motion artifact: ask the subject to stay still, and secure cables so they cannot pull on the electrodes.",
}

# ---------------------------------------------------------------- loading
TIME_NAMES = {"time", "t", "timestamp", "seconds", "sec", "time_s", "time_ms"}
PREFERRED = ["ecg", "signal", "lead", "lead_i", "lead1", "mlii", "value", "mv"]


def read_csv_smart(src):
    """Read a CSV that may or may not have a header row."""
    df = pd.read_csv(src)
    try:
        [float(c) for c in df.columns]      # header is actually data
        if hasattr(src, "seek"):
            src.seek(0)
        df = pd.read_csv(src, header=None)
    except ValueError:
        pass
    return df


def extract_signal(df):
    """Return (signal array, sampling rate or None, name of signal column)."""
    num = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    if num.shape[1] == 0:
        raise ValueError("No numeric columns found in the file.")
    lower = {str(c).strip().lower(): c for c in num.columns}
    time_col = next((lower[k] for k in lower if k in TIME_NAMES), None)
    sig_col = next((lower[k] for k in PREFERRED if k in lower), None)
    if sig_col is None:
        others = [c for c in num.columns if c != time_col]
        sig_col = others[0] if others else num.columns[0]
    fs = None
    if time_col is not None:
        dt = np.median(np.diff(num[time_col].dropna().to_numpy(float)))
        if dt > 0:
            if dt > 0.5:                    # looks like milliseconds
                dt /= 1000.0
            fs = 1.0 / dt
            if not 20 <= fs <= 5000:
                fs = None
    x = num[sig_col].interpolate(limit_direction="both").to_numpy(float)
    return x, (round(fs) if fs else None), sig_col


# ---------------------------------------------------------------- filters
def bandpass(x, fs, lo=0.5, hi=40.0, order=4):
    hi = min(hi, 0.45 * fs)
    sos = sp.butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sp.sosfiltfilt(sos, x)


def notch(x, fs, f0=50.0, q=30.0):
    if f0 >= fs / 2:
        return x
    b, a = sp.iirnotch(f0, q, fs=fs)
    return sp.filtfilt(b, a, x)


def baseline(x, fs, fc=0.5):
    sos = sp.butter(3, fc, btype="low", fs=fs, output="sos")
    return sp.sosfiltfilt(sos, x)


# ---------------------------------------------------------------- beats
def detect_r_peaks(xf, fs):
    """Pan-Tompkins-style R-peak detector on the filtered signal."""
    hi = min(15.0, 0.45 * fs)
    sos = sp.butter(2, [5.0, hi], btype="band", fs=fs, output="sos")
    y = sp.sosfiltfilt(sos, xf)
    y = np.diff(y, prepend=y[0]) ** 2
    w = max(1, int(0.12 * fs))
    y = np.convolve(y, np.ones(w) / w, mode="same")
    cand, props = sp.find_peaks(y, height=0, distance=int(0.3 * fs))
    if len(cand) == 0:
        return np.array([], dtype=int)
    thr = 0.3 * np.percentile(props["peak_heights"], 75)
    peaks = cand[props["peak_heights"] >= thr]
    r = int(0.08 * fs)
    refined = []
    for p in peaks:
        a, b = max(0, p - r), min(len(xf), p + r + 1)
        refined.append(a + int(np.argmax(np.abs(xf[a:b]))))
    return np.unique(refined)


def beat_correlations(xf, peaks, fs, pre=0.25, post=0.45):
    """Correlate every beat with the median beat (template) of the recording."""
    a, b = int(pre * fs), int(post * fs)
    segs, idx = [], []
    for p in peaks:
        if p - a >= 0 and p + b < len(xf):
            s = xf[p - a:p + b]
            segs.append((s - s.mean()) / (s.std() + 1e-12))
            idx.append(p)
    if len(segs) < 3:
        return np.array(idx, dtype=int), np.zeros(len(idx))
    segs = np.array(segs)
    tmpl = np.median(segs, axis=0)
    tmpl = (tmpl - tmpl.mean()) / (tmpl.std() + 1e-12)
    return np.array(idx, dtype=int), segs @ tmpl / segs.shape[1]


# ---------------------------------------------------------------- scoring
def _longest_run(mask):
    if not mask.any():
        return 0
    d = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    return int((np.where(d == -1)[0] - np.where(d == 1)[0]).max())


def _sub(value, limits):
    good, bad = limits
    return float(1.0 - np.clip((value - good) / (bad - good), 0.0, 1.0))


def analyze(x, fs, win_s=5.0, mains=50.0, use_notch=False, model_bundle=None):
    """Score every window of the recording. Returns a dict with all results.

    model_bundle: trained AI model (from train_ecg_model.py). If given, the AI decides
    the score / status / issues of each window; otherwise the rule-based scoring is used.
    """
    x = np.asarray(x, float)
    wn = int(win_s * fs)
    nwin = len(x) // wn
    if nwin < 1:
        raise ValueError("Recording is shorter than one analysis window.")
    if len(x) < 5 * fs:
        raise ValueError("Recording is too short (need at least 5 seconds).")

    xm = notch(x, fs, mains) if use_notch else x      # signal used for the noise metrics
    xf = bandpass(xm, fs)
    base = baseline(xm, fs)
    peaks = detect_r_peaks(xf, fs)
    pidx, pcorr = beat_correlations(xf, peaks, fs)
    rng = float(x.max() - x.min()) + 1e-12

    # typical (median) peak amplitude of a window: reference for spike detection
    win_max = np.array([np.max(np.abs(xf[i * wn:(i + 1) * wn] - np.median(xf[i * wn:(i + 1) * wn])))
                        for i in range(nwin)])
    ref_amp = float(np.median(win_max[win_max > 0])) if (win_max > 0).any() else 1.0

    if model_bundle is not None:
        from ecg_features import ISSUE_KEY, ISSUES, predict_probs, score_from_probs, window_features

    rows = []
    for i in range(nwin):
        s, e = i * wn, (i + 1) * wn
        seg, segf = xm[s:e], xf[s:e]
        raw = x[s:e]
        issues = []

        # --- hard failures: flatline and clipping
        flat_s = _longest_run(np.abs(np.diff(raw)) <= 1e-6 * rng) / fs
        sat = float(np.mean((raw >= x.max() - 1e-3 * rng) | (raw <= x.min() + 1e-3 * rng)))
        flat = flat_s >= MIN_FLAT_SECONDS
        saturated = sat > MAX_SAT_FRACTION

        # --- metrics
        bw = float(np.var(base[s:e]) / (np.var(seg) + 1e-12))
        f, P = sp.periodogram(seg - seg.mean(), fs, window="hann")
        total = P[f >= 0.5].sum() + 1e-12
        mains_mask = np.abs(f - mains) <= 1.0
        hf = float(P[(f >= 40.0) & ~mains_mask].sum() / total)
        ml = float(P[mains_mask].sum() / total)
        amp = float(np.max(np.abs(segf - np.median(segf))) / ref_amp)
        m = (pidx >= s) & (pidx < e)
        n_beats = int(m.sum())
        corr = float(np.mean(pcorr[m])) if n_beats >= 2 else 0.0

        sub = {
            "bw": _sub(bw, LIMITS["bw"]),
            "hf": _sub(hf, LIMITS["hf"]),
            "mains": _sub(ml, LIMITS["mains"]),
            "amp": _sub(amp, LIMITS["amp"]),
            "corr": float(np.clip((corr - CORR_LIMITS[0]) / (CORR_LIMITS[1] - CORR_LIMITS[0]), 0, 1)),
        }
        score = 100.0 * sum(WEIGHTS[k] * sub[k] for k in WEIGHTS)

        # one severe problem is enough to make the window red;
        # power-line hum alone is fixable with a notch filter, so it only caps at "caution"
        if any(v < 0.2 for k, v in sub.items() if k != "mains"):
            score = min(score, GREEN - 31)
        elif sub["mains"] < 0.2:
            score = min(score, 55.0)
        for k, v in sub.items():
            if v < 0.6:
                issues.append(ISSUE_TEXT[k])
        if n_beats < 2:
            issues.append(ISSUE_TEXT["beats"])
        if flat:
            score = 0.0
            issues = [ISSUE_TEXT["flat"]]
        elif saturated:
            score = min(score, 20.0)
            issues.append(ISSUE_TEXT["sat"])

        ai_cols = {}
        if model_bundle is not None:                          # the trained AI takes over the decision
            probs = predict_probs(model_bundle["model"], window_features(seg, fs))[0]
            score, _, flagged = score_from_probs(probs)
            issues = [ISSUE_TEXT[ISSUE_KEY[k]] for k in flagged]
            ai_cols = {f"ai_{k}_%": int(round(100 * probs[j])) for j, k in enumerate(ISSUES)}

        status = "good" if score >= GREEN else ("caution" if score >= YELLOW else "poor")
        rows.append({
            "window": i + 1, "start_s": round(s / fs, 2), "end_s": round(e / fs, 2),
            "score": round(score, 1), "status": status,
            "issues": "; ".join(dict.fromkeys(issues)) if issues else "-",
            "beats": n_beats, "baseline_ratio": round(bw, 3), "hf_ratio": round(hf, 3),
            "mains_ratio": round(ml, 3), "amp_ratio": round(amp, 2), "beat_corr": round(corr, 3),
            **ai_cols,
        })

    table = pd.DataFrame(rows)
    hr = heart_rates(peaks, fs, table, wn)
    return {
        "table": table, "xf": xf, "peaks": peaks, "wn": wn,
        "overall": float(table["score"].mean()),
        "pct_good": float((table["status"] == "good").mean() * 100),
        **hr,
        "segments": safe_segments(table),
        "tips": tips_for(table),
    }


def _rmssd(rr_seconds):
    d = np.diff(rr_seconds)
    return float(np.sqrt(np.mean(d ** 2)) * 1000.0) if len(d) >= 2 else None


def heart_rates(peaks, fs, table, wn):
    """Heart rate and HRV (RMSSD) from clean windows only vs. the whole recording."""
    out = {"hr_safe": None, "hr_naive": None, "rmssd_safe": None, "rmssd_naive": None, "n_safe_beats": 0}
    if len(peaks) < 3:
        return out
    good = table["status"].to_numpy() == "good"
    widx = peaks // wn
    safe = np.array([(w < len(good)) and good[w] for w in widx])
    rr = np.diff(peaks) / fs
    ok = safe[:-1] & safe[1:] & (rr > 0.3) & (rr < 2.0)
    out["hr_naive"] = float(60.0 / rr.mean())
    out["rmssd_naive"] = _rmssd(rr)
    if ok.sum() >= 2:
        out["hr_safe"] = float(60.0 / rr[ok].mean())
        both = ok[:-1] & ok[1:]                       # two consecutive clean intervals
        d = np.diff(rr)[both]
        out["rmssd_safe"] = float(np.sqrt(np.mean(d ** 2)) * 1000.0) if len(d) >= 2 else None
    out["n_safe_beats"] = int(ok.sum())
    return out


def safe_segments(table):
    """Merge consecutive good windows into (start, end) segments in seconds."""
    segs, cur = [], None
    for _, r in table.iterrows():
        if r["status"] == "good":
            cur = [r["start_s"], r["end_s"]] if cur is None else [cur[0], r["end_s"]]
        elif cur is not None:
            segs.append(tuple(cur))
            cur = None
    if cur is not None:
        segs.append(tuple(cur))
    return segs


def tips_for(table):
    """Turn the most frequent issues in non-good windows into practical advice."""
    bad = table[table["status"] != "good"]["issues"]
    counts = {}
    for text in bad:
        for k, v in ISSUE_TEXT.items():
            if v in text:
                counts[k] = counts.get(k, 0) + 1
    return [TIPS[k] for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])]
