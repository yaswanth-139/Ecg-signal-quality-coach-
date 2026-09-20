"""ECG Signal Quality Coach - run with:  streamlit run app.py"""
import glob
import os

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import ecg_quality as q

MODEL_PATH = "ecg_quality_model.joblib"


def train_model_now():
    """Build the training data and train the model (used on first launch, e.g. on Streamlit Cloud)."""
    import subprocess
    import sys
    for script in ("generate_ecg_data.py", "train_ecg_model.py"):
        subprocess.run([sys.executable, script], check=True, capture_output=True)


@st.cache_resource(show_spinner="Training the AI model for the first time (about 1 minute)...")
def load_model(path):
    try:
        return joblib.load(path)
    except Exception:
        train_model_now()
        return joblib.load(path)


COLORS = {"good": "#2ecc71", "caution": "#f1c40f", "poor": "#e74c3c"}
EMOJI = {"good": "🟢 Good", "caution": "🟡 Caution", "poor": "🔴 Poor"}

st.set_page_config(page_title="ECG Signal Quality Coach", page_icon="🫀", layout="wide")
st.title("🫀 ECG Signal Quality Coach")
st.caption("Upload an ECG recording and find out which parts are safe to analyze, and why the rest are not.")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("1. Load ECG")
    upload = st.file_uploader("Upload a CSV file", type=["csv", "txt"])
    demos = sorted(f for f in glob.glob("ecg_*.csv") if "training" not in f)
    demo = st.selectbox("...or pick a demo file", ["(none)"] + demos)

if upload is None and demo == "(none)":
    st.info("👈 Upload an ECG CSV (a `time` column and an `ecg` column, or just one column of samples) "
            "or pick a demo file in the sidebar to begin.")
    st.stop()

source = upload if upload is not None else demo
try:
    df = q.read_csv_smart(source)
    x, fs_detected, sig_col = q.extract_signal(df)
except Exception as err:
    st.error(f"Could not read this file: {err}")
    st.stop()

with st.sidebar:
    st.header("2. Settings")
    fs = st.number_input("Sampling rate (Hz)", min_value=50, max_value=5000,
                         value=int(fs_detected or 250), step=10,
                         help="Detected automatically when the file has a time column.")
    try:
        bundle = load_model(MODEL_PATH)
    except Exception:
        bundle = None
    if bundle is not None:
        engine = st.radio("Analysis engine", ["AI model (Random Forest)", "Rule-based"])
    else:
        engine = "Rule-based"
        st.caption("No trained model found. Run generate_ecg_data.py, then train_ecg_model.py, to enable the AI engine.")
    use_ai = engine.startswith("AI")
    win_s = st.slider("Analysis window (seconds)", 4, 10, 5)
    mains = st.radio("Power-line frequency (Hz)", [50, 60], horizontal=True)
    use_notch = st.checkbox("Apply notch filter", value=False,
                            help="Removes 50/60 Hz hum before scoring. Try it on the noisy demo file.")

try:
    res = q.analyze(x, float(fs), win_s=float(win_s), mains=float(mains), use_notch=use_notch,
                    model_bundle=bundle if use_ai else None)
except Exception as err:
    st.error(f"Analysis failed: {err}")
    st.stop()

table = res["table"]
st.write(f"**File:** `{getattr(source, 'name', source)}` · signal column `{sig_col}` · "
         f"{len(x) / fs:.0f} s at {fs} Hz · {len(table)} windows of {win_s} s")
if use_ai:
    meta = bundle["meta"]
    st.info(f"🤖 **AI engine:** Random Forest trained on {meta['n_windows']:,} labeled ECG windows. "
            f"On {meta['n_test']:,} unseen test windows it agreed with the true usable / not-usable label "
            f"{meta['usable_acc'] * 100:.0f}% of the time.")


# ------------------------------------------------------------------ plot
def merge_runs(tbl):
    """Merge consecutive windows with the same status into one shaded block."""
    runs = []
    for _, r in tbl.iterrows():
        if runs and runs[-1][2] == r["status"]:
            runs[-1][1] = r["end_s"]
        else:
            runs.append([r["start_s"], r["end_s"], r["status"]])
    return runs


def make_figure(x, res, fs, tbl, max_points=20000):
    xf, peaks, wn = res["xf"], res["peaks"], res["wn"]
    step = max(1, len(x) // max_points)
    t = np.arange(0, len(x), step) / fs
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Raw signal", "Filtered signal (0.5-40 Hz) with detected R-peaks"))
    fig.add_trace(go.Scatter(x=t, y=x[::step], mode="lines", name="Raw",
                             line=dict(width=1, color="#4c78a8")), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=xf[::step], mode="lines", name="Filtered",
                             line=dict(width=1, color="#4c78a8")), row=2, col=1)
    status = tbl["status"].to_numpy()
    if len(peaks):
        clean = status[np.minimum(peaks // wn, len(status) - 1)] == "good"
        for mask, name, color in ((clean, "R-peak (clean window)", COLORS["good"]),
                                  (~clean, "R-peak (flagged window)", COLORS["poor"])):
            if mask.any():
                fig.add_trace(go.Scatter(x=peaks[mask] / fs, y=xf[peaks[mask]], mode="markers", name=name,
                                         marker=dict(size=6, color=color)), row=2, col=1)
    for a, b, s in merge_runs(tbl):
        fig.add_vrect(x0=a, x1=b, fillcolor=COLORS[s], opacity=0.16, line_width=0,
                      layer="below", row="all", col="all")
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude")
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.12))
    return fig


st.plotly_chart(make_figure(x, res, fs, table))
st.caption("Background colour = window quality: 🟢 good · 🟡 caution (e.g. fixable hum) · 🔴 poor")

# ------------------------------------------------------------------ summary
c1, c2, c3 = st.columns(3)
c1.metric("Overall quality score", f"{res['overall']:.0f} / 100")
c2.metric("Clean (green) windows", f"{res['pct_good']:.0f}%")
c3.metric("Clean windows", f"{int((table['status'] == 'good').sum())} of {len(table)}")

if res["segments"]:
    st.success("✅ **Safe to analyze:** " + ", ".join(f"{a:.0f}-{b:.0f} s" for a, b in res["segments"]))
else:
    st.error("❌ No window is clean enough to analyze. Re-record the signal (see the tips below).")


def fmt(v, unit, digits=0):
    return "n/a" if v is None else f"{v:.{digits}f} {unit}"


st.subheader("Why quality matters")
cmp = pd.DataFrame(
    {"Whole recording (naive)": [fmt(res["hr_naive"], "bpm"), fmt(res["rmssd_naive"], "ms")],
     "Clean windows only": [fmt(res["hr_safe"], "bpm"), fmt(res["rmssd_safe"], "ms")]},
    index=["Heart rate", "HRV (RMSSD)"])
st.dataframe(cmp)
if res["rmssd_naive"] and res["rmssd_safe"] and res["rmssd_naive"] > 1.5 * res["rmssd_safe"]:
    st.warning(f"Analyzing the whole recording would overestimate HRV by "
               f"**{res['rmssd_naive'] / res['rmssd_safe']:.1f}x**: noise creates false and missed beats.")

# ------------------------------------------------------------------ guidance
if res["tips"]:
    st.subheader("How to get a better recording")
    st.markdown("\n".join(f"- {t}" for t in res["tips"]))

st.subheader("Window-by-window report")
show = table[["window", "start_s", "end_s", "score", "status", "issues"]].copy()
show["status"] = show["status"].map(EMOJI)
show.columns = ["Window", "Start (s)", "End (s)", "Score", "Status", "Issues found"]
st.dataframe(show, hide_index=True)

with st.expander("Show raw metric values"):
    st.dataframe(table, hide_index=True)
    st.caption("baseline_ratio = drift power share · hf_ratio = >40 Hz noise share · mains_ratio = 50/60 Hz share · "
               "amp_ratio = peak amplitude vs typical · beat_corr = similarity of each beat to the median beat")

if use_ai:
    with st.expander("How the AI was trained"):
        st.markdown(
            "1. **Generate data:** thousands of synthetic ECG windows with random heart rate, shape and noise, "
            "each labeled with the artifacts that were injected (drift, hum, muscle noise, motion, pops, flatline, clipping).\n"
            "2. **Extract features:** 21 numbers per window (frequency-band power, waveform shape, beat regularity).\n"
            "3. **Train:** a Random Forest learns to predict the probability of each artifact.\n"
            "4. **Score:** the window score is 100 minus the most likely problem (hum counts half because a notch filter fixes it).")
        st.caption("What the AI looks at most (feature importance):")
        st.bar_chart(pd.Series(meta["importance"]).head(10))
        st.caption("Accuracy per artifact type on unseen test windows (F1 score, 1.0 = perfect):")
        st.dataframe(pd.Series(meta["f1"], name="F1 score").round(2))
        st.caption("Trained and tested on synthetic data. Real-world validation (e.g. PhysioNet recordings) is future work.")

st.download_button("⬇️ Download report (CSV)", table.to_csv(index=False), "ecg_quality_report.csv", "text/csv")
st.caption("Educational tool: it checks signal quality only and is not a diagnostic device.")
