"""Step 2 of the AI pipeline: train the ECG signal-quality model.

Run:  python train_ecg_model.py
Needs: ecg_training_data.csv (from generate_ecg_data.py)
Output: ecg_quality_model.joblib  (loaded automatically by the app)
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from ecg_features import FEATURES, ISSUES, predict_probs, score_from_probs, truth_status

df = pd.read_csv("ecg_training_data.csv")
X, Y = df[FEATURES].to_numpy(), df[ISSUES].to_numpy()
X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f"Training on {len(X_tr)} windows, testing on {len(X_te)} unseen windows")

model = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42)
model.fit(X_tr, Y_tr)

# ---- how well does it detect each problem?
P = predict_probs(model, X_te)
print("\nPer-problem results on unseen windows (F1 = balance of precision and recall):")
f1 = {}
for i, k in enumerate(ISSUES):
    f1[k] = f1_score(Y_te[:, i], P[:, i] >= 0.5, zero_division=0)
    acc = np.mean((P[:, i] >= 0.5) == Y_te[:, i])
    print(f"  {k:10s} accuracy {acc * 100:5.1f}%   F1 {f1[k]:.2f}")

# ---- how well does the final good / caution / poor call match the truth?
pred = [score_from_probs(p)[1] for p in P]
true = [truth_status(y) for y in Y_te]
status_acc = float(np.mean([a == b for a, b in zip(pred, true)]))
usable_acc = float(np.mean([(a == "good") == (b == "good") for a, b in zip(pred, true)]))
print(f"\nGood/caution/poor exact match: {status_acc * 100:.1f}%")
print(f"'Usable or not' agreement:      {usable_acc * 100:.1f}%")

# ---- what did the model learn to look at?
imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nMost important features:")
print(imp.head(8).round(3).to_string())

# refit on all data for the final model, save with metadata for the app
final = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42).fit(X, Y)
joblib.dump({"model": final, "features": FEATURES, "issues": ISSUES,
             "meta": {"n_windows": int(len(df)), "n_test": int(len(X_te)),
                      "status_acc": status_acc, "usable_acc": usable_acc, "f1": f1,
                      "importance": imp.round(4).to_dict()}},
            "ecg_quality_model.joblib", compress=3)
print("\nSaved ecg_quality_model.joblib")
