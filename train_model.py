"""Train the line-level fault model and save it to model.joblib.

    python train_model.py                 (uses ./fl_cache and ./fl_*.json)

Outputs: model.joblib + a leave-one-project-out comparison (Ochiai vs the model).
"""
import argparse
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from fl_common import *
from fl_features import bug_table, TOPK

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="fl_cache")
ap.add_argument("--results", default="fl_*.json")
ap.add_argument("--out", default="model.joblib")
a = ap.parse_args()

gt, used = load_ground_truth(a.results)
bugs = []
for bug in sorted(gt):
    d = load_bug(a.cache, bug, used[bug])
    if d is None:
        continue
    idxs = [d["index"][k] for k in gt[bug] if k in d["index"]]
    sc = formulas(d["ef"], d["ep"], d["F"], d["P"])
    X, y, order = bug_table(d, sc, gt[bug])
    bugs.append({"id": bug, "proj": bug.rsplit("-", 1)[0], "X": X, "y": y, "order": order,
                 "idxs": idxs, "found": len(idxs) > 0, "n": len(d["universe"]),
                 "base": avg_ranks(sc["ochiai"], idxs) if idxs else None})
if not bugs:
    raise SystemExit("No bugs loaded - check fl_cache/ and the fl_*.json files are in this folder.")
FEATS = list(bugs[0]["X"].columns)
found = [b for b in bugs if b["found"]]
print(f"{len(bugs)} bugs with cache, {len(found)} with the fault inside the search space")


def new_model():
    return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=7,
                              min_child_samples=10, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, random_state=0, verbose=-1)


def fit(train):
    X = pd.concat([b["X"] for b in train]); y = np.concatenate([b["y"] for b in train])
    m = new_model(); m.fit(X, y)
    return m


# ---- leave-one-project-out predictions (honest, never trained on the tested project) ----
for p in sorted({b["proj"] for b in bugs}):
    m = fit([b for b in found if b["proj"] != p])
    for b in bugs:
        if b["proj"] == p:
            b["oof"] = m.predict_proba(b["X"][FEATS])[:, 1]

# ---- calibrate: raw score -> real probability that the line is faulty ----
raw = np.concatenate([b["oof"] for b in found]); lab = np.concatenate([b["y"] for b in found])
iso_line = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw, lab)

# ---- bug level: probability that the fault is inside the model's top-k lines ----
iso_bug, ks = {}, (1, 3, 5)
for k in ks:
    xs, ys = [], []
    for b in bugs:                                    # includes bugs whose fault is not in the search space
        top = np.argsort(-b["oof"], kind="stable")[:k]
        pc = iso_line.predict(b["oof"][top])
        xs.append(1 - np.prod(1 - pc)); ys.append(int(b["y"][top].sum() > 0))
    iso_bug[k] = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(xs, ys)

# ---- report: Ochiai vs model, same bugs ----
rows = {"ochiai": [], "model": []}
for b in found:
    rows["ochiai"].append(metrics_from_ranks(b["base"], b["n"]))
    s = b["oof"]; pos = {int(i): j for j, i in enumerate(b["order"])}
    r = []
    for k, i in enumerate(b["idxs"]):
        if i in pos:
            j = pos[i]; r.append((s > s[j]).sum() + ((s == s[j]).sum() + 1) / 2)
        else:
            r.append(b["base"][k])
    rows["model"].append(metrics_from_ranks(np.array(r, dtype=float), b["n"]))
stats = {nm: summarize(v) for nm, v in rows.items()}
print(pd.DataFrame(stats).T.to_string())

final = fit(found)
joblib.dump({"model": final, "feats": FEATS, "iso_line": iso_line, "iso_bug": iso_bug,
             "topk": TOPK, "stats": stats, "n_bugs": len(bugs), "n_found": len(found)}, a.out)
print("saved", a.out)