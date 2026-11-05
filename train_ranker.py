"""Step 2: LightGBM learning-to-rank on top of SBFL scores, leave-one-project-out validation."""
import argparse, collections
import numpy as np, pandas as pd
import lightgbm as lgb
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fl_common import *

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="fl_cache")
ap.add_argument("--results", default="fl_*.json")
ap.add_argument("--topk", type=int, default=100, help="candidate lines per bug")
a = ap.parse_args()

FEATS = None


def bug_table(d, sc, gtset, K):
    """Feature matrix for the top-K candidate lines of one bug."""
    uni = d["universe"]; n = len(uni)
    ef, ep, F, P = d["ef"], d["ep"], d["F"], d["P"]
    order = np.lexsort((-sc["dstar"], -sc["ochiai"]))[:K]
    ranks = {nm: avg_ranks(sc[nm], order) for nm in FORMULAS}
    mx = {nm: max(sc[nm].max(), 1e-9) for nm in FORMULAS}
    # per-file info over the whole search space
    file_size = collections.Counter(k[0] for k in uni)
    file_max = collections.defaultdict(float)
    by_file = collections.defaultdict(list)
    for i, k in enumerate(uni):
        file_max[k[0]] = max(file_max[k[0]], sc["ochiai"][i])
        by_file[k[0]].append((k[1], sc["ochiai"][i]))
    maxline = {f: max(l for l, _ in v) for f, v in by_file.items()}
    # tie group size under (ef, ep)
    pair = collections.Counter(zip(ef.tolist(), ep.tolist()))
    rows, labels = [], []
    for pos, i in enumerate(order):
        fn, ln = uni[i]
        near = [s for l, s in by_file[fn] if l != ln and abs(l - ln) <= 3]
        r = {"ef": ef[i], "ep": ep[i], "nf": F - ef[i], "F": F, "P": P,
             "ef_ratio": ef[i] / F, "ep_ratio": ep[i] / P if P else 0,
             "tie_group": pair[(ef[i], ep[i])], "search_space": n,
             "file_lines": file_size[fn], "rel_line": ln / maxline[fn],
             "file_max_ochiai": file_max[fn], "neighbor_max": max(near) if near else 0.0}
        for nm in FORMULAS:
            r[nm] = sc[nm][i]; r[nm + "_norm"] = sc[nm][i] / mx[nm]
            r[nm + "_logrank"] = np.log1p(ranks[nm][pos])
        rows.append(r)
        labels.append(int((fn, ln) in gtset))
    return pd.DataFrame(rows), np.array(labels), order


gt, used = load_ground_truth(a.results)
bugs = []
for bug in sorted(gt):
    d = load_bug(a.cache, bug, used[bug])
    if d is None: continue
    idxs = [d["index"][k] for k in gt[bug] if k in d["index"]]
    if not idxs: continue
    sc = formulas(d["ef"], d["ep"], d["F"], d["P"])
    X, y, order = bug_table(d, sc, gt[bug], a.topk)
    base = avg_ranks(sc["ochiai"], idxs)
    bugs.append({"id": bug, "proj": bug.rsplit("-", 1)[0], "X": X, "y": y, "order": order,
                 "idxs": idxs, "n": len(d["universe"]), "base": base, "sc": sc,
                 "dstar": avg_ranks(sc["dstar"], idxs)})
FEATS = list(bugs[0]["X"].columns)
inK = [b for b in bugs if b["y"].sum() > 0]
print(f"{len(bugs)} bugs, {len(inK)} have a fault line inside the top-{a.topk} Ochiai candidates "
      f"({100*len(inK)/len(bugs):.0f}%)")


def fit(train):
    X = pd.concat([b["X"] for b in train]); y = np.concatenate([b["y"] for b in train])
    groups = [len(b["y"]) for b in train]
    m = lgb.LGBMRanker(objective="lambdarank", n_estimators=150, learning_rate=0.05, num_leaves=7,
                       min_child_samples=5, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                       random_state=0, verbose=-1)
    m.fit(X, y, group=groups)
    return m


def ml_ranks(m, b):
    s = m.predict(b["X"])
    pos = {int(i): j for j, i in enumerate(b["order"])}
    out = []
    for k, i in enumerate(b["idxs"]):
        if i in pos:
            j = pos[i]
            out.append(float((s > s[j]).sum() + ((s == s[j]).sum() + 1) / 2))
        else:
            out.append(float(b["base"][k]))          # outside candidates: keep baseline rank
    return np.array(out)


res = collections.defaultdict(list)
imp = np.zeros(len(FEATS))
projects = sorted({b["proj"] for b in bugs})
for p in projects:
    train = [b for b in inK if b["proj"] != p]
    m = fit(train)
    imp += m.booster_.feature_importance("gain") / max(m.booster_.feature_importance("gain").sum(), 1)
    for b in [b for b in bugs if b["proj"] == p]:
        for name, r in (("ochiai", b["base"]), ("dstar", b["dstar"]), ("lightgbm", ml_ranks(m, b))):
            res[(p, name)].append(metrics_from_ranks(r, b["n"]))

rows = []
for name in ("ochiai", "dstar", "lightgbm"):
    for p in projects + ["ALL"]:
        data = [r for (pp_, nn), v in res.items() if nn == name and (p == "ALL" or pp_ == p) for r in v]
        rows.append({"model": name, "project": p, **summarize(data)})
df = pd.DataFrame(rows); df.to_csv("results_lightgbm.csv", index=False)
pd.set_option("display.width", 200)
print("\n=== Overall (leave-one-project-out) ===")
print(df[df.project == "ALL"].drop(columns="project").to_string(index=False))
print("\n=== Top-5 % per project ===")
print(df.pivot(index="project", columns="model", values="top5%").to_string())
print("\n=== MFR per project ===")
print(df.pivot(index="project", columns="model", values="MFR").to_string())

fi = pd.Series(imp / len(projects), index=FEATS).sort_values(ascending=False)
fi.to_csv("feature_importance.csv", header=["gain_share"])
print("\n=== Top 10 features (gain share) ===\n", fi.head(10).round(3).to_string())

o = df[df.project == "ALL"].set_index("model")
fig, ax = plt.subplots(figsize=(7, 4)); w = 0.25
for j, k in enumerate(["top1%", "top5%", "top10%"]):
    ax.bar([i + (j - 1) * w for i in range(3)], [o.loc[m, k] for m in o.index], w, label=k.replace("%", "").title())
ax.set_xticks(range(3)); ax.set_xticklabels(o.index); ax.set_ylabel("% of bugs"); ax.legend()
ax.set_title("Ochiai vs DStar vs LightGBM (leave-one-project-out)"); plt.tight_layout()
plt.savefig("chart_lightgbm.png", dpi=150)