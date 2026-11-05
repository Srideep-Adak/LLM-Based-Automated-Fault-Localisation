"""Step 1: compare SBFL formulas (Ochiai, Tarantula, DStar, Jaccard, Op2) on all cached bugs."""
import argparse, collections
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fl_common import *

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="fl_cache")
ap.add_argument("--results", default="fl_*.json")
a = ap.parse_args()

gt, used = load_ground_truth(a.results)
per = collections.defaultdict(list)       # (project, formula) -> rows
skipped = []
for bug in sorted(gt):
    proj = bug.rsplit("-", 1)[0]
    d = load_bug(a.cache, bug, used[bug])
    if d is None:
        skipped.append((bug, "no cache")); continue
    idxs = [d["index"][k] for k in gt[bug] if k in d["index"]]
    if not idxs:
        skipped.append((bug, "fault not in search space")); continue
    sc = formulas(d["ef"], d["ep"], d["F"], d["P"])
    for name in FORMULAS:
        r = avg_ranks(sc[name], idxs)
        per[(proj, name)].append(metrics_from_ranks(r, len(d["universe"])))

projects = sorted({p for p, _ in per})
rows = []
for name in FORMULAS:
    for p in projects + ["ALL"]:
        data = [r for (pp_, nn), v in per.items() if nn == name and (p == "ALL" or pp_ == p) for r in v]
        if data:
            rows.append({"formula": name, "project": p, **summarize(data)})
df = pd.DataFrame(rows)
df.to_csv("results_formulas.csv", index=False)

pd.set_option("display.width", 200)
print("Bugs evaluated:", int(df[(df.project == 'ALL') & (df.formula == 'ochiai')].bugs.iloc[0]))
print("Skipped:", len(skipped), collections.Counter(r for _, r in skipped))
print("\n=== Overall ===")
print(df[df.project == "ALL"].drop(columns="project").to_string(index=False))
print("\n=== Per project (Ochiai) ===")
print(df[df.formula == "ochiai"].drop(columns="formula").to_string(index=False))

# charts
o = df[df.project == "ALL"].set_index("formula")
fig, ax = plt.subplots(figsize=(7, 4))
x = range(len(FORMULAS)); w = 0.25
for j, k in enumerate(["top1%", "top5%", "top10%"]):
    ax.bar([i + (j - 1) * w for i in x], [o.loc[f, k] for f in FORMULAS], w, label=k.replace("%", "").title())
ax.set_xticks(list(x)); ax.set_xticklabels(FORMULAS); ax.set_ylabel("% of bugs"); ax.legend()
ax.set_title("Formula comparison (all bugs)"); plt.tight_layout(); plt.savefig("chart_formulas.png", dpi=150)