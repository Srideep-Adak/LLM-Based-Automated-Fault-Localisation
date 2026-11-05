"""STEP B: predict the fault line for EVERY bug of EVERY project in the result files.

    python predict_all.py                         # all fl_*.json in this folder
    python predict_all.py fl_csv.json fl_lang20.json
    python predict_all.py --out predictions.csv --top 10

Needs fl_cache/<bug>_modified.json for the ML model (falls back to the stored Ochiai
list when a bug has no cache). Prints per-project and overall Top-1/3/5 accuracy.
"""
import argparse, glob, json, warnings
import pandas as pd
from fl_predict import load_bundle, is_result_file, predict_result_file

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="*", help="result files (default: fl_*.json)")
ap.add_argument("--cache", default="fl_cache")
ap.add_argument("--model", default="model.joblib")
ap.add_argument("--top", type=int, default=10)
ap.add_argument("--out", default="predictions_all.csv")
a = ap.parse_args()

files = a.files or sorted(glob.glob("fl_*.json"))
bundle = load_bundle(a.model)
rows = []
for f in files:
    data = json.load(open(f, encoding="utf-8"))
    if not is_result_file(data):
        print("skip (not a result file):", f); continue
    rows += predict_result_file(data, bundle, a.cache, a.top)
df = pd.DataFrame(rows).drop_duplicates("bug", keep="last")
df.insert(1, "project", df.bug.str.rsplit("-", n=1).str[0])
df.to_csv(a.out, index=False)

ok = df[df.hit_top1.notna()].copy()
for c in ("hit_top1", "hit_top3", "hit_top5"):
    ok[c] = ok[c].astype(bool)
summ = ok.groupby("project")[["hit_top1", "hit_top3", "hit_top5"]].agg(["sum"]).droplevel(1, axis=1)
summ.insert(0, "bugs", ok.groupby("project").size())
summ.loc["ALL"] = [len(ok), ok.hit_top1.sum(), ok.hit_top3.sum(), ok.hit_top5.sum()]
pd.set_option("display.width", 200)
print(summ.astype(int).to_string())
print(f"\n{len(df)} bugs, {df.error.notna().sum()} with errors. Saved {a.out}")