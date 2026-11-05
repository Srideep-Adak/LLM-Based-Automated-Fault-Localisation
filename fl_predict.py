"""Predict the actual faulty line(s) from a coverage JSON file.

    python fl_predict.py fl_cache/Gson-1_modified.json
    python fl_predict.py my_coverage.json --top 10 --json-out prediction.json

Accepted JSON shapes are described in fl_input.py.
"""
import argparse, json, warnings
import numpy as np
import pandas as pd
import joblib
from fl_common import formulas
from fl_features import spectra_from_cache, bug_table
from fl_input import normalize, load as load_json

warnings.filterwarnings("ignore")


def load_bundle(path="model.joblib"):
    return joblib.load(path)


def check_format(obj):
    """Kept for backward compatibility: validates + normalises, returns the native cache."""
    return normalize(obj)[0]


def confidence_label(p):
    return "high" if p >= 0.5 else "medium" if p >= 0.25 else "low"


def predict(cache, bundle, top=20):
    """Returns (table, bug, info). `cache` may be any shape fl_input understands."""
    cache, notes = normalize(cache)
    d = spectra_from_cache(cache)
    sc = formulas(d["ef"], d["ep"], d["F"], d["P"])
    X, _, order = bug_table(d, sc, set(), bundle["topk"])
    raw = bundle["model"].predict_proba(X[bundle["feats"]])[:, 1]
    cal = bundle["iso_line"].predict(raw)
    # sort by model score, break ties with calibrated prob, then Ochiai
    idx = np.lexsort((-sc["ochiai"][order], -cal, -raw))
    uni = d["universe"]
    n = min(top, len(idx))
    sel = idx[:n]
    table = pd.DataFrame({
        "rank": np.arange(1, n + 1),
        "file": [uni[order[i]][0] for i in sel],
        "line": [uni[order[i]][1] for i in sel],
        "ochiai": [round(float(sc["ochiai"][order[i]]), 3) for i in sel],
        "dstar": [round(float(sc["dstar"][order[i]]), 3) for i in sel],
        "failing_tests_covering": [int(d["ef"][order[i]]) for i in sel],
        "passing_tests_covering": [int(d["ep"][order[i]]) for i in sel],
        "P(line is faulty)": [round(float(cal[i]), 3) for i in sel],
    })
    bug = {}
    for k, iso in bundle["iso_bug"].items():
        bug[k] = float(iso.predict([1 - np.prod(1 - cal[idx[:k]])])[0])
    info = {"failing_tests": d["F"], "passing_tests": d["P"], "lines_covered": len(uni), "notes": notes}
    return table, bug, info


def actual_fault(table, bug, info):
    """Single answer: the most likely faulty line + nearby lines of the same file that are also suspicious."""
    best = table.iloc[0]
    same = table[(table.file == best.file) & (table.line != best.line)
                 & (table["P(line is faulty)"] >= 0.5 * best["P(line is faulty)"])
                 & ((table.line - best.line).abs() <= 5)]
    p = float(best["P(line is faulty)"])
    return {
        "predicted_fault": {"file": best.file, "line": int(best.line),
                            "probability": p, "confidence": confidence_label(p)},
        "also_check_nearby": [{"file": r.file, "line": int(r.line),
                               "probability": float(r["P(line is faulty)"])} for _, r in same.iterrows()],
        "fault_in_top_k_probability": {f"top{k}": round(v, 3) for k, v in bug.items()},
        "stats": {k: v for k, v in info.items() if k != "notes"},
        "notes": info.get("notes", []),
        "ranking": table.to_dict(orient="records"),
    }


def is_result_file(obj):
    return (isinstance(obj, list) and obj and isinstance(obj[0], dict)
            and ("ground_truth" in obj[0] or "top_ochiai" in obj[0] or "error" in obj[0]))


def predict_result_file(results, bundle, cache_dir="fl_cache", top=10):
    """Handle a result file such as fl_csv.json (a list of bugs).

    For every bug: if fl_cache/<id>_modified.json exists, use the full ML model;
    otherwise fall back to the Ochiai top list stored inside the result file.
    The true fault (ground_truth) is compared with the prediction when present.
    """
    import os
    out = []
    for r in results:
        bid = r.get("id", "?")
        row = {"bug": bid, "method": None, "file": None, "line": None, "probability": None,
               "true_fault": None, "hit_top1": None, "hit_top3": None, "hit_top5": None, "error": None}
        gt = {(f, int(l)) for f, l in (r.get("ground_truth") or [])}
        row["true_fault"] = "; ".join(f"{f.split('/')[-1]}:{l}" for f, l in sorted(gt)) or None
        if "error" in r:
            row["error"] = str(r["error"])[:80]; out.append(row); continue
        path = os.path.join(cache_dir, f"{bid}_modified.json")
        ranking = None
        if os.path.exists(path):
            try:
                tbl, bug, info = predict(json.load(open(path, encoding="utf-8")), bundle, top)
                ranking = [(t.file, int(t.line), float(t["P(line is faulty)"])) for _, t in tbl.iterrows()]
                row["method"] = "ML model"
            except ValueError as e:
                row["error"] = str(e)[:80]
        if ranking is None and r.get("top_ochiai"):
            ranking = [(x["file"], int(x["line"]), float(x["score"])) for x in r["top_ochiai"][:top]]
            row["method"] = "Ochiai (no coverage in fl_cache)"
        if not ranking:
            row["error"] = row["error"] or "no coverage and no Ochiai list"; out.append(row); continue
        row["file"], row["line"], row["probability"] = ranking[0][0], ranking[0][1], round(ranking[0][2], 3)
        if gt:
            row["hit_top1"] = (ranking[0][0], ranking[0][1]) in gt
            row["hit_top3"] = any((f, l) in gt for f, l, _ in ranking[:3])
            row["hit_top5"] = any((f, l) in gt for f, l, _ in ranking[:5])
        out.append(row)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("json_file")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--model", default="model.joblib")
    ap.add_argument("--json-out", help="also save the full prediction as JSON")
    a = ap.parse_args()
    bundle = load_bundle(a.model)
    raw = json.load(open(a.json_file, encoding="utf-8"))
    if is_result_file(raw):
        rows = predict_result_file(raw, bundle, top=a.top)
        df = pd.DataFrame(rows).drop(columns=["error"]).assign(error=[r["error"] for r in rows])
        print(df.to_string(index=False))
        ok = [r for r in rows if r["hit_top1"] is not None]
        if ok:
            print(f"\nTop-1 correct: {sum(r['hit_top1'] for r in ok)}/{len(ok)}   "
                  f"Top-5 correct: {sum(r['hit_top5'] for r in ok)}/{len(ok)}")
        if a.json_out:
            json.dump(rows, open(a.json_out, "w"), indent=2); print("saved", a.json_out)
        raise SystemExit
    try:
        cache, _ = normalize(raw)
    except ValueError as e:
        raise SystemExit(f"ERROR: {e}")
    table, bug, info = predict(cache, bundle, a.top)
    res = actual_fault(table, bug, info)
    pf = res["predicted_fault"]
    print(f"\nPREDICTED FAULT: {pf['file']}  line {pf['line']}   "
          f"(P={pf['probability']:.2f}, confidence: {pf['confidence']})")
    if res["also_check_nearby"]:
        print("Also check nearby:", ", ".join(str(x["line"]) for x in res["also_check_nearby"]))
    print(f"\n{info['failing_tests']} failing / {info['passing_tests']} passing tests, {info['lines_covered']} lines\n")
    print(table.to_string(index=False))
    for k, v in bug.items():
        print(f"P(fault in top-{k}) ~ {100 * v:.0f}%")
    if a.json_out:
        json.dump(res, open(a.json_out, "w"), indent=2)
        print("saved", a.json_out)