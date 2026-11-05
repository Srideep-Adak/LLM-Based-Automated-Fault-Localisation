"""Shared code: load cached coverage + ground truth, compute SBFL formulas and ranks."""
import glob, json, os
import numpy as np

FORMULAS = ["ochiai", "tarantula", "dstar", "jaccard", "op2"]


def load_ground_truth(results_glob="fl_*.json"):
    """bug id -> set of (file, line) true fault lines, and the tests that were used."""
    gt, tests = {}, {}
    for f in sorted(glob.glob(results_glob)):
        for r in json.load(open(f)):
            if "error" in r or not r.get("ground_truth"):
                continue
            gt[r["id"]] = {tuple(x) for x in r["ground_truth"]}
            tests[r["id"]] = {t["test"] for t in r["tests"]}
    return gt, tests


def load_bug(cache_dir, bug_id, used_tests, scope="modified"):
    """Return dict with universe, ef, ep arrays, F, P, or None if no cache."""
    path = os.path.join(cache_dir, f"{bug_id}_{scope}.json")
    if not os.path.exists(path):
        return None
    cache = json.load(open(path))
    cache = {t: c for t, c in cache.items() if t in used_tests}
    universe = set()
    for c in cache.values():
        universe |= {tuple(x) for x in c["uni"]}
    universe = sorted(universe)
    index = {k: i for i, k in enumerate(universe)}
    ef = np.zeros(len(universe)); ep = np.zeros(len(universe))
    F = P = 0
    for c in cache.values():
        arr = ef if c["failed"] else ep
        if c["failed"]: F += 1
        else: P += 1
        for k in c["cov"]:
            i = index.get(tuple(k))
            if i is not None:
                arr[i] += 1
    if F == 0:
        return None
    return {"universe": universe, "index": index, "ef": ef, "ep": ep, "F": F, "P": P}


def formulas(ef, ep, F, P):
    nf = F - ef
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        out["ochiai"] = np.where(ef > 0, ef / np.sqrt(F * (ef + ep)), 0.0)
        fr = ef / F
        pr = ep / P if P else np.zeros_like(ep)
        out["tarantula"] = np.where((fr + pr) > 0, fr / (fr + pr), 0.0)
        den = ep + nf
        out["dstar"] = np.where(ef > 0, np.where(den > 0, ef ** 2 / np.where(den > 0, den, 1), ef ** 2 * 1e6), 0.0)
        out["jaccard"] = np.where(ef > 0, ef / (F + ep), 0.0)
        out["op2"] = ef - ep / (P + 1)
    return out


def avg_ranks(scores, idxs):
    """Average-tie rank (1 = most suspicious) of the given line indices."""
    a = np.sort(scores)
    n = len(a)
    res = []
    for i in idxs:
        s = scores[i]
        hi = np.searchsorted(a, s, "right"); lo = np.searchsorted(a, s, "left")
        better = n - hi; ties = hi - lo
        res.append(better + (ties + 1) / 2)
    return np.array(res)


def metrics_from_ranks(ranks, n_lines):
    """ranks = ranks of all true fault lines of one bug."""
    first = float(ranks.min())
    return {"first": first, "mar": float(ranks.mean()), "exam": first / n_lines}


def summarize(rows, ns=(1, 3, 5, 10)):
    """rows: list of dicts with first/mar/exam. Returns summary dict."""
    n = len(rows)
    first = np.array([r["first"] for r in rows])
    out = {"bugs": n}
    for k in ns:
        out[f"top{k}"] = int((first <= k).sum())
        out[f"top{k}%"] = round(100 * (first <= k).mean(), 1)
    out["MFR"] = round(float(first.mean()), 2)
    out["MAR"] = round(float(np.mean([r["mar"] for r in rows])), 2)
    out["EXAM%"] = round(100 * float(np.mean([r["exam"] for r in rows])), 2)
    return out