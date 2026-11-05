"""Feature builder shared by train_model.py and fl_predict.py."""
import collections
import numpy as np
import pandas as pd
from fl_common import FORMULAS, formulas, avg_ranks

TOPK = 100          # candidate lines per bug (top by Ochiai)


def spectra_from_cache(cache):
    """cache = {test: {"failed": bool, "cov": [[file,line],..], "uni": [[file,line],..]}}"""
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
        raise ValueError("No failing test in this file - a failing test is needed to localize a fault.")
    if not universe:
        raise ValueError("No covered lines found in this file.")
    return {"universe": universe, "index": index, "ef": ef, "ep": ep, "F": F, "P": P}


def bug_table(d, sc, gtset, K=TOPK):
    """Feature matrix for the top-K candidate lines of one bug (gtset may be empty)."""
    uni = d["universe"]; n = len(uni)
    ef, ep, F, P = d["ef"], d["ep"], d["F"], d["P"]
    order = np.lexsort((-sc["dstar"], -sc["ochiai"]))[:K]
    ranks = {nm: avg_ranks(sc[nm], order) for nm in FORMULAS}
    mx = {nm: max(sc[nm].max(), 1e-9) for nm in FORMULAS}
    file_size = collections.Counter(k[0] for k in uni)
    file_max = collections.defaultdict(float)
    by_file = collections.defaultdict(list)
    for i, k in enumerate(uni):
        file_max[k[0]] = max(file_max[k[0]], sc["ochiai"][i])
        by_file[k[0]].append((k[1], sc["ochiai"][i]))
    maxline = {f: max(max(l for l, _ in v), 1) for f, v in by_file.items()}
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