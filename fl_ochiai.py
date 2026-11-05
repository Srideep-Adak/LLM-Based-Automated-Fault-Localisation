#!/usr/bin/env python3
"""
Spectrum-based fault localization (Ochiai + Tarantula) on Defects4J bugs.

Run INSIDE Ubuntu/WSL. Needs preprocess_d4j_full.py in the SAME folder (~/).

    python3 ~/fl_ochiai.py --bugs Lang-1 --max-passing 10

How it works
  1. check out the buggy version, compile, list the relevant test methods
  2. run every failing (trigger) test and a sample of passing tests ONE BY ONE
     with `defects4j coverage -t Class::method` and record the covered lines
  3. score each line (Ochiai, Tarantula) and rank them
  4. compare with the true fault lines (from the Defects4J patch) and report
     the rank of the first faulty line + Top-N / MFR over all bugs

Options
  --bugs Lang-1 Math-5 ...   bugs to localize
  --project Lang --limit 5   or: the first N bugs of a project
  --max-same-class N         passing tests from the failing test's own class
                             (default 100, 0 = all). Most informative tests.
  --max-passing N            passing tests sampled from OTHER classes
                             (default 20, 0 = all). Each run takes ~5 s.
  --scope modified|loaded    instrument only the modified classes (default,
                             fast) or all loaded classes (bigger search space)
  --seed N                   sampling seed (default 1)
  --out PATH                 results JSON (default ~/work/fl_results.json)
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import preprocess_d4j_full as pp
except ImportError:
    sys.exit("preprocess_d4j_full.py must be in the same folder as fl_ochiai.py "
             "(copy it to ~/ first).")


# =============================================================================
# Defects4J helpers
# =============================================================================
def d4j(a, args, timeout=1800):
    return pp.run(f'"{a.d4j_bin}" {args}', timeout)


def parse_all_tests(path):
    """all_tests lines look like 'testFoo(pkg.Class)' -> 'pkg.Class::testFoo'."""
    out = []
    for l in pp.read_list(path):
        m = re.match(r"^([\w$]+)\(([\w.$]+)\)$", l)
        if m:
            out.append(f"{m.group(2)}::{m.group(1)}")
        elif "::" in l:
            out.append(l)
    return out


def scan_test_methods(wd, test_dir, classes):
    """Fallback if all_tests is unavailable: find test methods in the sources."""
    out = []
    for cls in classes:
        rel = cls.split("$")[0].replace(".", "/") + ".java"
        path = pp.resolve_file(wd, test_dir, rel)
        if not path:
            continue
        src = pp.scan_java(pp.read_text(path), False, True)
        names = set(re.findall(r"public\s+void\s+(test\w*)\s*\(", src))
        names |= set(re.findall(r"@Test\b[^{;]*?\bvoid\s+(\w+)\s*\(", src, re.S))
        out += [f"{cls}::{n}" for n in sorted(names)]
    return out


def run_test_coverage(a, wd, test, instr):
    """Run ONE test with coverage. Returns (failed, covered_set, universe_set)."""
    for f in ("coverage.xml", "failing_tests"):
        p = os.path.join(wd, f)
        if os.path.exists(p):
            os.remove(p)
    extra = f'-i "{instr}"' if instr else ""
    rc, out, err = d4j(a, f'coverage -w "{wd}" -t "{test}" {extra}')
    xmlp = os.path.join(wd, "coverage.xml")
    if not os.path.exists(xmlp):
        return None, (err or out)[-400:]
    covered, universe = set(), set()
    for cls in ET.parse(xmlp).getroot().iter("class"):
        fn = cls.get("filename")
        for ln in cls.iter("line"):
            key = (fn, int(ln.get("number")))
            universe.add(key)
            if int(ln.get("hits", "0")) > 0:
                covered.add(key)
    ft = os.path.join(wd, "failing_tests")
    failed = os.path.exists(ft) and any(
        l.startswith("--- ") for l in pp.read_text(ft).split("\n"))
    return (failed, covered, universe), ""


def ground_truth(a, wd, proj, bid, src_dir):
    """True fault lines from the patch, with automatic patch-direction detection."""
    patch = os.path.join(a.proj_root, proj, "patches", f"{bid}.src.patch")
    gt = set()
    for fname, rf in pp.parse_patch_raw(patch).items():
        if rf["is_new"] or rf["is_deleted"]:
            continue
        path = pp.resolve_file(wd, src_dir, fname)
        if not path:
            continue
        blines = pp.split_lines(pp.read_text(path))
        best = None
        for rev in (True, False):                     # True = fixed->buggy
            pf = pp.orient({fname: rf}, rev)[fname]
            _, mm = pp.apply_hunks(blines, pf["hunks"])
            if best is None or mm < best[0]:
                best = (mm, pf)
        pf = best[1]
        rel = os.path.relpath(path, os.path.join(wd, src_dir)).replace(os.sep, "/")
        lines = {n for n, t in pf["removed"] if pp.is_code(t)}
        lines |= {min(max(n, 1), max(len(blines), 1)) for n in pf["insertions"]}
        gt |= {(rel, n) for n in lines}
    return gt


# =============================================================================
# Scoring and ranking
# =============================================================================
def score_lines(spectra, universe):
    F = sum(1 for s in spectra.values() if s["failed"])
    P = len(spectra) - F
    ef, ep = {}, {}
    for s in spectra.values():
        for key in s["cov"]:
            d = ef if s["failed"] else ep
            d[key] = d.get(key, 0) + 1
    ochiai, taran = {}, {}
    for key in universe:
        f, p = ef.get(key, 0), ep.get(key, 0)
        ochiai[key] = f / math.sqrt(F * (f + p)) if f > 0 else 0.0
        fr = f / F if F else 0.0
        pr = p / P if P else 0.0
        taran[key] = fr / (fr + pr) if (fr + pr) > 0 else 0.0
    return {"ochiai": ochiai, "tarantula": taran}


def rank_first_fault(scores, gt):
    """Rank of the first faulty line (best / average / worst case for ties)."""
    vals = list(scores.values())
    best = None
    for g in gt:
        if g not in scores:
            continue
        s = scores[g]
        better = sum(1 for v in vals if v > s)
        ties = sum(1 for v in vals if v == s)
        r = {"best": better + 1, "avg": better + (ties + 1) / 2, "worst": better + ties}
        if best is None or r["avg"] < best["avg"]:
            best = r
    return best


def normalise_gt(gt, universe):
    files = {fn for fn, _ in universe}
    out = set()
    for rel, n in gt:
        for fn in files:
            if rel == fn or rel.endswith("/" + fn) or fn.endswith("/" + rel):
                out.add((fn, n))
    return out


# =============================================================================
# One bug
# =============================================================================
def localize(a, proj, bid):
    bug_id = f"{proj}-{bid}"
    wd = os.path.join(a.workdir, f"{proj}_{bid}b")
    pdir = os.path.join(a.proj_root, proj)

    if not os.path.isdir(wd):
        rc, _, err = d4j(a, f'checkout -p {proj} -v {bid}b -w "{wd}"')
        if rc != 0:
            return {"id": bug_id, "error": "checkout failed: " + err[-200:]}
    props = pp.read_props(wd)
    src_dir = pp.get_dir(a.d4j_bin, wd, props, "d4j.dir.src.classes", "dir.src.classes")
    test_dir = pp.get_dir(a.d4j_bin, wd, props, "d4j.dir.src.tests", "dir.src.tests")

    rc, _, err = d4j(a, f'compile -w "{wd}"')
    if rc != 0:
        return {"id": bug_id, "error": "compile failed: " + err[-200:]}

    # ---- which tests? ------------------------------------------------------
    relevant = pp.read_list(os.path.join(pdir, "relevant_tests", bid))
    failing = [t["test"] for t in
               pp.parse_trigger_tests(os.path.join(pdir, "trigger_tests", bid))]
    # The test list is saved ONCE: every per-test coverage run overwrites the
    # checkout's all_tests file with just that one test.
    os.makedirs(a.cache_dir, exist_ok=True)
    pool_path = os.path.join(a.cache_dir, f"{bug_id}_pool.json")
    if os.path.exists(pool_path):
        pool = json.load(open(pool_path))
    else:
        all_tests_file = os.path.join(wd, "all_tests")
        if os.path.exists(all_tests_file):
            os.remove(all_tests_file)
        d4j(a, f'test -r -w "{wd}"')
        if not os.path.exists(all_tests_file):
            d4j(a, f'test -w "{wd}"')
        methods = parse_all_tests(all_tests_file)
        rel_set = set(relevant)
        pool = [t for t in methods if t.split("::")[0] in rel_set] or methods
        if len(pool) < 5:                       # suspiciously small: scan sources
            pool = list(set(pool) | set(scan_test_methods(wd, test_dir, relevant)))
        pool = sorted(t for t in set(pool) if t not in failing)
        json.dump(pool, open(pool_path, "w"))
    # Tests from the failing test's own class are the most informative, so they
    # are taken first; other classes are only sampled.
    fail_cls = {t.split("::")[0] for t in failing}
    same = [t for t in pool if t.split("::")[0] in fail_cls]
    other = [t for t in pool if t.split("::")[0] not in fail_cls]
    rnd = random.Random(a.seed)
    if a.max_same_class > 0 and len(same) > a.max_same_class:
        same = sorted(rnd.sample(same, a.max_same_class))
    if a.max_passing > 0 and len(other) > a.max_passing:
        other = sorted(rnd.sample(other, a.max_passing))
    print(f"  test pool: {len(pool)} candidates -> using {len(same)} from the "
          f"failing test's class + {len(other)} from other classes")
    pool = same + other
    tests = [(t, True) for t in failing] + [(t, False) for t in pool]

    instr = None
    if a.scope == "loaded":
        instr = os.path.join(pdir, "loaded_classes", f"{bid}.src")
        if not os.path.exists(instr):
            instr = None

    # ---- run tests one by one (cached, so you can stop and resume) ---------
    cache_path = os.path.join(a.cache_dir, f"{bug_id}_{a.scope}.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    spectra, universe = {}, set()
    print(f"  {len(failing)} failing + {len(pool)} passing tests to run")
    for i, (t, _) in enumerate(tests, 1):
        if t not in cache:
            t0 = time.time()
            res, err = run_test_coverage(a, wd, t, instr)
            if res is None:
                print(f"    [{i}/{len(tests)}] {t}: NO COVERAGE ({err[:120]})")
                continue
            failed, cov, uni = res
            cache[t] = {"failed": failed, "cov": sorted(cov), "uni": sorted(uni)}
            json.dump(cache, open(cache_path, "w"))
            print(f"    [{i}/{len(tests)}] {'FAIL' if failed else 'pass'} {t} "
                  f"({time.time() - t0:.0f}s)")
        c = cache[t]
        spectra[t] = {"failed": c["failed"], "cov": {tuple(x) for x in c["cov"]}}
        universe |= {tuple(x) for x in c["uni"]}

    F = sum(1 for s in spectra.values() if s["failed"])
    if F == 0:
        return {"id": bug_id, "error": "no failing test was reproduced"}

    gt = normalise_gt(ground_truth(a, wd, proj, bid, src_dir), universe)
    scores = score_lines(spectra, universe)
    result = {
        "id": bug_id, "num_failing": F, "num_passing": len(spectra) - F,
        "search_space_lines": len(universe), "scope": a.scope,
        "ground_truth": sorted(gt),
        "tests": [{"test": t, "failed": s["failed"]} for t, s in spectra.items()],
    }
    for name, sc in scores.items():
        result[name] = rank_first_fault(sc, gt)
        result["top_" + name] = [
            {"file": k[0], "line": k[1], "score": round(v, 4)}
            for k, v in sorted(sc.items(), key=lambda kv: -kv[1])[:20]]
    if not gt:
        result["note"] = "no ground-truth line inside the instrumented classes"
    return result


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d4j", default="~/defects4j")
    ap.add_argument("--bugs", nargs="*")
    ap.add_argument("--project")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--max-passing", type=int, default=20,
                    help="passing tests sampled from OTHER classes (0 = all)")
    ap.add_argument("--max-same-class", type=int, default=100,
                    help="passing tests taken from the failing test's class (0 = all)")
    ap.add_argument("--scope", choices=["modified", "loaded"], default="modified")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workdir", default="~/work/fl_checkouts")
    ap.add_argument("--cache-dir", default="~/work/fl_cache")
    ap.add_argument("--out", default="~/work/fl_results.json")
    a = ap.parse_args()

    d4j_home = os.path.expanduser(a.d4j)
    a.d4j_bin = os.path.join(d4j_home, "framework", "bin", "defects4j")
    a.proj_root = os.path.join(d4j_home, "framework", "projects")
    a.workdir = os.path.expanduser(a.workdir)
    a.cache_dir = os.path.expanduser(a.cache_dir)
    a.out = os.path.expanduser(a.out)
    os.makedirs(a.workdir, exist_ok=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    jobs = []
    if a.bugs:
        for b in a.bugs:
            proj, _, bid = b.rpartition("-")
            jobs.append((proj, bid))
    elif a.project:
        rows = pp.read_bug_table(os.path.join(a.proj_root, a.project))[:a.limit]
        jobs = [(a.project, r["bug.id"]) for r in rows]
    else:
        sys.exit("Give --bugs Lang-1 ... or --project Lang")

    results = []
    for i, (proj, bid) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {proj}-{bid}")
        try:
            r = localize(a, proj, bid)
        except Exception as e:
            r = {"id": f"{proj}-{bid}", "error": f"{type(e).__name__}: {e}"}
        results.append(r)
        if "error" in r:
            print("  ERROR:", r["error"])
        else:
            for f in ("ochiai", "tarantula"):
                x = r.get(f)
                print(f"  {f:9s} first faulty line rank: "
                      + (f'avg {x["avg"]}  (best {x["best"]}, worst {x["worst"]})'
                         if x else "n/a (fault not in search space)")
                      + f"   of {r['search_space_lines']} lines")
        json.dump(results, open(a.out, "w"), indent=2)

    ok = [r for r in results if "error" not in r]
    print("\n==== SUMMARY over", len(ok), "bugs ====")
    for f in ("ochiai", "tarantula"):
        ranks = [r[f]["avg"] for r in ok if r.get(f)]
        if not ranks:
            continue
        tops = {n: sum(1 for x in ranks if x <= n) for n in (1, 3, 5, 10)}
        print(f"{f:9s} Top-1 {tops[1]}  Top-3 {tops[3]}  Top-5 {tops[5]}  "
              f"Top-10 {tops[10]}  | mean first rank {sum(ranks) / len(ranks):.1f}"
              f"  (of {len(ranks)} bugs)")
    print("Results saved to", a.out)


if __name__ == "__main__":
    main()