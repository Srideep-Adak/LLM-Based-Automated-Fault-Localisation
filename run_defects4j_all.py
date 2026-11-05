#!/usr/bin/env python3
"""STEP A (run in Ubuntu/WSL with Defects4J installed): collect coverage for ALL projects.

    python3 run_defects4j_all.py                      # every project, every bug
    python3 run_defects4j_all.py --projects Lang Math # only some projects
    python3 run_defects4j_all.py --limit 20           # first 20 bugs of each project
    python3 run_defects4j_all.py --list               # just show projects + bug counts

For each project it calls fl_ochiai.py (must be in the same folder, together with
preprocess_d4j_full.py) and writes   fl_<project>_all.json   (results + ground truth).
Per-test coverage is cached in ~/work/fl_cache/ (copy that folder into your project).
Safe to stop and re-run: finished tests are cached, finished projects are skipped.
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    import preprocess_d4j_full as pp
except ImportError:
    sys.exit("Put preprocess_d4j_full.py and fl_ochiai.py in the same folder as this script.")

ap = argparse.ArgumentParser()
ap.add_argument("--d4j", default="~/defects4j")
ap.add_argument("--projects", nargs="*", help="default: all projects found in Defects4J")
ap.add_argument("--limit", type=int, default=100000, help="max bugs per project")
ap.add_argument("--max-passing", type=int, default=20)
ap.add_argument("--max-same-class", type=int, default=100)
ap.add_argument("--scope", choices=["modified", "loaded"], default="modified")
ap.add_argument("--workdir", default="~/work/fl_checkouts")
ap.add_argument("--cache-dir", default="~/work/fl_cache")
ap.add_argument("--outdir", default=".", help="where fl_<project>_all.json is written")
ap.add_argument("--redo", action="store_true", help="redo projects that already have a result file")
ap.add_argument("--list", action="store_true")
a = ap.parse_args()

root = os.path.join(os.path.expanduser(a.d4j), "framework", "projects")
if not os.path.isdir(root):
    sys.exit(f"Defects4J projects folder not found: {root}  (use --d4j /path/to/defects4j)")
found = sorted(p for p in os.listdir(root)
               if os.path.exists(os.path.join(root, p, "active-bugs.csv")))
projects = a.projects or found
print("Projects:", ", ".join(f"{p}({len(pp.read_bug_table(os.path.join(root, p)))})" for p in projects if p in found))
if a.list:
    sys.exit()

os.makedirs(a.outdir, exist_ok=True)
for p in projects:
    if p not in found:
        print(f"skip {p}: not a Defects4J project"); continue
    out = os.path.join(a.outdir, f"fl_{p.lower()}_all.json")
    if os.path.exists(out) and not a.redo:
        print(f"skip {p}: {out} exists (use --redo)"); continue
    print(f"\n===== {p} =====")
    cmd = [sys.executable, os.path.join(HERE, "fl_ochiai.py"), "--d4j", a.d4j, "--project", p,
           "--limit", str(a.limit), "--max-passing", str(a.max_passing),
           "--max-same-class", str(a.max_same_class), "--scope", a.scope,
           "--workdir", a.workdir, "--cache-dir", a.cache_dir, "--out", out]
    subprocess.run(cmd)
print("\nDone. Copy the fl_*_all.json files and the fl_cache folder into your project folder, then run:")
print("  python train_model.py            # optional: retrain on all projects")
print("  python predict_all.py            # predict faults for every project")