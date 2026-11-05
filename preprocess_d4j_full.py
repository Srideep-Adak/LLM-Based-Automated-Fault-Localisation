#!/usr/bin/env python3
"""
Full Defects4J preprocessing: metadata + patch + REAL source code -> one JSON.

Run INSIDE Ubuntu/WSL (where Defects4J is installed):
    python3 preprocess_d4j_full.py --projects Lang

For every bug it
  1. checks out the buggy version with `defects4j checkout`
  2. maps every changed line of the patch onto the real source file
  3. extracts the buggy method, the fixed method (patch applied), and the
     code of the failing test methods
  4. cleans the code (comments removed) and tokenizes it

It resumes automatically: finished bugs are stored in <out>.partial.jsonl.

Options:
  --d4j PATH         Defects4J folder        (default ~/defects4j)
  --out PATH         final JSON file         (default ~/work/dataset_processed.json)
  --projects A B     only these projects     (default: all)
  --limit N          only the first N bugs of each project (quick test)
  --workdir PATH     where bugs are checked out (default ~/work/d4j_checkouts)
  --keep-checkouts   do not delete a checkout after processing it
  --no-tokens        skip token lists (smaller file)
  --include-loaded   also store the loaded_classes list
"""
import argparse
import bisect
import csv
import json
import os
import re
import shutil
import subprocess
import sys

CONTROL = {"if", "for", "while", "switch", "catch", "synchronized", "try",
           "else", "do"}
ANNOT_RE = re.compile(r"@\w+(?:\.\w+)*\s*(?:\([^()]*\))?")
CLASS_RE = re.compile(r"\b(class|interface|enum)\s+([A-Za-z_$][\w$]*)")
HUNK_RE = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
TOKEN_RE = re.compile(
    r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|[A-Za-z_$][\w$]*|\d[\w.]*'
    r'|>>>=|<<=|>>=|>>>|\.\.\.|->|::|\+\+|--|&&|\|\||[=!<>+\-*/%&|^]=|<<|>>|\S')


# =============================================================================
# Small helpers
# =============================================================================
def run(cmd, timeout=900):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"


def read_text(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


def split_lines(text):
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def read_list(path):
    if not os.path.exists(path):
        return []
    return [l.strip() for l in read_text(path).split("\n") if l.strip()]


def is_code(text):
    s = text.strip()
    if not s:
        return False
    return not (s.startswith("//") or s.startswith("/*")
                or s.startswith("*") or s.endswith("*/"))


# =============================================================================
# Patch parsing and application
# =============================================================================
def parse_patch_raw(path):
    """Read a unified diff exactly as written (no direction assumed)."""
    files, cur, h = {}, None, None
    rem_old = rem_new = 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if h is not None and (rem_old > 0 or rem_new > 0):
                op = line[:1]
                if op == "\\":
                    continue
                if op in ("-", "+"):
                    text = line[1:]
                else:
                    op, text = " ", line[1:]
                h["lines"].append((op, text))
                if op != "+":
                    rem_old -= 1
                if op != "-":
                    rem_new -= 1
                continue
            if line.startswith("diff --git "):
                m = re.match(r"diff --git a/(.+?) b/(.+)$", line)
                if m:
                    cur = {"hunks": [], "is_new": False, "is_deleted": False}
                    files[m.group(2)] = cur
                h = None
                continue
            if cur is None:
                continue
            if line.startswith("new file mode") or line.startswith("--- /dev/null"):
                cur["is_new"] = True
            elif line.startswith("deleted file mode") or line.startswith("+++ /dev/null"):
                cur["is_deleted"] = True
            else:
                m = HUNK_RE.match(line)
                if m:
                    os_, oc, ns, nc = m.groups()
                    oc = 1 if oc is None else int(oc)
                    nc = 1 if nc is None else int(nc)
                    h = {"os": int(os_), "oc": oc, "ns": int(ns), "nc": nc,
                         "lines": []}
                    cur["hunks"].append(h)
                    rem_old, rem_new = oc, nc
    return files


def orient(raw, reverse):
    """Turn raw hunks into buggy->fixed form.
    Defects4J's *.src.patch goes FIXED->BUGGY, so reverse=True swaps sides:
    after orientation, 'removed' = buggy lines, 'added' = fixed lines."""
    out = {}
    for fname, rf in raw.items():
        c = {"hunks": [], "removed": [], "added": [], "insertions": [],
             "is_new": rf["is_deleted"] if reverse else rf["is_new"],
             "is_deleted": rf["is_new"] if reverse else rf["is_deleted"]}
        for h in rf["hunks"]:
            if reverse:
                os_, oc, ns, nc = h["ns"], h["nc"], h["os"], h["oc"]
                lines = [({"-": "+", "+": "-"}.get(op, op), t) for op, t in h["lines"]]
            else:
                os_, oc, ns, nc, lines = h["os"], h["oc"], h["ns"], h["nc"], h["lines"]

            ordered, blk_r, blk_a = [], [], []        # removals before additions
            for op, t in lines + [(" ", None)]:
                if op == "-":
                    blk_r.append((op, t))
                elif op == "+":
                    blk_a.append((op, t))
                else:
                    ordered += blk_r + blk_a
                    blk_r, blk_a = [], []
                    if t is not None:
                        ordered.append((op, t))

            old_ln = os_ + (1 if oc == 0 else 0)
            new_ln = ns + (1 if nc == 0 else 0)
            prev, run_ins, ins = None, None, []
            for op, t in ordered:
                if op == "-":
                    c["removed"].append((old_ln, t))
                    old_ln += 1
                    run_ins = None
                elif op == "+":
                    if prev in (None, " "):
                        run_ins = [old_ln, [t]]
                        ins.append(run_ins)
                    elif prev == "+" and run_ins is not None:
                        run_ins[1].append(t)
                    c["added"].append((new_ln, t))
                    new_ln += 1
                else:
                    old_ln += 1
                    new_ln += 1
                    run_ins = None
                prev = op
            c["insertions"] += [o for o, texts in ins if any(is_code(x) for x in texts)]
            c["hunks"].append({"old_start": os_, "old_count": oc, "ops": ordered})
        out[fname] = c
    return out


def parse_patch(path, reverse=False):
    return orient(parse_patch_raw(path), reverse)


def apply_hunks(buggy_lines, hunks):
    """Rebuild the FIXED file from buggy lines + hunks. Also counts lines where
    the patch disagrees with the real file (should be 0 for the right direction)."""
    out, pos, mismatch = [], 0, 0
    same = lambda i, t: i < len(buggy_lines) and buggy_lines[i].rstrip() == t.rstrip()
    for h in hunks:
        start = h["old_start"] - 1 if h["old_count"] > 0 else h["old_start"]
        out.extend(buggy_lines[pos:start])
        pos = start
        for op, text in h["ops"]:
            if op == " ":
                if not same(pos, text):
                    mismatch += 1
                out.append(buggy_lines[pos] if pos < len(buggy_lines) else text)
                pos += 1
            elif op == "-":
                if not same(pos, text):
                    mismatch += 1
                pos += 1
            else:
                out.append(text)
    out.extend(buggy_lines[pos:])
    return out, mismatch


# =============================================================================
# Java scanning (no external dependencies)
# =============================================================================
def scan_java(src, mask_strings, drop_comments):
    """Remove/blank comments and (optionally) blank string/char literals.
    Newlines are always preserved when blanking, so line numbers stay valid."""
    out, i, n = [], 0, len(src)
    blank = lambda s: re.sub(r"[^\n]", " ", s)
    while i < n:
        c, two = src[i], src[i:i + 2]
        if two == "//":
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append("" if drop_comments else " " * (j - i))
            i = j
        elif two == "/*":
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(" " if drop_comments else blank(src[i:j]))
            i = j
        elif c == '"' or c == "'":
            j = i + 1
            while j < n and src[j] != c and src[j] != "\n":
                j += 2 if src[j] == "\\" else 1
            j = min(j + 1, n)
            out.append(blank(src[i:j]) if mask_strings else src[i:j])
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def classify_block(header, stack):
    h = header.replace("@interface", "interface")
    stripped = ANNOT_RE.sub(" ", h)
    m = CLASS_RE.search(stripped)
    if m:
        return {"kind": "class", "name": m.group(2),
                "is_enum": m.group(1) == "enum", "semi": False}
    top = stack[-1] if stack else None
    if top is None or top["kind"] != "class":
        return {"kind": "other"}
    if top.get("is_enum") and not top["semi"]:
        return {"kind": "other"}            # enum constant body
    hh = stripped.strip()
    if "=" in hh or re.search(r"\bnew\b", hh):
        return {"kind": "other"}            # field initialiser / anonymous class
    if not re.search(r"\)\s*(?:throws\s+[\w.$,\s<>?]+)?$", hh):
        return {"kind": "other"}            # static init, instance init ...
    nm = re.search(r"([A-Za-z_$][\w$]*)\s*\(", hh)
    if not nm or nm.group(1) in CONTROL:
        return {"kind": "other"}
    return {"kind": "method", "name": nm.group(1), "hh": hh,
            "pos": nm.end() - 1}


def param_types(hh, pos):
    depth, end = 0, len(hh)
    for j in range(pos, len(hh)):
        if hh[j] == "(":
            depth += 1
        elif hh[j] == ")":
            depth -= 1
            if depth == 0:
                end = j
                break
    inner, parts, d, cur = hh[pos + 1:end], [], 0, ""
    for ch in inner:
        if ch in "<([":
            d += 1
        elif ch in ">)]":
            d -= 1
        if ch == "," and d == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    types = []
    for p in parts:
        p = re.sub(r"\bfinal\b", " ", ANNOT_RE.sub(" ", p)).strip()
        bits = p.rsplit(None, 1)
        types.append(re.sub(r"\s+", "", bits[0] if len(bits) == 2 else p))
    return types


def find_methods(src):
    """Return every method/constructor with its start and end line."""
    masked = scan_java(src, True, False)
    starts = [0] + [m.end() for m in re.finditer("\n", masked)]
    line_of = lambda off: bisect.bisect_right(starts, off)
    stack, methods, seg = [], [], 0

    for i, ch in enumerate(masked):
        if ch == "{":
            header = masked[seg:i]
            e = classify_block(header, stack)
            e["start_off"] = seg + len(header) - len(header.lstrip())
            e["classes"] = [s["name"] for s in stack if s["kind"] == "class"]
            stack.append(e)
            seg = i + 1
        elif ch == "}":
            if stack:
                e = stack.pop()
                if e["kind"] == "method":
                    types = param_types(e["hh"], e["pos"])
                    cls = ".".join(e["classes"])
                    methods.append({
                        "name": e["name"], "class": cls,
                        "param_types": types,
                        "signature": f'{cls}.{e["name"]}({",".join(types)})',
                        "start_line": line_of(e["start_off"]),
                        "end_line": line_of(i),
                    })
            seg = i + 1
        elif ch == ";":
            if stack and stack[-1]["kind"] == "class":
                stack[-1]["semi"] = True
            seg = i + 1
    return methods


def enclosing(methods, line):
    c = [m for m in methods if m["start_line"] <= line <= m["end_line"]]
    return min(c, key=lambda m: m["end_line"] - m["start_line"]) if c else None


def match_method(m, others):
    for o in others:
        if o["signature"] == m["signature"]:
            return o
    same = [o for o in others if o["class"] == m["class"] and o["name"] == m["name"]]
    if len(same) == 1:
        return same[0]
    same = [o for o in same if len(o["param_types"]) == len(m["param_types"])]
    return same[0] if same else None


# =============================================================================
# Cleaning + tokenizing
# =============================================================================
def clean_code(code):
    t = scan_java(code, False, True)
    lines = [re.sub(r"[ \t]+", " ", l).strip() for l in t.split("\n")]
    return "\n".join(l for l in lines if l)


def tokenize(code):
    toks = []
    for t in TOKEN_RE.findall(scan_java(code, False, True)):
        if t[0] == '"':
            toks.append("<STR>")
        elif t[0] == "'":
            toks.append("<CHAR>")
        elif t[0].isdigit():
            toks.append("<NUM>")
        else:
            toks.append(t)
    return toks


def split_identifier(tok):
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", tok)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return [p.lower() for p in re.split(r"[_$\s]+", s) if p]


def subtokenize(tokens):
    out = []
    for t in tokens:
        if re.match(r"[A-Za-z_$]", t):
            out.extend(split_identifier(t))
        else:
            out.append(t)
    return out


def method_info(m, lines, with_tokens):
    code = "\n".join(lines[m["start_line"] - 1:m["end_line"]])
    d = {"class": m["class"], "name": m["name"], "signature": m["signature"],
         "start_line": m["start_line"], "end_line": m["end_line"],
         "code": code, "code_clean": clean_code(code)}
    if with_tokens:
        d["tokens"] = tokenize(code)
        d["subtokens"] = subtokenize(d["tokens"])
    return d


# =============================================================================
# Defects4J access
# =============================================================================
def read_props(wd):
    p = os.path.join(wd, "defects4j.build.properties")
    d = {}
    if os.path.exists(p):
        for l in read_text(p).split("\n"):
            l = l.strip()
            if l and not l.startswith("#") and "=" in l:
                k, v = l.split("=", 1)
                d[k.strip()] = v.strip()
    return d


def get_dir(d4j, wd, props, key, export_name):
    v = props.get(key)
    if v:
        return v
    rc, out, _ = run(f'"{d4j}" export -p {export_name} -w "{wd}"', 300)
    return out.split("\n")[-1].strip() if rc == 0 and out else ""


def resolve_file(wd, base_dir, rel):
    rel = rel.replace("\\", "/").lstrip("/")
    for c in (os.path.join(wd, base_dir or "", rel), os.path.join(wd, rel)):
        if os.path.isfile(c):
            return c
    target = rel.split("/")
    name, best, best_n = target[-1], None, 0
    for root, dirs, files in os.walk(wd):
        dirs[:] = [d for d in dirs if d != ".git"]
        if name in files:
            full = os.path.join(root, name)
            parts = full.replace("\\", "/").split("/")
            n = 0
            while n < min(len(parts), len(target)) and parts[-1 - n] == target[-1 - n]:
                n += 1
            if n > best_n:
                best, best_n = full, n
    return best if best_n >= min(2, len(target)) else None


def read_bug_table(pdir):
    p = os.path.join(pdir, "active-bugs.csv")
    if not os.path.exists(p):
        return []
    with open(p, newline="", encoding="utf-8", errors="ignore") as f:
        return list(csv.DictReader(f))


def parse_trigger_tests(path):
    tests, cur = [], None
    if not os.path.exists(path):
        return tests
    for line in read_text(path).split("\n"):
        if line.startswith("--- "):
            cur = {"test": line[4:].strip(), "exception": ""}
            tests.append(cur)
        elif cur and not cur["exception"] and line.strip():
            cur["exception"] = line.strip()
    return tests


# =============================================================================
# Per-file / per-bug analysis
# =============================================================================
def analyse_file(wd, src_dir, fname, raw_file, with_tokens):
    default_pf = orient({fname: raw_file}, True)[fname]
    info = {"file": fname, "found": False, "_pf": default_pf}
    if raw_file["is_new"] or raw_file["is_deleted"]:
        info["note"] = "new or deleted file"
        return info
    path = resolve_file(wd, src_dir, fname)
    if not path:
        info["note"] = "source file not found in checkout"
        return info

    blines = split_lines(read_text(path))
    cands = []
    for rev in (True, False):                      # True = fixed->buggy (Defects4J)
        pf_try = orient({fname: raw_file}, rev)[fname]
        fl, mm = apply_hunks(blines, pf_try["hunks"])
        cands.append((mm, 0 if rev else 1, pf_try, fl, rev))
    mismatch, _, pf, flines, rev = min(cands, key=lambda c: (c[0], c[1]))
    info["_pf"] = pf
    info["patch_direction"] = "fixed_to_buggy" if rev else "buggy_to_fixed"
    bmeths = find_methods("\n".join(blines))
    fmeths = find_methods("\n".join(flines))

    removed = {n for n, t in pf["removed"] if is_code(t)}
    insert = {min(max(n, 1), max(len(blines), 1)) for n in pf["insertions"]}
    fault_lines = sorted(removed | insert)

    info.update({"found": True,
                 "path": os.path.relpath(path, wd),
                 "patch_mismatch_lines": mismatch,
                 "fault_lines": fault_lines,
                 "num_methods_buggy_file": len(bmeths)})

    statements, groups, class_level = [], {}, []
    for n in fault_lines:
        m = enclosing(bmeths, n)
        statements.append({
            "line": n,
            "text": blines[n - 1].strip() if 0 < n <= len(blines) else "",
            "kind": "removed" if n in removed else "insertion_point",
            "method": m["signature"] if m else None})
        if m:
            groups.setdefault(m["signature"], (m, []))[1].append(n)
        else:
            class_level.append(n)
    info["fault_statements"] = statements
    info["class_level_fault_lines"] = class_level

    fault_methods = []
    for sig, (m, lns) in groups.items():
        d = method_info(m, blines, with_tokens)
        d["fault_lines"] = lns
        d["fault_line_offsets"] = [n - m["start_line"] for n in lns]
        fm = match_method(m, fmeths)
        d["fixed"] = method_info(fm, flines, with_tokens) if fm else None
        fault_methods.append(d)
    info["fault_methods"] = fault_methods

    added_methods, seen = [], set()
    for n, _ in pf["added"]:
        fm = enclosing(fmeths, n)
        if fm and fm["signature"] not in seen and not match_method(fm, bmeths):
            seen.add(fm["signature"])
            added_methods.append(method_info(fm, flines, with_tokens))
    info["added_methods"] = added_methods
    return info


def trigger_test_code(wd, test_dir, tests):
    out = []
    for t in tests:
        cls, _, meth = t["test"].partition("::")
        rel = cls.split("$")[0].replace(".", "/") + ".java"
        path = resolve_file(wd, test_dir, rel)
        if not path:
            continue
        lines = split_lines(read_text(path))
        for m in find_methods("\n".join(lines)):
            if m["name"] == meth:
                out.append({"test": t["test"],
                            "file": os.path.relpath(path, wd),
                            "start_line": m["start_line"],
                            "end_line": m["end_line"],
                            "code": "\n".join(lines[m["start_line"] - 1:m["end_line"]])})
                break
    return out


def process_bug(a, d4j, proj, pdir, row):
    bid = row["bug.id"]
    src_patch = os.path.join(pdir, "patches", f"{bid}.src.patch")
    if not os.path.exists(src_patch):
        return {"id": f"{proj}-{bid}", "error": "no src patch"}

    raw = parse_patch_raw(src_patch)
    trig = parse_trigger_tests(os.path.join(pdir, "trigger_tests", bid))
    rec = {
        "id": f"{proj}-{bid}", "project": proj, "bug_id": int(bid),
        "revision_buggy": row.get("revision.id.buggy"),
        "revision_fixed": row.get("revision.id.fixed"),
        "report_id": row.get("report.id"), "report_url": row.get("report.url"),
        "modified_classes": read_list(os.path.join(pdir, "modified_classes", f"{bid}.src")),
        "trigger_tests": trig,
        "relevant_tests": read_list(os.path.join(pdir, "relevant_tests", bid)),
        "src_changes": None,
    }
    if a.include_loaded:
        rec["loaded_classes"] = read_list(os.path.join(pdir, "loaded_classes", f"{bid}.src"))

    wd = os.path.join(a.workdir, f"{proj}_{bid}b")
    if os.path.isdir(wd):
        shutil.rmtree(wd, ignore_errors=True)
    rc, _, err = run(f'"{d4j}" checkout -p {proj} -v {bid}b -w "{wd}"')
    if rc != 0:
        return {"id": rec["id"], "error": "checkout failed: " + err[-300:]}

    try:
        props = read_props(wd)
        src_dir = get_dir(d4j, wd, props, "d4j.dir.src.classes", "dir.src.classes")
        test_dir = get_dir(d4j, wd, props, "d4j.dir.src.tests", "dir.src.tests")
        rec["src_dir"], rec["test_dir"] = src_dir, test_dir

        files = [analyse_file(wd, src_dir, f, rf, not a.no_tokens)
                 for f, rf in raw.items()]
        rec["src_changes"] = [{
            "file": fi["file"],
            "removed": [{"line": n, "text": t} for n, t in fi["_pf"]["removed"]],
            "added": [{"line": n, "text": t} for n, t in fi["_pf"]["added"]],
        } for fi in files]
        for fi in files:
            fi.pop("_pf")
        rec["files"] = files
        rec["trigger_test_code"] = trigger_test_code(wd, test_dir, trig)
        rec["summary"] = {
            "files_changed": len(files),
            "files_with_code": sum(1 for f in files if f["found"]),
            "fault_lines": sum(len(f.get("fault_lines", [])) for f in files),
            "fault_methods": sum(len(f.get("fault_methods", [])) for f in files),
            "added_methods": sum(len(f.get("added_methods", [])) for f in files),
            "patch_mismatch_lines": sum(f.get("patch_mismatch_lines", 0) for f in files),
        }
    finally:
        if not a.keep_checkouts:
            shutil.rmtree(wd, ignore_errors=True)
    return rec


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d4j", default="~/defects4j")
    ap.add_argument("--out", default="~/work/dataset_processed.json")
    ap.add_argument("--projects", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workdir", default="~/work/d4j_checkouts")
    ap.add_argument("--keep-checkouts", action="store_true")
    ap.add_argument("--no-tokens", action="store_true")
    ap.add_argument("--include-loaded", action="store_true")
    a = ap.parse_args()

    d4j_home = os.path.expanduser(a.d4j)
    out_path = os.path.expanduser(a.out)
    a.workdir = os.path.expanduser(a.workdir)
    d4j = os.path.join(d4j_home, "framework", "bin", "defects4j")
    proj_root = os.path.join(d4j_home, "framework", "projects")
    if not os.path.isdir(proj_root):
        sys.exit(f"Cannot find {proj_root}. Check --d4j.")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.makedirs(a.workdir, exist_ok=True)
    partial = out_path + ".partial.jsonl"

    done = {}
    if os.path.exists(partial):
        for l in read_text(partial).split("\n"):
            if l.strip():
                r = json.loads(l)
                done[r["id"]] = r
    ok_ids = {k for k, v in done.items() if "error" not in v}

    all_projects = sorted(p for p in os.listdir(proj_root)
                          if os.path.exists(os.path.join(proj_root, p, "active-bugs.csv")))
    projects = a.projects or all_projects

    work = []
    for proj in projects:
        pdir = os.path.join(proj_root, proj)
        rows = read_bug_table(pdir)
        if a.limit:
            rows = rows[:a.limit]
        work += [(proj, pdir, r) for r in rows]

    print(f"{len(work)} bugs to process, {len(ok_ids)} already done earlier")
    with open(partial, "a", encoding="utf-8") as pf:
        for i, (proj, pdir, row) in enumerate(work, 1):
            bug_id = f'{proj}-{row["bug.id"]}'
            if bug_id in ok_ids:
                continue
            try:
                rec = process_bug(a, d4j, proj, pdir, row)
            except Exception as e:                      # keep going
                rec = {"id": bug_id, "error": f"{type(e).__name__}: {e}"}
            done[bug_id] = rec
            pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            pf.flush()
            s = rec.get("summary", {})
            print(f"[{i}/{len(work)}] {bug_id}: "
                  + (f'ERROR {rec["error"][:80]}' if "error" in rec else
                     f'fault_methods={s["fault_methods"]} fault_lines={s["fault_lines"]}'))

    good = sorted((r for r in done.values() if "error" not in r),
                  key=lambda r: (r["project"], r["bug_id"]))
    bad = [{"id": r["id"], "error": r["error"]} for r in done.values() if "error" in r]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(good, f, indent=2, ensure_ascii=False)
    if bad:
        with open(out_path + ".errors.json", "w") as f:
            json.dump(bad, f, indent=2)

    with_m = sum(1 for r in good if r["summary"]["fault_methods"] > 0)
    print(f"\nWrote {len(good)} bugs to {out_path}")
    print(f"  bugs with >=1 fault method extracted: {with_m}")
    print(f"  failed bugs: {len(bad)}" + (f" (see {out_path}.errors.json)" if bad else ""))
    print("Windows path:  \\\\wsl.localhost\\Ubuntu" + out_path.replace("/", "\\"))


if __name__ == "__main__":
    main()