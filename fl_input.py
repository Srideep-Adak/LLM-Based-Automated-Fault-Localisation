"""Read a coverage JSON in several common shapes and convert it to the internal format

    {test_name: {"failed": bool, "cov": [[file, line], ...], "uni": [[file, line], ...]}}

Accepted shapes
---------------
1. Native (what fl_cache/*_modified.json uses)
       {"TestA::t1": {"failed": true, "cov": [["a/B.java", 10], ...], "uni": [...]}}
2. Wrapped:   {"tests": ...}  /  {"coverage": ...}  /  {"data": ...}  around any shape here
3. List of tests
       [{"test": "TestA::t1", "failed": true, "cov": [...]}, ...]
4. Line formats inside cov / uni (any of these, can be mixed):
       ["a/B.java", 10]      "a/B.java:10"      {"file": "a/B.java", "line": 10}
       {"a/B.java": [10, 11, 12]}   (file -> list of lines)
5. Pass/fail flag written as any of:
       "failed": true | "passed": false | "status": "FAIL"/"PASS" | "result": "failure"/"success"
6. "uni" (all instrumented lines) is optional. If missing it is built from all covered lines.
"""
import json

_CONTAINERS = ("tests", "coverage", "data", "spectra", "results")
_COV_KEYS = ("cov", "covered", "covered_lines", "coverage", "lines")
_UNI_KEYS = ("uni", "universe", "all_lines", "instrumented", "instrumented_lines")
_NAME_KEYS = ("test", "name", "id", "test_name")


def _lines(x):
    """Convert any supported line collection to a list of (file, line) tuples."""
    out = []
    if x is None:
        return out
    if isinstance(x, dict):                              # {file: [lines]}
        for f, ls in x.items():
            if isinstance(ls, (list, tuple, set)):
                out += [(str(f), int(l)) for l in ls]
            else:
                raise ValueError(f"Line list for '{f}' must be a list of numbers.")
        return out
    if not isinstance(x, (list, tuple)):
        raise ValueError("Covered lines must be a list.")
    for item in x:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out.append((str(item[0]), int(item[1])))
        elif isinstance(item, str) and ":" in item:
            f, l = item.rsplit(":", 1)
            out.append((f, int(l)))
        elif isinstance(item, dict) and "file" in item and "line" in item:
            out.append((str(item["file"]), int(item["line"])))
        else:
            raise ValueError(f"Cannot understand line entry: {item!r}")
    return out


def _failed(c):
    if "failed" in c:
        return bool(c["failed"])
    if "passed" in c:
        return not bool(c["passed"])
    for k in ("status", "result", "outcome"):
        if k in c:
            return str(c[k]).strip().lower() in ("fail", "failed", "failure", "error", "false", "ko")
    raise ValueError("Test has no pass/fail flag (use 'failed': true/false).")


def _first(c, keys):
    for k in keys:
        if k in c:
            return c[k]
    return None


def normalize(obj):
    """Return (cache, notes). Raises ValueError with a friendly message if the shape is wrong."""
    notes = []
    for _ in range(3):                                   # unwrap {"tests": ...} etc.
        if isinstance(obj, dict) and len(obj) <= 4:
            hit = [k for k in _CONTAINERS if k in obj and isinstance(obj[k], (dict, list))]
            if hit and not ({"failed", "cov"} & set(obj)):
                obj = obj[hit[0]]
                continue
        break

    if isinstance(obj, list):
        if obj and isinstance(obj[0], dict) and ("ground_truth" in obj[0] or "ochiai" in obj[0]):
            raise ValueError("This is a result file (fl_*.json), not a coverage file. "
                             "It has no per-test line coverage, so a fault cannot be predicted from it. "
                             "Use the matching fl_cache/<bug>_modified.json instead.")
        items = []
        for c in obj:
            if not isinstance(c, dict):
                raise ValueError("Each test in the list must be an object.")
            name = _first(c, _NAME_KEYS)
            if name is None:
                raise ValueError("A test in the list has no 'test' name.")
            items.append((str(name), c))
    elif isinstance(obj, dict) and obj:
        items = list(obj.items())
    else:
        raise ValueError("Empty or unsupported JSON. Expected {test: {failed, cov, uni}}.")

    cache = {}
    for name, c in items:
        if not isinstance(c, dict):
            raise ValueError(f"Test '{name}' must be an object with failed + cov.")
        cov = _lines(_first(c, _COV_KEYS))
        uni_raw = _first(c, _UNI_KEYS)
        uni = _lines(uni_raw) if uni_raw is not None else []
        if uni_raw is None:
            notes.append("no 'uni' list found; search space built from covered lines")
        cache[name] = {"failed": _failed(c),
                       "cov": [list(k) for k in cov],
                       "uni": [list(k) for k in (uni or cov)]}
    notes = sorted(set(notes))
    if not any(c["failed"] for c in cache.values()):
        raise ValueError("No failing test found. A fault can only be localised when at least one test fails.")
    if not any(c["cov"] for c in cache.values()):
        raise ValueError("No covered lines found in any test.")
    return cache, notes


def load(path_or_file):
    """Load from a path or an open file object (e.g. Streamlit upload)."""
    if hasattr(path_or_file, "read"):
        obj = json.load(path_or_file)
    else:
        with open(path_or_file, encoding="utf-8") as fh:
            obj = json.load(fh)
    return normalize(obj)