import json
from collections import Counter

path = r"\\wsl.localhost\Ubuntu\home\kiit\work\dataset_processed.json"
data = json.load(open(path, encoding="utf-8"))
print("Bugs in file:", len(data))

dirs = Counter()
missing = 0
for r in data:
    for f in r["files"]:
        d = f.get("patch_direction")
        if d is None:
            missing += 1
        elif f.get("found"):
            dirs[d] += 1
        if len(data) <= 10:
            print(r["id"], d, "mismatch", f.get("patch_mismatch_lines"))

if missing:
    print(f"\nOLD FILE: {missing} entries have no patch_direction. "
          "Regenerate the dataset with the new preprocess_d4j_full.py.")
else:
    print("\nDirections:", dict(dirs))
    print("Bugs with mismatch > 0:",
          sum(1 for r in data if r["summary"]["patch_mismatch_lines"] > 0))