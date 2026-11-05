import subprocess, xml.etree.ElementTree as ET

xml = subprocess.run(
    ["wsl", "-e", "bash", "-lc", "cat ~/work/lang_1b/coverage.xml"],
    capture_output=True, text=True).stdout

root = ET.fromstring(xml)
covered, missed = set(), set()
for cls in root.iter("class"):
    fn = cls.get("filename")
    for ln in cls.iter("line"):
        (covered if int(ln.get("hits")) > 0 else missed).add((fn, int(ln.get("number"))))

print("covered lines:", len(covered), "| never executed:", len(missed))
print("sample never executed:", sorted(missed)[:7])