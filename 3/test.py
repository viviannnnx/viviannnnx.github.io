import re, os
here = os.path.dirname(os.path.abspath(__file__))
html = open(os.path.join(here, "index.html"), encoding="utf-8").read()
need = sorted(set(re.findall(r'results/[ab]/[\w.\-]+\.png', html)))
missing = [p for p in need if not os.path.exists(os.path.join(here, p))]
print(f"{len(need) - len(missing)}/{len(need)} found")
print("MISSING:", *missing, sep="\n  ")
print("\nFILES ACTUALLY PRESENT:")
for root, _, files in os.walk(os.path.join(here, "results")):
    for f in sorted(files):
        print(" ", os.path.relpath(os.path.join(root, f), here))