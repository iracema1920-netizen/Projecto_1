"""
Run the whole OGP1 pipeline (steps 1-8) in order.

Usage (from the project root folder):
    python run_all.py        run every step, 01 -> 08
    python run_all.py 3      start at step 3 (outputs of steps 1-2 must already exist)

The run stops at the first step that fails and tells you how to resume.
Step 6 (machine learning) takes a few minutes; the others take seconds.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = sorted((ROOT / "src").glob("0[1-8]_*.py"))
start = int(sys.argv[1]) if len(sys.argv) > 1 else 1

t_total = time.time()
for script in STEPS:
    step = int(script.name[:2])
    if step < start:
        continue
    print(f"\n{'=' * 70}\n{script.name}\n{'=' * 70}", flush=True)
    t = time.time()
    result = subprocess.run([sys.executable, str(script)], cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"\nSTOPPED: {script.name} failed (exit code {result.returncode}).\n"
                 f"Fix the error above, then resume with:  python run_all.py {step}")
    print(f"-- {script.name} finished in {time.time() - t:.0f} s", flush=True)

print(f"\nAll steps finished in {(time.time() - t_total) / 60:.1f} min")
print("Outputs: data/processed/, reports/, reports/figures/, powerbi/data/, models/")
