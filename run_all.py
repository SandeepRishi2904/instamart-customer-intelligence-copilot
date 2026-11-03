"""Runs the whole Copilot pipeline (Parts A-D) in order with one command: python run_all.py
Stops at the first script that fails. Run it from the repository root, inside the virtual environment."""
import subprocess
import sys
import time

STEPS = [
    "generate_data.py",
    "src/part_a_audit.py",
    "src/part_a_clean.py",
    "src/validate.py",
    "src/part_a_sql.py",
    "src/part_a_pandas.py",
    "src/part_b_segments.py",
    "src/part_b_lapse.py",
    "src/part_c_triage.py",
    "src/part_d_run.py",
]

for i, script in enumerate(STEPS, 1):
    print(f"\n===== [{i}/{len(STEPS)}] python {script} =====", flush=True)
    start = time.time()
    if subprocess.run([sys.executable, script]).returncode != 0:
        sys.exit(f"\nSTOPPED: {script} failed (see the error above).")
    print(f"----- done in {time.time() - start:.1f}s", flush=True)
print("\nALL STEPS COMPLETED")
