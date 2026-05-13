"""
run_all.py — LenaDena BankBot Phase Orchestrator

Runs all phases in sequence, captures stdout to logs/phaseN_output.txt
so the Streamlit Phase Evidence page can display captured results.

Usage:
  python run_all.py              # run all phases
  python run_all.py --phase 2    # run only phase 2
  python run_all.py --phase 4 5  # run phases 4 and 5
"""

import argparse
import importlib
import io
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

PHASES = {
    2: ("phases.phase2_baseline",      "Phase 2: Baseline Rule-Based Agent"),
    3: ("phases.phase3_llm_prompting",  "Phase 3: LLM Integration & Prompt Engineering"),
    4: ("phases.phase4_rag",            "Phase 4: RAG — Knowledge & Retrieval"),
    5: ("phases.phase5_tools",          "Phase 5: Tool Usage"),
    6: ("phases.phase6_memory_planning","Phase 6: LangGraph Multi-Agent + Memory"),
    7: ("phases.phase7_adaptive",       "Phase 7: Adaptive Behaviour"),
    8: ("phases.phase8_deployment",     "Phase 8: Deployment Readiness"),
    9: ("phases.phase9_evaluation",     "Phase 9: Evaluation & Engineering Review"),
}

DIVIDER = "=" * 70


def run_phase(phase_num: int) -> bool:
    module_path, description = PHASES[phase_num]
    output_file = LOG_DIR / f"phase{phase_num}_output.txt"

    print(f"\n{DIVIDER}")
    print(f"Running {description}...")
    print(DIVIDER)

    buf = io.StringIO()
    start = time.perf_counter()
    success = True

    try:
        module = importlib.import_module(module_path)
        # Tee: write to both stdout and buffer
        class Tee(io.TextIOBase):
            def write(self, s):
                sys.__stdout__.write(s)
                buf.write(s)
                return len(s)
            def flush(self):
                sys.__stdout__.flush()

        with redirect_stdout(Tee()):
            module.main()

    except Exception as e:
        error_msg = f"\n[ERROR in {description}]: {type(e).__name__}: {e}\n"
        sys.__stdout__.write(error_msg)
        buf.write(error_msg)
        success = False

    elapsed = time.perf_counter() - start
    footer = f"\n--- Completed in {elapsed:.1f}s ---\n"
    sys.__stdout__.write(footer)
    buf.write(footer)

    output_file.write_text(buf.getvalue())
    print(f"Output saved → {output_file}")
    return success


def main():
    parser = argparse.ArgumentParser(description="LenaDena BankBot Phase Runner")
    parser.add_argument(
        "--phase", nargs="*", type=int,
        help="Phase numbers to run (e.g. --phase 2 3). Default: all phases."
    )
    args = parser.parse_args()

    phases_to_run = sorted(args.phase) if args.phase else sorted(PHASES.keys())

    # Validate
    invalid = [p for p in phases_to_run if p not in PHASES]
    if invalid:
        print(f"Error: unknown phase numbers {invalid}. Valid: {sorted(PHASES.keys())}")
        sys.exit(1)

    print(DIVIDER)
    print("LenaDena BankBot — Phase Runner")
    print(DIVIDER)
    print(f"Phases to run: {phases_to_run}")
    print("Outputs will be saved to logs/phaseN_output.txt")

    results = {}
    total_start = time.perf_counter()

    for p in phases_to_run:
        ok = run_phase(p)
        results[p] = "✅ OK" if ok else "❌ FAILED"

    total_elapsed = time.perf_counter() - total_start
    print(f"\n{DIVIDER}")
    print("RUN SUMMARY")
    print(DIVIDER)
    for p, status in results.items():
        print(f"  Phase {p}: {status}")
    print(f"\nTotal time: {total_elapsed:.1f}s")
    print(f"Outputs: logs/phaseN_output.txt")

    if any("FAILED" in s for s in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
