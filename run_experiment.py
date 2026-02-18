import argparse
import random
import subprocess
from pathlib import Path


# ----- Configuration ---------------------------------------------------------

TREATMENTS = ["naive", "blocked", "strassen", "blas"]
BLOCKS = ["cpp", "python", "r"]
REPS_PER_CELL = 10

def build_command(language, algorithm, csv_path):
    if language == "cpp":
        exe_name = {
            "naive": "cpp_naive",
            "blocked": "cpp_blocked",
            "strassen": "cpp_strassen",
            "blas": "cpp_blas",
        }[algorithm]
        return [f"./cpp/{exe_name}", "1", csv_path]

    elif language == "python":
        # Assumes Python scripts take: --reps 1 --csv <csv_path>
        script_name = {
            "naive": "python_naive.py",
            "blocked": "python_blocked.py",
            "strassen": "python_strassen.py",
            "blas": "python_blas.py",
        }[algorithm]
        return ["python3", f"python/{script_name}", "--reps", "1", "--csv", csv_path]

    elif language == "r":
        # Assumes R scripts take positionals: <reps> <csv_path>
        script_name = {
            "naive": "r_naive.R",
            "blocked": "r_blocked.R",
            "strassen": "r_strassen.R",
            "blas": "r_blas.R",
        }[algorithm]
        return ["Rscript", f"r/{script_name}", "1", csv_path]

    else:
        raise ValueError(f"Unknown language: {language}")


# ----- Experiment scheduling -------------------------------------------------

def build_schedule(seed: int | None = None):
    """
    Build and shuffle the full list of runs.

    Each run is a tuple:
      (global_run_id, language, algorithm, replicate_within_cell)

    global_run_id is 1..120 (after shuffling order).
    """
    rng = random.Random(seed)

    runs = []
    for lang in BLOCKS:
        for algo in TREATMENTS:
            for r in range(1, REPS_PER_CELL + 1):
                runs.append((lang, algo, r))

    # Shuffle runs to randomize execution order
    rng.shuffle(runs)

    # Attach a global run index 1..len(runs)
    scheduled = []
    for idx, (lang, algo, r) in enumerate(runs, start=1):
        scheduled.append((idx, lang, algo, r))

    return scheduled


# ----- Main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default="results_experiment.csv",
        help="CSV file to which all implementations will append results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffling.",
    )
    args = parser.parse_args()

    csv_path = args.csv
    seed = args.seed

    # Ensure parent directory exists (if CSV path includes dirs)
    csv_parent = Path(csv_path).parent
    if csv_parent and not csv_parent.exists():
        csv_parent.mkdir(parents=True, exist_ok=True)

    schedule = build_schedule(seed=seed)

    print(f"Experiment schedule created with {len(schedule)} runs.")
    print(f"Results will be appended to: {csv_path}")
    print(f"Random seed: {seed}")
    print()

    for global_run_id, lang, algo, rep_in_cell in schedule:
        print(
            f"=== Global run {global_run_id:03d} | "
            f"language={lang} | algorithm={algo} | replicate={rep_in_cell} ==="
        )
        cmd = build_command(lang, algo, csv_path)
        print("Command:", " ".join(cmd))

        try:
            # Run and propagate errors if any
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: command failed with exit code {e.returncode}")
            print(f"       command was: {' '.join(cmd)}")
            # Depending on preference, you can either stop or continue:
            raise

    print("\nAll runs completed.")


if __name__ == "__main__":
    main()
