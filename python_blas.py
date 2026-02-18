#!/usr/bin/env python3
"""
mm_blas.py

State-of-the-art practical matrix multiplication for 1024x1024:
Direct NumPy/BLAS call C = A @ B.
"""

import argparse
import csv
import os
import statistics
import time

import numpy as np

N = 1024
LANGUAGE = "Python"
ALGORITHM = "blas"  # or "numpy_blas"


def append_results_csv(csv_path, rows):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["language", "algorithm", "n", "rep", "time_s"])
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reps",
        type=int,
        default=5,
        help="Number of repetitions (default: 5)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="results_python.csv",
        help="CSV file to append results to",
    )
    args = parser.parse_args()

    # Load matrices as NumPy arrays
    A = np.load("A_1024.npy")
    B = np.load("B_1024.npy")

    times = []
    rows = []

    print(
        f"Running {ALGORITHM} algorithm in {LANGUAGE} on {N}x{N} matrices "
        f"({args.reps} reps, excluding warm-up)..."
    )

    # ----------------------------------------------------------------------
    # MEASURED REPS
    # ----------------------------------------------------------------------
    for rep in range(1, args.reps + 1):
        start = time.perf_counter()
        C = A @ B
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        rows.append([LANGUAGE, ALGORITHM, N, rep, elapsed])
        print(f"Rep {rep}: {elapsed:.4f} s")

        # Optional check
        # print(C[0, 0])

    append_results_csv(args.csv, rows)

    mean_t = statistics.mean(times)
    sd_t = statistics.stdev(times) if len(times) > 1 else float("nan")
    print("\nSummary statistics:")
    print(f"  mean   = {mean_t:.4f} s")
    print(f"  std    = {sd_t:.4f} s")
    print(f"  min    = {min(times):.4f} s")
    print(f"  max    = {max(times):.4f} s")


if __name__ == "__main__":
    main()
