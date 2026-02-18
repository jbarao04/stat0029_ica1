#!/usr/bin/env python3
"""
mm_naive.py

Baseline naive triple-loop matrix multiplication for 1024x1024 matrices.
- Loads A_1024.npy and B_1024.npy
- Converts to Python lists of lists
- Runs naive matmul multiple times
- Records per-run times and summary statistics
- Appends results to results_python.csv
"""

import argparse
import csv
import os
import statistics
import time

import numpy as np

N = 1024
LANGUAGE = "Python"
ALGORITHM = "naive"


def matmul_naive(A, B):
    """
    A: list of lists, size n x p
    B: list of lists, size p x m
    Returns C = A * B as list of lists.
    """
    n = len(A)
    p = len(A[0])
    m = len(B[0])
    C = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            s = 0.0
            for k in range(p):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C


def append_results_csv(csv_path, rows):
    """Append rows to CSV; create header if file does not exist."""
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

    # Load matrices
    A_np = np.load("A_1024.npy")
    B_np = np.load("B_1024.npy")

    # Convert to Python lists for pure Python loops
    A = A_np.tolist()
    B = B_np.tolist()

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
        C = matmul_naive(A, B)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        rows.append([LANGUAGE, ALGORITHM, N, rep, elapsed])
        print(f"Rep {rep}: {elapsed:.4f} s")

        # Optionally, sanity-check a single element to ensure it depends on data
        # print(C[0][0])  # commented to avoid clutter

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
