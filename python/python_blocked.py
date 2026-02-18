#!/usr/bin/env python3
"""
mm_blocked.py

Cache-friendly (loop-blocked) triple-loop matrix multiplication for 1024x1024.
Same math as naive but loops blocked to improve cache behaviour.
"""

import argparse
import csv
import os
import statistics
import time

import numpy as np

N = 1024
LANGUAGE = "Python"
ALGORITHM = "blocked"

BLOCK_SIZE = 64  # or 32; you can tweak


def matmul_blocked(A, B):
    n = len(A)
    m = len(B[0])
    p = len(A[0])
    C = [[0.0] * m for _ in range(n)]

    for ii in range(0, n, BLOCK_SIZE):
        for kk in range(0, p, BLOCK_SIZE):
            for jj in range(0, m, BLOCK_SIZE):
                for i in range(ii, min(ii + BLOCK_SIZE, n)):
                    for k in range(kk, min(kk + BLOCK_SIZE, p)):
                        aik = A[i][k]
                        for j in range(jj, min(jj + BLOCK_SIZE, m)):
                            C[i][j] += aik * B[k][j]
    return C


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

    # Load matrices
    A_np = np.load("A_1024.npy")
    B_np = np.load("B_1024.npy")

    # Convert to lists for pure Python loops
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
        C = matmul_blocked(A, B)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        rows.append([LANGUAGE, ALGORITHM, N, rep, elapsed])
        print(f"Rep {rep}: {elapsed:.4f} s")

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
