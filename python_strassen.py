#!/usr/bin/env python3
"""
mm_strassen.py

Strassen's matrix multiplication for 1024x1024 matrices.
- Uses recursive Strassen for large blocks
- Uses NumPy's @ (BLAS) at leaves for performance
- Still conceptually distinct from a single A @ B at top level
"""

import argparse
import csv
import os
import statistics
import time

import numpy as np

N = 1024
LANGUAGE = "Python"
ALGORITHM = "strassen"

LEAF_SIZE = 64  # when block size <= LEAF_SIZE, use direct A @ B


def strassen(A, B):
    """
    Strassen multiplication of two square NumPy arrays A and B of the same size.
    n must be a power of two (we assume 1024), or you can pad if needed.
    """
    n = A.shape[0]
    if n <= LEAF_SIZE:
        # Leaf computation uses BLAS-backed @
        return A @ B

    # Split into quadrants
    mid = n // 2
    A11 = A[:mid, :mid]
    A12 = A[:mid, mid:]
    A21 = A[mid:, :mid]
    A22 = A[mid:, mid:]

    B11 = B[:mid, :mid]
    B12 = B[:mid, mid:]
    B21 = B[mid:, :mid]
    B22 = B[mid:, mid:]

    # Strassen's 7 products
    M1 = strassen(A11 + A22, B11 + B22)
    M2 = strassen(A21 + A22, B11)
    M3 = strassen(A11, B12 - B22)
    M4 = strassen(A22, B21 - B11)
    M5 = strassen(A11 + A12, B22)
    M6 = strassen(A21 - A11, B11 + B12)
    M7 = strassen(A12 - A22, B21 + B22)

    # Combine subresults into C
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6

    C = np.empty_like(A)
    C[:mid, :mid] = C11
    C[:mid, mid:] = C12
    C[mid:, :mid] = C21
    C[mid:, mid:] = C22

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
        C = strassen(A, B)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        rows.append([LANGUAGE, ALGORITHM, N, rep, elapsed])
        print(f"Rep {rep}: {elapsed:.4f} s")

        # Optional sanity check (commented)
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
