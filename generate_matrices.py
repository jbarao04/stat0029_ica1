#!/usr/bin/env python3
"""
generate_matrices_1024.py

Generate reproducible random matrices A and B of size 1024x1024.
Saved as both .npy (for Python) and .csv (for R/C++).
"""

import numpy as np

N = 1024
SEED = 2025

def main():
    np.random.seed(SEED)
    A = np.random.randn(N, N)
    B = np.random.randn(N, N)

    # Save NumPy binaries (fast to load in Python)
    np.save("A_1024.npy", A)
    np.save("B_1024.npy", B)

    # Also save CSV for other languages if needed
    np.savetxt("A_1024.csv", A, delimiter=",")
    np.savetxt("B_1024.csv", B, delimiter=",")

    print(f"Generated A_1024 and B_1024 with seed={SEED}, size={N}x{N}.")

if __name__ == "__main__":
    main()
