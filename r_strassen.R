#!/usr/bin/env Rscript
# mm_strassen.R
#
# Strassen's matrix multiplication for 1024x1024.
# - Recursive Strassen on square matrices
# - At leaf size, uses A %*% B (BLAS-backed)
# Usage:
#   Rscript mm_strassen.R [reps] [csv_path]

N <- 1024L
LANGUAGE <- "R"
ALGORITHM <- "strassen"
LEAF_SIZE <- 64L  # when block size <= LEAF_SIZE, use A %*% B

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  reps <- if (length(args) >= 1L) as.integer(args[[1]]) else 3L
  csv  <- if (length(args) >= 2L) args[[2]] else "results_r.csv"
  list(reps = reps, csv = csv)
}

load_matrix_csv <- function(filename, n) {
  x <- scan(filename, what = numeric(), sep = ",", quiet = TRUE)
  matrix(x, nrow = n, ncol = n, byrow = TRUE)
}

append_results_csv <- function(csv_path, rows) {
  header_needed <- !file.exists(csv_path)
  con <- file(csv_path, open = "a")
  on.exit(close(con), add = TRUE)
  if (header_needed) {
    writeLines("language,algorithm,n,rep,time_s", con)
  }
  apply(rows, 1, function(r) {
    line <- paste(r[["language"]], r[["algorithm"]], r[["n"]],
                  r[["rep"]], r[["time_s"]], sep = ",")
    writeLines(line, con)
  })
}

compute_stats <- function(times) {
  list(
    mean = mean(times),
    sd   = if (length(times) > 1L) sd(times) else NA_real_,
    min  = min(times),
    max  = max(times)
  )
}

strassen <- function(A, B) {
  n <- nrow(A)
  if (n <= LEAF_SIZE) {
    return(A %*% B)
  }
  mid <- n %/% 2L

  A11 <- A[1:mid,       1:mid      ]
  A12 <- A[1:mid,       (mid+1):n  ]
  A21 <- A[(mid+1):n,   1:mid      ]
  A22 <- A[(mid+1):n,   (mid+1):n  ]

  B11 <- B[1:mid,       1:mid      ]
  B12 <- B[1:mid,       (mid+1):n  ]
  B21 <- B[(mid+1):n,   1:mid      ]
  B22 <- B[(mid+1):n,   (mid+1):n  ]

  M1 <- strassen(A11 + A22, B11 + B22)
  M2 <- strassen(A21 + A22, B11)
  M3 <- strassen(A11,       B12 - B22)
  M4 <- strassen(A22,       B21 - B11)
  M5 <- strassen(A11 + A12, B22)
  M6 <- strassen(A21 - A11, B11 + B12)
  M7 <- strassen(A12 - A22, B21 + B22)

  C11 <- M1 + M4 - M5 + M7
  C12 <- M3 + M5
  C21 <- M2 + M4
  C22 <- M1 - M2 + M3 + M6

  C <- matrix(0, nrow = n, ncol = n)
  C[1:mid,       1:mid      ] <- C11
  C[1:mid,       (mid+1):n  ] <- C12
  C[(mid+1):n,   1:mid      ] <- C21
  C[(mid+1):n,   (mid+1):n  ] <- C22
  C
}

main <- function() {
  args <- parse_args()
  reps <- args$reps
  csv_path <- args$csv

  cat("Loading matrices A_1024.csv and B_1024.csv...\n")
  A <- load_matrix_csv("A_1024.csv", N)
  B <- load_matrix_csv("B_1024.csv", N)

  times <- numeric(reps)
  rows <- data.frame(
    language  = character(reps),
    algorithm = character(reps),
    n         = integer(reps),
    rep       = integer(reps),
    time_s    = numeric(reps),
    stringsAsFactors = FALSE
  )

  cat(sprintf(
    "Running %s algorithm in %s on %dx%d matrices (%d reps, excluding warm-up)...\n",
    ALGORITHM, LANGUAGE, N, N, reps
  ))

  # --------------------------------------------------------------
  # MEASURED REPS
  # --------------------------------------------------------------
  for (r in seq_len(reps)) {
    t0 <- proc.time()
    C <- strassen(A, B)
    elapsed <- (proc.time() - t0)[["elapsed"]]
    times[r] <- elapsed

    rows$language[r]  <- LANGUAGE
    rows$algorithm[r] <- ALGORITHM
    rows$n[r]         <- N
    rows$rep[r]       <- r
    rows$time_s[r]    <- elapsed

    cat(sprintf("Rep %d: %.4f s\n", r, elapsed))
  }

  append_results_csv(csv_path, rows)

  stats <- compute_stats(times)
  cat("\nSummary statistics:\n")
  cat(sprintf("  mean = %.4f s\n", stats$mean))
  cat(sprintf("  sd   = %.4f s\n", stats$sd))
  cat(sprintf("  min  = %.4f s\n", stats$min))
  cat(sprintf("  max  = %.4f s\n", stats$max))
}

if (identical(environment(), globalenv())) {
  main()
}
