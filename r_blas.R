#!/usr/bin/env Rscript
# mm_blas.R
#
# State-of-the-art practical matrix multiplication for 1024x1024:
# Direct BLAS-backed call: C <- A %*% B
# Usage:
#   Rscript mm_blas.R [reps] [csv_path]

N <- 1024L
LANGUAGE <- "R"
ALGORITHM <- "blas"

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  reps <- if (length(args) >= 1L) as.integer(args[[1]]) else 5L
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
    C <- A %*% B
    elapsed <- (proc.time() - t0)[["elapsed"]]
    times[r] <- elapsed

    rows$language[r]  <- LANGUAGE
    rows$algorithm[r] <- ALGORITHM
    rows$n[r]         <- N
    rows$rep[r]       <- r
    rows$time_s[r]    <- elapsed

    cat(sprintf("Rep %d: %.4f s\n", r, elapsed))
    # Optional: cat("C[1,1] =", C[1, 1], "\n")
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
