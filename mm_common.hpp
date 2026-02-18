// mm_common.hpp
#pragma once

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

static const int N = 1024;
static const std::string LANGUAGE = "C++";

// Load an n x n matrix from CSV (comma-separated)
inline Eigen::MatrixXd load_matrix_csv(const std::string& filename, int n) {
    Eigen::MatrixXd M(n, n);
    std::ifstream in(filename);
    if (!in) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    std::string line;
    int i = 0;
    while (std::getline(in, line) && i < n) {
        std::stringstream ss(line);
        std::string cell;
        int j = 0;
        while (std::getline(ss, cell, ',') && j < n) {
            M(i, j) = std::stod(cell);
            ++j;
        }
        ++i;
    }
    if (i != n) {
        throw std::runtime_error("File " + filename + " does not have " + std::to_string(n) + " rows");
    }
    return M;
}

// Append rows to CSV; create header if file doesn't exist
inline void append_results_csv(
    const std::string& csv_path,
    const std::vector<std::array<std::string, 5>>& rows
) {
    bool file_exists = false;
    {
        std::ifstream f(csv_path);
        file_exists = f.good();
    }

    std::ofstream out(csv_path, std::ios::app);
    if (!out) {
        throw std::runtime_error("Cannot open CSV for writing: " + csv_path);
    }

    if (!file_exists) {
        out << "language,algorithm,n,rep,time_s\n";
    }
    for (const auto& r : rows) {
        out << r[0] << "," << r[1] << "," << r[2] << "," << r[3] << "," << r[4] << "\n";
    }
}

// Compute mean, sd, min, max of vector<double>
struct Stats {
    double mean;
    double sd;
    double min;
    double max;
};

inline Stats compute_stats(const std::vector<double>& v) {
    Stats s;
    if (v.empty()) {
        s.mean = s.sd = s.min = s.max = std::numeric_limits<double>::quiet_NaN();
        return s;
    }
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    s.mean = sum / v.size();
    s.min = *std::min_element(v.begin(), v.end());
    s.max = *std::max_element(v.begin(), v.end());
    if (v.size() > 1) {
        double accum = 0.0;
        for (double x : v) {
            double diff = x - s.mean;
            accum += diff * diff;
        }
        s.sd = std::sqrt(accum / (v.size() - 1));
    } else {
        s.sd = std::numeric_limits<double>::quiet_NaN();
    }
    return s;
}

// Simple helper to parse reps / csv from argv
inline void parse_args(int argc, char** argv, int& reps, std::string& csv) {
    reps = 5;
    csv = "results_cpp.csv";
    if (argc >= 2) {
        reps = std::stoi(argv[1]);
    }
    if (argc >= 3) {
        csv = argv[2];
    }
}
