// mm_naive.cpp
#include "mm_common.hpp"
#include <sstream>

static const std::string ALGORITHM = "naive";

int main(int argc, char** argv) {
    int reps;
    std::string csv_path;
    parse_args(argc, argv, reps, csv_path);

    try {
        std::cout << "Loading matrices A_1024.csv and B_1024.csv...\n";
        Eigen::MatrixXd A = load_matrix_csv("A_1024.csv", N);
        Eigen::MatrixXd B = load_matrix_csv("B_1024.csv", N);
        Eigen::MatrixXd C(N, N);

        std::vector<double> times;
        std::vector<std::array<std::string, 5>> rows;
        times.reserve(reps);

        std::cout << "Running " << ALGORITHM << " algorithm in " << LANGUAGE
                  << " on " << N << "x" << N << " matrices (" << reps << " reps)...\n";

        for (int rep = 1; rep <= reps; ++rep) {
            C.setZero();
            auto start = std::chrono::high_resolution_clock::now();

            // Naive triple loop: i, j, k
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    double s = 0.0;
                    for (int k = 0; k < N; ++k) {
                        s += A(i, k) * B(k, j);
                    }
                    C(i, j) = s;
                }
            }

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            double t = elapsed.count();
            times.push_back(t);

            std::cout << "Rep " << rep << ": " << t << " s\n";

            // CSV row
            std::array<std::string, 5> r{
                LANGUAGE,
                ALGORITHM,
                std::to_string(N),
                std::to_string(rep),
                std::to_string(t)
            };
            rows.push_back(r);
        }

        append_results_csv(csv_path, rows);
        Stats s = compute_stats(times);
        std::cout << "\nSummary statistics:\n";
        std::cout << "  mean = " << s.mean << " s\n";
        std::cout << "  sd   = " << s.sd   << " s\n";
        std::cout << "  min  = " << s.min  << " s\n";
        std::cout << "  max  = " << s.max  << " s\n";

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
