// mm_blocked.cpp
#include "mm_common.hpp"
#include <sstream>

static const std::string ALGORITHM = "blocked";

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

            const int BLOCK_SIZE = 64;

            for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
                for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
                    for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
                        int i_max = std::min(ii + BLOCK_SIZE, N);
                        int k_max = std::min(kk + BLOCK_SIZE, N);
                        int j_max = std::min(jj + BLOCK_SIZE, N);

                        for (int i = ii; i < i_max; ++i) {
                            for (int k = kk; k < k_max; ++k) {
                                double aik = A(i, k);
                                for (int j = jj; j < j_max; ++j) {
                                    C(i, j) += aik * B(k, j);
                                }
                            }
                        }
                    }
                }
            }

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            double t = elapsed.count();
            times.push_back(t);

            std::cout << "Rep " << rep << ": " << t << " s\n";

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
