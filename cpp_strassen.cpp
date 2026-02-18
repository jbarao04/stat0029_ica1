// mm_strassen.cpp
#include "mm_common.hpp"
#include <sstream>

static const std::string ALGORITHM = "strassen";
static const int LEAF_SIZE = 64;  // at or below this, use Eigen's A*B

Eigen::MatrixXd strassen(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
    int n = A.rows();
    if (n <= LEAF_SIZE) {
        // Leaf: use Eigen's highly optimized multiplication
        return A * B;
    }

    int mid = n / 2;
    // Quadrants
    Eigen::MatrixXd A11 = A.topLeftCorner(mid, mid);
    Eigen::MatrixXd A12 = A.topRightCorner(mid, mid);
    Eigen::MatrixXd A21 = A.bottomLeftCorner(mid, mid);
    Eigen::MatrixXd A22 = A.bottomRightCorner(mid, mid);

    Eigen::MatrixXd B11 = B.topLeftCorner(mid, mid);
    Eigen::MatrixXd B12 = B.topRightCorner(mid, mid);
    Eigen::MatrixXd B21 = B.bottomLeftCorner(mid, mid);
    Eigen::MatrixXd B22 = B.bottomRightCorner(mid, mid);

    // Strassen's 7 products
    Eigen::MatrixXd M1 = strassen(A11 + A22, B11 + B22);
    Eigen::MatrixXd M2 = strassen(A21 + A22, B11);
    Eigen::MatrixXd M3 = strassen(A11,       B12 - B22);
    Eigen::MatrixXd M4 = strassen(A22,       B21 - B11);
    Eigen::MatrixXd M5 = strassen(A11 + A12, B22);
    Eigen::MatrixXd M6 = strassen(A21 - A11, B11 + B12);
    Eigen::MatrixXd M7 = strassen(A12 - A22, B21 + B22);

    Eigen::MatrixXd C11 = M1 + M4 - M5 + M7;
    Eigen::MatrixXd C12 = M3 + M5;
    Eigen::MatrixXd C21 = M2 + M4;
    Eigen::MatrixXd C22 = M1 - M2 + M3 + M6;

    Eigen::MatrixXd C(n, n);
    C.topLeftCorner(mid, mid)     = C11;
    C.topRightCorner(mid, mid)    = C12;
    C.bottomLeftCorner(mid, mid)  = C21;
    C.bottomRightCorner(mid, mid) = C22;

    return C;
}

int main(int argc, char** argv) {
    int reps;
    std::string csv_path;
    parse_args(argc, argv, reps, csv_path);

    try {
        std::cout << "Loading matrices A_1024.csv and B_1024.csv...\n";
        Eigen::MatrixXd A = load_matrix_csv("A_1024.csv", N);
        Eigen::MatrixXd B = load_matrix_csv("B_1024.csv", N);

        std::vector<double> times;
        std::vector<std::array<std::string, 5>> rows;
        times.reserve(reps);

        std::cout << "Running " << ALGORITHM << " algorithm in " << LANGUAGE
                  << " on " << N << "x" << N << " matrices (" << reps << " reps)...\n";

        for (int rep = 1; rep <= reps; ++rep) {
            auto start = std::chrono::high_resolution_clock::now();
            Eigen::MatrixXd C = strassen(A, B);
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
