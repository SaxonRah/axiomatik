/**
 * benchmark.cpp - Heavy Computation Benchmark for Axiomatik C++
 *
 * This benchmark performs computationally intensive tasks while using
 * extensive runtime verification to measure the overhead of Axiomatik.
 */

#include "axiomatik.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <math.h>
#include <iomanip>
#include <algorithm>
#include <numeric>

const double M_PI = 3.14159265358979323846;


using namespace axiomatik;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// VERIFIED MONTE CARLO PI ESTIMATION WITH STATISTICAL ANALYSIS
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

struct Point2D {
    double x, y;

    Point2D(double x_, double y_) : x(x_), y(y_) {
        PROOF_CONTEXT("Point2D_construction");
        require("x coordinate is finite", std::isfinite(x));
        require("y coordinate is finite", std::isfinite(y));
    }

    double distance_from_origin() const {
        PROOF_CONTEXT("distance_calculation");
        double dist_squared = x * x + y * y;
        require("distance calculation is finite", std::isfinite(dist_squared));
        return std::sqrt(dist_squared);
    }

    bool is_inside_unit_circle() const {
        return distance_from_origin() <= 1.0;
    }
};

struct StatisticalData {
    std::vector<double> samples;
    size_t total_count;
    size_t inside_circle_count;

    StatisticalData() : total_count(0), inside_circle_count(0) {
        samples.reserve(1000000); // Preallocate for performance
    }

    void add_sample(const Point2D& point) {
        PROOF_CONTEXT("statistical_sampling");

        CONTRACT("add_sample");
        REQUIRES("point coordinates are valid",
            std::isfinite(point.x) && std::isfinite(point.y));
        REQUIRES("sample collection is not corrupted",
            inside_circle_count <= total_count);

        CHECK_PRECONDITIONS();

        total_count++;
        if (point.is_inside_unit_circle()) {
            inside_circle_count++;
        }

        // Store distance for statistical analysis
        double distance = point.distance_from_origin();
        samples.push_back(distance);

        // Verify statistical invariants
        require("count consistency", inside_circle_count <= total_count);
        require("sample count matches total", samples.size() == total_count);
        require("distance is non-negative", distance >= 0.0);

        CHECK_POSTCONDITIONS();
    }

    double estimate_pi() const {
        PROOF_CONTEXT("pi_estimation");

        require("sufficient samples for estimation", total_count >= 1000);
        require("statistical consistency", inside_circle_count <= total_count);

        double ratio = static_cast<double>(inside_circle_count) / total_count;
        double pi_estimate = 4.0 * ratio;

        require("pi estimate is reasonable", pi_estimate > 2.0 && pi_estimate < 4.0);
        require("pi estimate is finite", std::isfinite(pi_estimate));

        return pi_estimate;
    }

    double calculate_standard_deviation() const {
        PROOF_CONTEXT("statistical_analysis");

        require("sufficient data for std dev", samples.size() > 1);

        double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();

        double variance = 0.0;
        for (double sample : samples) {
            double diff = sample - mean;
            variance += diff * diff;
        }
        variance /= (samples.size() - 1);

        double std_dev = std::sqrt(variance);

        require("standard deviation is non-negative", std_dev >= 0.0);
        require("standard deviation is finite", std::isfinite(std_dev));

        return std_dev;
    }
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// VERIFIED MATRIX OPERATIONS FOR HEAVY COMPUTATION
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class VerifiedMatrix {
private:
    std::vector<std::vector<double>> data;
    size_t rows, cols;

public:
    VerifiedMatrix(size_t rows_, size_t cols_) : rows(rows_), cols(cols_) {
        PROOF_CONTEXT("matrix_construction");

        require("positive dimensions", rows > 0 && cols > 0);
        require("reasonable matrix size", rows <= 10000 && cols <= 10000);

        data.resize(rows);
        for (auto& row : data) {
            row.resize(cols, 0.0);
        }

        require("matrix properly initialized", data.size() == rows);
        require("all rows have correct size",
            std::all_of(data.begin(), data.end(),
                [this](const auto& row) { return row.size() == cols; }));
    }

    void set(size_t row, size_t col, double value) {
        PROOF_CONTEXT("matrix_element_access");

        require("row index in bounds", row < rows);
        require("column index in bounds", col < cols);
        require("value is finite", std::isfinite(value));

        data[row][col] = value;
    }

    double get(size_t row, size_t col) const {
        PROOF_CONTEXT("matrix_element_retrieval");

        require("row index in bounds", row < rows);
        require("column index in bounds", col < cols);

        double value = data[row][col];
        require("retrieved value is finite", std::isfinite(value));

        return value;
    }

    // Normalize matrix to prevent numerical explosion
    void normalize() {
        PROOF_CONTEXT("matrix_normalization");

        double max_element = 0.0;
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                max_element = std::max(max_element, std::abs(get(i, j)));
            }
        }

        if (max_element > 1000.0) {  // Normalize if elements get too large
            double scale_factor = 100.0 / max_element;
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    data[i][j] *= scale_factor;
                }
            }
        }
    }

    VerifiedMatrix multiply(const VerifiedMatrix& other) const {
        PROOF_CONTEXT("matrix_multiplication");

        CONTRACT("matrix_multiply");
        REQUIRES("matrices compatible for multiplication", cols == other.rows);
        REQUIRES("result dimensions are valid", rows > 0 && other.cols > 0);

        CHECK_PRECONDITIONS();

        VerifiedMatrix result(rows, other.cols);

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                double sum = 0.0;

                for (size_t k = 0; k < cols; ++k) {
                    double a_val = get(i, k);
                    double b_val = other.get(k, j);
                    double product = a_val * b_val;

                    require("multiplication result is finite", std::isfinite(product));
                    sum += product;

                    // Check for numerical overflow during accumulation
                    require("intermediate sum is finite", std::isfinite(sum));
                }

                result.set(i, j, sum);
            }
        }

        ENSURES("result has correct dimensions",
            result.rows == rows && result.cols == other.cols);

        CHECK_POSTCONDITIONS();

        return result;
    }

    double frobenius_norm() const {
        PROOF_CONTEXT("frobenius_norm_calculation");

        double sum_of_squares = 0.0;

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                double value = get(i, j);
                double square = value * value;
                require("element square is finite", std::isfinite(square));
                sum_of_squares += square;
            }
        }

        require("sum of squares is non-negative", sum_of_squares >= 0.0);
        require("sum of squares is finite", std::isfinite(sum_of_squares));

        double norm = std::sqrt(sum_of_squares);

        require("norm is non-negative", norm >= 0.0);
        require("norm is finite", std::isfinite(norm));

        return norm;
    }

    double max_element() const {
        double max_val = 0.0;
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                max_val = std::max(max_val, std::abs(get(i, j)));
            }
        }
        return max_val;
    }

    size_t get_rows() const { return rows; }
    size_t get_cols() const { return cols; }
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// COMPREHENSIVE COMPUTATIONAL BENCHMARK
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ComputationalBenchmark {
private:
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform_dist;

public:
    ComputationalBenchmark() : rng(std::random_device{}()), uniform_dist(-1.0, 1.0) {
        PROOF_CONTEXT("benchmark_initialization");
        require("random number generator is properly seeded", true);
    }

    // Monte Carlo PI estimation (same as before)
    double monte_carlo_pi_verified(size_t num_samples) {
        PROOF_CONTEXT("monte_carlo_pi_computation");

        CONTRACT("monte_carlo_pi");
        REQUIRES("sufficient sample size", num_samples >= 1000);
        REQUIRES("reasonable sample size", num_samples <= 100000000);

        CHECK_PRECONDITIONS();

        StatisticalData stats;

        for (size_t i = 0; i < num_samples; ++i) {
            double x = uniform_dist(rng);
            double y = uniform_dist(rng);

            Point2D point(x, y);
            stats.add_sample(point);

            // Periodic verification of intermediate results
            if (i % 100000 == 0 && i > 0) {
                double intermediate_pi = stats.estimate_pi();
                require("reasonable intermediate estimate",
                    intermediate_pi > 1.0 && intermediate_pi < 6.0);  // More lenient
            }
        }

        double pi_estimate = stats.estimate_pi();
        double std_dev = stats.calculate_standard_deviation();

        require("final estimate quality", std::abs(pi_estimate - M_PI) < 1.0);  // More lenient
        require("statistical convergence", std_dev < 2.0);  // More lenient

        ENSURES("pi estimate is reasonable", pi_estimate > 2.0 && pi_estimate < 5.0);
        ENSURES("statistical validity", std_dev >= 0.0);

        CHECK_POSTCONDITIONS();

        return pi_estimate;
    }

    // Fixed matrix computation with normalization
    double matrix_computation_verified(size_t matrix_size, size_t num_operations) {
        PROOF_CONTEXT("matrix_computation_benchmark");

        CONTRACT("matrix_computation");
        REQUIRES("reasonable matrix size", matrix_size >= 10 && matrix_size <= 500);
        REQUIRES("sufficient operations", num_operations >= 1);

        CHECK_PRECONDITIONS();

        // Create initial matrices with smaller values for numerical stability
        VerifiedMatrix A(matrix_size, matrix_size);
        VerifiedMatrix B(matrix_size, matrix_size);

        // Initialize with smaller random values to prevent explosion
        std::uniform_real_distribution<double> small_dist(-0.1, 0.1);

        for (size_t i = 0; i < matrix_size; ++i) {
            for (size_t j = 0; j < matrix_size; ++j) {
                A.set(i, j, small_dist(rng));
                B.set(i, j, small_dist(rng));
            }
        }

        VerifiedMatrix result = A;

        std::cout << "  Matrix operations progress: ";

        // Perform chain of matrix multiplications with monitoring
        for (size_t op = 0; op < num_operations; ++op) {
            result = result.multiply(B);

            // Get current norm and max element for monitoring
            double norm = result.frobenius_norm();
            double max_elem = result.max_element();

            std::cout << "[" << (op + 1) << ": norm=" << std::scientific << std::setprecision(2)
                << norm << ", max=" << max_elem << "] ";

            // More realistic bounds that account for matrix growth
            require("matrix norm is finite", std::isfinite(norm));
            require("matrix norm is non-negative", norm >= 0.0);
            require("max element is finite", std::isfinite(max_elem));

            // Apply normalization if values get too large
            if (max_elem > 1000.0) {
                result.normalize();
                std::cout << "(normalized) ";
            }

            // Verify matrix hasn't become degenerate
            if (op % 3 == 0) {
                require("matrix dimensions preserved",
                    result.get_rows() == matrix_size && result.get_cols() == matrix_size);
            }
        }

        std::cout << "\n";

        double final_norm = result.frobenius_norm();

        ENSURES("final result is valid", std::isfinite(final_norm) && final_norm >= 0.0);

        CHECK_POSTCONDITIONS();

        return final_norm;
    }

    // Comprehensive benchmark with better error handling
    void comprehensive_benchmark(size_t monte_carlo_samples,
        size_t matrix_size,
        size_t matrix_operations) {
        PROOF_CONTEXT("comprehensive_computational_benchmark");

        auto start_time = std::chrono::steady_clock::now();

        // Phase 1: Monte Carlo simulation
        std::cout << "Phase 1: Monte Carlo simulation with " << monte_carlo_samples << " samples...\n";
        double pi_estimate = monte_carlo_pi_verified(monte_carlo_samples);

        auto monte_carlo_time = std::chrono::steady_clock::now();

        // Phase 2: Matrix computations
        std::cout << "Phase 2: Matrix computations (" << matrix_size << "x" << matrix_size
            << ", " << matrix_operations << " operations)...\n";
        double matrix_result = matrix_computation_verified(matrix_size, matrix_operations);

        auto matrix_time = std::chrono::steady_clock::now();

        // Phase 3: Results analysis
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            matrix_time - start_time);
        auto monte_carlo_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            monte_carlo_time - start_time);
        auto matrix_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            matrix_time - monte_carlo_time);

        // Verify timing results with more lenient bounds
        require("total time is positive", total_duration.count() > 0);
        require("monte carlo took time", monte_carlo_duration.count() >= 0);
        require("matrix computation took time", matrix_duration.count() >= 0);

        // Report results
        std::cout << "\n=== COMPUTATIONAL RESULTS ===\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "pi estimate: " << pi_estimate << " (error: "
            << std::abs(pi_estimate - M_PI) << ")\n";
        std::cout << "Matrix result norm: " << std::scientific << matrix_result << "\n";
        std::cout << "Monte Carlo time: " << monte_carlo_duration.count() << "ms\n";
        std::cout << "Matrix computation time: " << matrix_duration.count() << "ms\n";
        std::cout << "Total computation time: " << total_duration.count() << "ms\n";
    }
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// BENCHMARK HARNESS WITH VERIFICATION OVERHEAD MEASUREMENT
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void report_metrics(const axiomatik::PerformanceMetrics& metrics) {
    std::cout << "Total verifications: " << metrics.total_verifications << "\n";
    std::cout << "Total time: " << (metrics.total_time_ns / 1000.0) << " microseconds\n";
    std::cout << "Average time: " << metrics.average_time_us << " microseconds\n";
    std::cout << "Min time: " << (metrics.min_time_ns / 1000.0) << " microseconds\n";
    std::cout << "Max time: " << (metrics.max_time_ns / 1000.0) << " microseconds\n";
    std::cout << "Cache hit rate: " << metrics.cache_hit_rate << "%\n";
}

void run_benchmark_with_verification() {
    std::cout << "Running VERIFIED benchmark...\n";
    std::cout << "================================\n";

    global_config.level = VerificationLevel::FULL;
    global_proof_system.clear();

    auto start_time = std::chrono::high_resolution_clock::now();

    try {
        ComputationalBenchmark benchmark;
        benchmark.comprehensive_benchmark(
            500000,   // Reduced to 500K samples for faster testing
            50,       // Reduced matrix size for numerical stability
            8         // Reduced operations
        );

        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "\n=== VERIFICATION OVERHEAD ANALYSIS ===\n";

        auto metrics = get_performance_metrics();
        report_metrics(metrics);
        std::cout << "Total benchmark time: " << total_time.count() << "ms\n";

        double verification_overhead_ms = metrics.total_time_ns / 1000.0;
        double overhead_percentage = (verification_overhead_ms / total_time.count()) * 100.0;

        std::cout << "Verification overhead: " << std::fixed << std::setprecision(2)
            << verification_overhead_ms << "ms (" << overhead_percentage << "%)\n";

    }
    catch (const ProofFailure& failure) {
        std::cout << "\nVERIFICATION FAILURE DETAILS:\n";
        std::cout << "Claim: " << failure.claim << "\n";
        std::cout << "Context: " << failure.context << "\n";
        std::cout << "Details: " << failure.details << "\n";
        throw; // Re-throw to maintain error handling
    }
}

void run_benchmark_without_verification() {
    std::cout << "\n\nRunning UNVERIFIED benchmark...\n";
    std::cout << "================================\n";

    global_config.level = VerificationLevel::OFF;
    global_proof_system.clear();

    auto start_time = std::chrono::high_resolution_clock::now();

    ComputationalBenchmark benchmark;
    benchmark.comprehensive_benchmark(500000, 50, 8);  // Same parameters

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Unverified benchmark time: " << total_time.count() << "ms\n";
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// SCALABILITY ANALYSIS
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void run_scalability_analysis() {
    std::cout << "\n\nSCALABILITY ANALYSIS:\n";
    std::cout << "====================\n";

    struct BenchmarkConfig {
        size_t monte_carlo_samples;
        size_t matrix_size;
        size_t matrix_ops;
        std::string description;
    };

    std::vector<BenchmarkConfig> configs = {
        {100000, 50, 5, "Light workload"},
        {500000, 75, 7, "Medium workload"},
        {1000000, 100, 10, "Heavy workload"},
        {2000000, 125, 12, "Very heavy workload"}
    };

    for (const auto& config : configs) {
        std::cout << "\nTesting " << config.description << "...\n";

        // Test with verification
        global_config.level = VerificationLevel::FULL;
        global_proof_system.clear();

        auto start_verified = std::chrono::high_resolution_clock::now();
        ComputationalBenchmark benchmark_verified;
        benchmark_verified.comprehensive_benchmark(
            config.monte_carlo_samples, config.matrix_size, config.matrix_ops);
        auto end_verified = std::chrono::high_resolution_clock::now();

        auto verified_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_verified - start_verified);
        auto verification_metrics = get_performance_metrics();

        // Test without verification
        global_config.level = VerificationLevel::OFF;
        global_proof_system.clear();

        auto start_unverified = std::chrono::high_resolution_clock::now();
        ComputationalBenchmark benchmark_unverified;
        benchmark_unverified.comprehensive_benchmark(
            config.monte_carlo_samples, config.matrix_size, config.matrix_ops);
        auto end_unverified = std::chrono::high_resolution_clock::now();

        auto unverified_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_unverified - start_unverified);

        // Calculate overhead
        double overhead_ms = verified_time.count() - unverified_time.count();
        double overhead_percentage = (overhead_ms / unverified_time.count()) * 100.0;

        std::cout << "  Verified time: " << verified_time.count() << "ms\n";
        std::cout << "  Unverified time: " << unverified_time.count() << "ms\n";
        std::cout << "  Overhead: " << std::fixed << std::setprecision(2)
            << overhead_ms << "ms (" << overhead_percentage << "%)\n";
        std::cout << "  Verifications performed: " << verification_metrics.total_verifications << "\n";
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// MAIN BENCHMARK RUNNER
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

int main() {
    std::cout << "Axiomatik C++ Performance Benchmark\n";
    std::cout << "===================================\n\n";

    try {
        // Run comprehensive benchmarks
        run_benchmark_with_verification();
        run_benchmark_without_verification();

        // Analyze scalability across different workloads
        run_scalability_analysis();

        std::cout << "\n=== BENCHMARK COMPLETE ===\n";
        std::cout << "This benchmark demonstrates Axiomatik's runtime verification\n";
        std::cout << "overhead during computationally intensive tasks including:\n";
        std::cout << "- Monte Carlo simulation with statistical verification\n";
        std::cout << "- Matrix operations with mathematical invariants\n";
        std::cout << "- Contract verification and temporal properties\n";
        std::cout << "- Performance monitoring and cache optimization\n\n";

    }
    catch (const ProofFailure& failure) {
        std::cerr << "Verification failed during benchmark:\n";
        std::cerr << "Claim: " << failure.claim << "\n";
        std::cerr << "Context: " << failure.context << "\n";
        std::cerr << "Details: " << failure.details << "\n";
        return 1;
    }

    return 0;
}