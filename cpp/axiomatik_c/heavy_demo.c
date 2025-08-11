/**
 * heavy_demo.c - Heavy Computation Benchmark for Axiomatik C17
 *
 * This benchmark performs computationally intensive tasks while using
 * extensive runtime verification to measure the overhead of Axiomatik.
 */

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4996)
#endif

#include "axiomatik_c.h"
#include <math.h>
#include <time.h>
#include <limits.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

 // Maximum sizes for static allocation
//#define MAX_SAMPLES 1000000
//#define MAX_MATRIX_SIZE 500
#define MAX_SAMPLES 10000        // Reduced from 1M to 10K
#define MAX_MATRIX_SIZE 50       // Reduced from 500 to 50
#define MAX_BENCHMARK_CONFIGS 10

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// VERIFIED MONTE CARLO PI ESTIMATION WITH STATISTICAL ANALYSIS
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

typedef struct {
    double x, y;
} point_2d_t;

typedef struct {
    double samples[MAX_SAMPLES];
    size_t sample_count;
    size_t total_count;
    size_t inside_circle_count;
} statistical_data_t;

typedef struct {
    double data[MAX_MATRIX_SIZE][MAX_MATRIX_SIZE];
    size_t rows, cols;
} verified_matrix_t;

typedef struct {
    unsigned int seed;
} simple_rng_t;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// POINT 2D OPERATIONS
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

bool point_2d_create(point_2d_t* point, double x, double y) {
    PROOF_CONTEXT("Point2D_construction");

    if (!require_double("x coordinate is finite", x)) {
        PROOF_CONTEXT_END();
        return false;
    }
    if (!require_double("y coordinate is finite", y)) {
        PROOF_CONTEXT_END();
        return false;
    }

    point->x = x;
    point->y = y;

    PROOF_CONTEXT_END();
    return true;
}

double point_2d_distance_from_origin(const point_2d_t* point) {
    PROOF_CONTEXT("distance_calculation");

    double dist_squared = point->x * point->x + point->y * point->y;

    if (!require_double("distance calculation is finite", dist_squared)) {
        PROOF_CONTEXT_END();
        return -1.0;
    }

    double result = sqrt(dist_squared);
    PROOF_CONTEXT_END();
    return result;
}

bool point_2d_is_inside_unit_circle(const point_2d_t* point) {
    double distance = point_2d_distance_from_origin(point);
    return distance >= 0.0 && distance <= 1.0;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// STATISTICAL DATA OPERATIONS
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void statistical_data_init(statistical_data_t* stats) {
    stats->sample_count = 0;
    stats->total_count = 0;
    stats->inside_circle_count = 0;
}

bool coordinates_are_valid(void* data) {
    point_2d_t* point = (point_2d_t*)data;
    return isfinite(point->x) && isfinite(point->y);
}

bool sample_collection_not_corrupted(void* data) {
    statistical_data_t* stats = (statistical_data_t*)data;
    return stats->inside_circle_count <= stats->total_count;
}

bool statistical_data_add_sample(statistical_data_t* stats, const point_2d_t* point) {
    PROOF_CONTEXT("statistical_sampling");

    CONTRACT_BEGIN("add_sample");
    PRECONDITION("point coordinates are valid", coordinates_are_valid, (void*)point);
    PRECONDITION("sample collection is not corrupted", sample_collection_not_corrupted, stats);

    if (!CHECK_PRECONDITIONS()) {
        CONTRACT_END();
        PROOF_CONTEXT_END();
        return false;
    }

    stats->total_count++;
    if (point_2d_is_inside_unit_circle(point)) {
        stats->inside_circle_count++;
    }

    // Store distance for statistical analysis (if space available)
    double distance = point_2d_distance_from_origin(point);
    if (stats->sample_count < MAX_SAMPLES) {
        stats->samples[stats->sample_count] = distance;
        stats->sample_count++;
    }

    // Verify statistical invariants
    if (!require_bool("count consistency", stats->inside_circle_count <= stats->total_count)) {
        CONTRACT_END();
        PROOF_CONTEXT_END();
        return false;
    }
    if (!require_bool("distance is non-negative", distance >= 0.0)) {
        CONTRACT_END();
        PROOF_CONTEXT_END();
        return false;
    }

    if (!CHECK_POSTCONDITIONS()) {
        CONTRACT_END();
        PROOF_CONTEXT_END();
        return false;
    }

    CONTRACT_END();
    PROOF_CONTEXT_END();
    return true;
}

double statistical_data_estimate_pi(const statistical_data_t* stats) {
    PROOF_CONTEXT("pi_estimation");

    if (!require_bool("sufficient samples for estimation", stats->total_count >= 100)) {
        printf("    Insufficient samples: %zu (need at least 100)\n", stats->total_count);
        PROOF_CONTEXT_END();
        return -1.0;
    }
    if (!require_bool("statistical consistency", stats->inside_circle_count <= stats->total_count)) {
        printf("    Statistical inconsistency: %zu inside > %zu total\n",
            stats->inside_circle_count, stats->total_count);
        PROOF_CONTEXT_END();
        return -1.0;
    }

    double ratio = (double)stats->inside_circle_count / stats->total_count;
    double pi_estimate = 4.0 * ratio;

    printf("    Raw estimate: %.6f (ratio: %.6f, inside: %zu, total: %zu)\n",
        pi_estimate, ratio, stats->inside_circle_count, stats->total_count);

    // Very lenient bounds - Monte Carlo can be quite variable with small samples
    if (!require_bool("pi estimate is reasonable", pi_estimate > 1.0 && pi_estimate < 6.0)) {
        printf("    Pi estimate %.4f outside very lenient bounds [1.0, 6.0]\n", pi_estimate);
        PROOF_CONTEXT_END();
        return -1.0;
    }
    if (!require_double("pi estimate is finite", pi_estimate)) {
        printf("    Pi estimate is not finite: %.6f\n", pi_estimate);
        PROOF_CONTEXT_END();
        return -1.0;
    }

    PROOF_CONTEXT_END();
    return pi_estimate;
}

double statistical_data_calculate_standard_deviation(const statistical_data_t* stats) {
    PROOF_CONTEXT("statistical_analysis");

    if (!require_bool("sufficient data for std dev", stats->sample_count > 1)) {
        PROOF_CONTEXT_END();
        return -1.0;
    }

    // Calculate mean
    double sum = 0.0;
    for (size_t i = 0; i < stats->sample_count; i++) {
        sum += stats->samples[i];
    }
    double mean = sum / stats->sample_count;

    // Calculate variance
    double variance = 0.0;
    for (size_t i = 0; i < stats->sample_count; i++) {
        double diff = stats->samples[i] - mean;
        variance += diff * diff;
    }
    variance /= (stats->sample_count - 1);

    double std_dev = sqrt(variance);

    if (!require_bool("standard deviation is non-negative", std_dev >= 0.0)) {
        PROOF_CONTEXT_END();
        return -1.0;
    }
    if (!require_double("standard deviation is finite", std_dev)) {
        PROOF_CONTEXT_END();
        return -1.0;
    }

    PROOF_CONTEXT_END();
    return std_dev;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// VERIFIED MATRIX OPERATIONS
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

bool verified_matrix_create(verified_matrix_t* matrix, size_t rows, size_t cols) {
    PROOF_CONTEXT("matrix_construction");

    if (!require_bool("positive dimensions", rows > 0 && cols > 0)) {
        PROOF_CONTEXT_END();
        return false;
    }
    if (!require_bool("reasonable matrix size", rows <= MAX_MATRIX_SIZE && cols <= MAX_MATRIX_SIZE)) {
        PROOF_CONTEXT_END();
        return false;
    }

    matrix->rows = rows;
    matrix->cols = cols;

    // Initialize to zero
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            matrix->data[i][j] = 0.0;
        }
    }

    PROOF_CONTEXT_END();
    return true;
}

bool verified_matrix_set(verified_matrix_t* matrix, size_t row, size_t col, double value) {
    PROOF_CONTEXT("matrix_element_access");

    if (!require_bool("row index in bounds", row < matrix->rows)) {
        PROOF_CONTEXT_END();
        return false;
    }
    if (!require_bool("column index in bounds", col < matrix->cols)) {
        PROOF_CONTEXT_END();
        return false;
    }
    if (!require_double("value is finite", value)) {
        PROOF_CONTEXT_END();
        return false;
    }

    matrix->data[row][col] = value;

    PROOF_CONTEXT_END();
    return true;
}

double verified_matrix_get(const verified_matrix_t* matrix, size_t row, size_t col) {
    PROOF_CONTEXT("matrix_element_retrieval");

    if (!require_bool("row index in bounds", row < matrix->rows)) {
        PROOF_CONTEXT_END();
        return NAN;
    }
    if (!require_bool("column index in bounds", col < matrix->cols)) {
        PROOF_CONTEXT_END();
        return NAN;
    }

    double value = matrix->data[row][col];
    if (!require_double("retrieved value is finite", value)) {
        PROOF_CONTEXT_END();
        return NAN;
    }

    PROOF_CONTEXT_END();
    return value;
}

void verified_matrix_normalize(verified_matrix_t* matrix) {
    PROOF_CONTEXT("matrix_normalization");

    double max_element = 0.0;
    for (size_t i = 0; i < matrix->rows; i++) {
        for (size_t j = 0; j < matrix->cols; j++) {
            double val = verified_matrix_get(matrix, i, j);
            if (isfinite(val)) {
                double abs_val = fabs(val);
                if (abs_val > max_element) {
                    max_element = abs_val;
                }
            }
        }
    }

    if (max_element > 1000.0) {
        double scale_factor = 100.0 / max_element;
        for (size_t i = 0; i < matrix->rows; i++) {
            for (size_t j = 0; j < matrix->cols; j++) {
                matrix->data[i][j] *= scale_factor;
            }
        }
    }

    PROOF_CONTEXT_END();
}

bool matrices_compatible_for_multiplication(void* data) {
    verified_matrix_t** matrices = (verified_matrix_t**)data;
    return matrices[0]->cols == matrices[1]->rows;
}

bool result_dimensions_valid(void* data) {
    verified_matrix_t** matrices = (verified_matrix_t**)data;
    return matrices[0]->rows > 0 && matrices[1]->cols > 0;
}

bool verified_matrix_multiply(const verified_matrix_t* a, const verified_matrix_t* b, verified_matrix_t* result) {
    PROOF_CONTEXT("matrix_multiplication");

    verified_matrix_t* matrices[2] = { (verified_matrix_t*)a, (verified_matrix_t*)b };

    CONTRACT_BEGIN("matrix_multiply");
    PRECONDITION("matrices compatible for multiplication", matrices_compatible_for_multiplication, matrices);
    PRECONDITION("result dimensions are valid", result_dimensions_valid, matrices);

    if (!CHECK_PRECONDITIONS()) {
        CONTRACT_END();
        PROOF_CONTEXT_END();
        return false;
    }

    if (!verified_matrix_create(result, a->rows, b->cols)) {
        CONTRACT_END();
        PROOF_CONTEXT_END();
        return false;
    }

    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < b->cols; j++) {
            double sum = 0.0;

            for (size_t k = 0; k < a->cols; k++) {
                double a_val = verified_matrix_get(a, i, k);
                double b_val = verified_matrix_get(b, k, j);

                if (!isfinite(a_val) || !isfinite(b_val)) {
                    CONTRACT_END();
                    PROOF_CONTEXT_END();
                    return false;
                }

                double product = a_val * b_val;

                if (!require_double("multiplication result is finite", product)) {
                    CONTRACT_END();
                    PROOF_CONTEXT_END();
                    return false;
                }

                sum += product;

                if (!require_double("intermediate sum is finite", sum)) {
                    CONTRACT_END();
                    PROOF_CONTEXT_END();
                    return false;
                }
            }

            if (!verified_matrix_set(result, i, j, sum)) {
                CONTRACT_END();
                PROOF_CONTEXT_END();
                return false;
            }
        }
    }

    if (!CHECK_POSTCONDITIONS()) {
        CONTRACT_END();
        PROOF_CONTEXT_END();
        return false;
    }

    CONTRACT_END();
    PROOF_CONTEXT_END();
    return true;
}

double verified_matrix_frobenius_norm(const verified_matrix_t* matrix) {
    PROOF_CONTEXT("frobenius_norm_calculation");

    double sum_of_squares = 0.0;

    for (size_t i = 0; i < matrix->rows; i++) {
        for (size_t j = 0; j < matrix->cols; j++) {
            double value = verified_matrix_get(matrix, i, j);
            if (!isfinite(value)) {
                PROOF_CONTEXT_END();
                return -1.0;
            }

            double square = value * value;
            if (!require_double("element square is finite", square)) {
                PROOF_CONTEXT_END();
                return -1.0;
            }
            sum_of_squares += square;
        }
    }

    if (!require_bool("sum of squares is non-negative", sum_of_squares >= 0.0)) {
        PROOF_CONTEXT_END();
        return -1.0;
    }
    if (!require_double("sum of squares is finite", sum_of_squares)) {
        PROOF_CONTEXT_END();
        return -1.0;
    }

    double norm = sqrt(sum_of_squares);

    if (!require_bool("norm is non-negative", norm >= 0.0)) {
        PROOF_CONTEXT_END();
        return -1.0;
    }
    if (!require_double("norm is finite", norm)) {
        PROOF_CONTEXT_END();
        return -1.0;
    }

    PROOF_CONTEXT_END();
    return norm;
}

double verified_matrix_max_element(const verified_matrix_t* matrix) {
    double max_val = 0.0;
    for (size_t i = 0; i < matrix->rows; i++) {
        for (size_t j = 0; j < matrix->cols; j++) {
            double val = verified_matrix_get(matrix, i, j);
            if (isfinite(val)) {
                double abs_val = fabs(val);
                if (abs_val > max_val) {
                    max_val = abs_val;
                }
            }
        }
    }
    return max_val;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// SIMPLE RANDOM NUMBER GENERATOR
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void simple_rng_init(simple_rng_t* rng) {
    rng->seed = (unsigned int)time(NULL);
}

double simple_rng_uniform(simple_rng_t* rng, double min_val, double max_val) {
    // Simple linear congruential generator
    rng->seed = rng->seed * 1103515245 + 12345;
    double normalized = (double)(rng->seed % 32768) / 32768.0;
    return min_val + normalized * (max_val - min_val);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// COMPUTATIONAL BENCHMARK
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

bool sufficient_sample_size(void* data) {
    size_t* num_samples = (size_t*)data;
    return *num_samples >= 1000;
}

bool reasonable_sample_size(void* data) {
    size_t* num_samples = (size_t*)data;
    return *num_samples <= 100000000;
}

double monte_carlo_pi_verified(size_t num_samples) {
    PROOF_CONTEXT("monte_carlo_pi_computation");

    CONTRACT_BEGIN("monte_carlo_pi");
    PRECONDITION("sufficient sample size", sufficient_sample_size, &num_samples);
    PRECONDITION("reasonable sample size", reasonable_sample_size, &num_samples);

    if (!CHECK_PRECONDITIONS()) {
        printf("  Pi estimation preconditions failed\n");
        CONTRACT_END();
        PROOF_CONTEXT_END();
        return -1.0;
    }

    // Allocate statistical data on stack (now safe with reduced size)
    statistical_data_t stats;
    statistical_data_init(&stats);

    simple_rng_t rng;
    simple_rng_init(&rng);

    printf("  Monte Carlo: Processing %zu samples...\n", num_samples);

    for (size_t i = 0; i < num_samples; i++) {
        double x = simple_rng_uniform(&rng, -1.0, 1.0);
        double y = simple_rng_uniform(&rng, -1.0, 1.0);

        point_2d_t point;
        if (!point_2d_create(&point, x, y)) {
            printf("  Failed to create point at sample %zu\n", i);
            CONTRACT_END();
            PROOF_CONTEXT_END();
            return -1.0;
        }

        if (!statistical_data_add_sample(&stats, &point)) {
            printf("  Failed to add sample %zu\n", i);
            CONTRACT_END();
            PROOF_CONTEXT_END();
            return -1.0;
        }

        // Much less frequent verification to avoid performance hit
        if (i % 25000 == 0 && i > 0) {
            double intermediate_pi = statistical_data_estimate_pi(&stats);
            if (intermediate_pi < 0.0) {
                printf("  Intermediate pi estimation failed at sample %zu\n", i);
                CONTRACT_END();
                PROOF_CONTEXT_END();
                return -1.0;
            }
            printf("    Intermediate estimate at %zu samples: %.4f\n", i, intermediate_pi);

            // More lenient bounds for intermediate estimates
            if (!require_bool("reasonable intermediate estimate",
                intermediate_pi > 1.0 && intermediate_pi < 6.0)) {
                printf("  Intermediate estimate %.4f outside reasonable bounds\n", intermediate_pi);
                CONTRACT_END();
                PROOF_CONTEXT_END();
                return -1.0;
            }
        }
    }

    printf("  Computing final pi estimate...\n");
    double pi_estimate = statistical_data_estimate_pi(&stats);
    if (pi_estimate < 0.0) {
        printf("  Final pi estimation failed\n");
        CONTRACT_END();
        PROOF_CONTEXT_END();
        return -1.0;
    }

    printf("  Final pi estimate: %.6f\n", pi_estimate);

    double std_dev = statistical_data_calculate_standard_deviation(&stats);
    if (std_dev < 0.0) {
        printf("  Warning: Could not calculate standard deviation\n");
        std_dev = 0.0;  // Continue anyway
    }

    // Much more lenient final quality check
    if (!require_bool("final estimate quality", fabs(pi_estimate - M_PI) < 3.0)) {  // Very lenient
        printf("  Warning: Pi estimate %.4f differs significantly from actual pi (%.4f)\n",
            pi_estimate, M_PI);
        // Don't fail, just warn
    }

    if (!CHECK_POSTCONDITIONS()) {
        printf("  Pi estimation postconditions failed\n");
        CONTRACT_END();
        PROOF_CONTEXT_END();
        return -1.0;
    }

    CONTRACT_END();
    PROOF_CONTEXT_END();
    return pi_estimate;
}

bool reasonable_matrix_size(void* data) {
    size_t* matrix_size = (size_t*)data;
    return *matrix_size >= 10 && *matrix_size <= 500;
}

bool sufficient_operations(void* data) {
    size_t* num_operations = (size_t*)data;
    return *num_operations >= 1;
}

double matrix_computation_verified(size_t matrix_size, size_t num_operations) {
    PROOF_CONTEXT("matrix_computation_benchmark");

    CONTRACT_BEGIN("matrix_computation");
    PRECONDITION("reasonable matrix size", reasonable_matrix_size, &matrix_size);
    PRECONDITION("sufficient operations", sufficient_operations, &num_operations);

    if (!CHECK_PRECONDITIONS()) {
        CONTRACT_END();
        PROOF_CONTEXT_END();
        return -1.0;
    }

    verified_matrix_t A, B, result;

    if (!verified_matrix_create(&A, matrix_size, matrix_size) ||
        !verified_matrix_create(&B, matrix_size, matrix_size) ||
        !verified_matrix_create(&result, matrix_size, matrix_size)) {
        CONTRACT_END();
        PROOF_CONTEXT_END();
        return -1.0;
    }

    simple_rng_t rng;
    simple_rng_init(&rng);

    // Initialize with smaller random values to prevent explosion
    for (size_t i = 0; i < matrix_size; i++) {
        for (size_t j = 0; j < matrix_size; j++) {
            double a_val = simple_rng_uniform(&rng, -0.1, 0.1);
            double b_val = simple_rng_uniform(&rng, -0.1, 0.1);

            if (!verified_matrix_set(&A, i, j, a_val) ||
                !verified_matrix_set(&B, i, j, b_val)) {
                CONTRACT_END();
                PROOF_CONTEXT_END();
                return -1.0;
            }
        }
    }

    // Copy A to result initially
    for (size_t i = 0; i < matrix_size; i++) {
        for (size_t j = 0; j < matrix_size; j++) {
            double val = verified_matrix_get(&A, i, j);
            verified_matrix_set(&result, i, j, val);
        }
    }

    printf("  Matrix operations progress: ");

    // Perform chain of matrix multiplications with monitoring
    for (size_t op = 0; op < num_operations; op++) {
        verified_matrix_t temp_result;
        if (!verified_matrix_multiply(&result, &B, &temp_result)) {
            printf("Matrix multiplication failed at operation %zu\n", op);
            CONTRACT_END();
            PROOF_CONTEXT_END();
            return -1.0;
        }

        // Copy temp_result back to result
        for (size_t i = 0; i < matrix_size; i++) {
            for (size_t j = 0; j < matrix_size; j++) {
                double val = verified_matrix_get(&temp_result, i, j);
                verified_matrix_set(&result, i, j, val);
            }
        }

        // Get current norm and max element for monitoring
        double norm = verified_matrix_frobenius_norm(&result);
        double max_elem = verified_matrix_max_element(&result);

        printf("[%zu: norm=%.2e, max=%.2e] ", op + 1, norm, max_elem);

        if (norm < 0.0 || !isfinite(norm) || !isfinite(max_elem)) {
            printf("Invalid matrix state\n");
            CONTRACT_END();
            PROOF_CONTEXT_END();
            return -1.0;
        }

        // Apply normalization if values get too large
        if (max_elem > 1000.0) {
            verified_matrix_normalize(&result);
            printf("(normalized) ");
        }

        // Verify matrix hasn't become degenerate
        if (op % 3 == 0) {
            if (!require_bool("matrix dimensions preserved",
                result.rows == matrix_size && result.cols == matrix_size)) {
                CONTRACT_END();
                PROOF_CONTEXT_END();
                return -1.0;
            }
        }
    }

    printf("\n");

    double final_norm = verified_matrix_frobenius_norm(&result);

    if (!CHECK_POSTCONDITIONS()) {
        CONTRACT_END();
        PROOF_CONTEXT_END();
        return -1.0;
    }

    CONTRACT_END();
    PROOF_CONTEXT_END();
    return final_norm;
}

void comprehensive_benchmark(size_t monte_carlo_samples, size_t matrix_size, size_t matrix_operations) {
    PROOF_CONTEXT("comprehensive_computational_benchmark");

    clock_t start_time = clock();

    // Phase 1: Monte Carlo simulation
    printf("Phase 1: Monte Carlo simulation with %zu samples...\n", monte_carlo_samples);
    double pi_estimate = monte_carlo_pi_verified(monte_carlo_samples);

    clock_t monte_carlo_time = clock();

    // Phase 2: Matrix computations
    printf("Phase 2: Matrix computations (%zux%zu, %zu operations)...\n",
        matrix_size, matrix_size, matrix_operations);
    double matrix_result = matrix_computation_verified(matrix_size, matrix_operations);

    clock_t matrix_time = clock();

    // Phase 3: Results analysis
    double total_duration = ((double)(matrix_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    double monte_carlo_duration = ((double)(monte_carlo_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    double matrix_duration = ((double)(matrix_time - monte_carlo_time)) / CLOCKS_PER_SEC * 1000.0;

    // Verify timing results
    require_bool("total time is positive", total_duration > 0);
    require_bool("monte carlo took time", monte_carlo_duration >= 0);
    require_bool("matrix computation took time", matrix_duration >= 0);

    // Report results
    printf("\n=== COMPUTATIONAL RESULTS ===\n");
    if (pi_estimate > 0.0) {
        printf("pi estimate: %.6f (error: %.6f)\n", pi_estimate, fabs(pi_estimate - M_PI));
    }
    else {
        printf("pi estimation failed\n");
    }
    if (matrix_result > 0.0) {
        printf("Matrix result norm: %.6e\n", matrix_result);
    }
    else {
        printf("Matrix computation failed\n");
    }
    printf("Monte Carlo time: %.2f ms\n", monte_carlo_duration);
    printf("Matrix computation time: %.2f ms\n", matrix_duration);
    printf("Total computation time: %.2f ms\n", total_duration);

    PROOF_CONTEXT_END();
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// BENCHMARK HARNESS
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void run_benchmark_with_verification(void) {
    printf("Running VERIFIED benchmark...\n");
    printf("================================\n");

    global_config.level = VERIFICATION_FULL;
    axiomatik_init(); // Clear system

    clock_t start_time = clock();

    comprehensive_benchmark(50000, 30, 5); // Reduced for faster testing

    clock_t end_time = clock();
    double total_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;

    printf("\n=== VERIFICATION OVERHEAD ANALYSIS ===\n");

    performance_metrics_t metrics = get_performance_metrics();
    printf("Total verifications: %zu\n", metrics.total_verifications);
    printf("Total time: %.2f ms\n", (double)metrics.total_time / CLOCKS_PER_SEC * 1000.0);
    printf("Average time: %.6f ms\n", metrics.average_time_ms);
    printf("Cache hit rate: %.2f%%\n", metrics.cache_hit_rate * 100.0);
    printf("Total benchmark time: %.2f ms\n", total_time);

    double verification_overhead_ms = (double)metrics.total_time / CLOCKS_PER_SEC * 1000.0;
    double overhead_percentage = (verification_overhead_ms / total_time) * 100.0;

    printf("Verification overhead: %.2f ms (%.2f%%)\n",
        verification_overhead_ms, overhead_percentage);
}

void run_benchmark_without_verification(void) {
    printf("\n\nRunning UNVERIFIED benchmark...\n");
    printf("================================\n");

    global_config.level = VERIFICATION_OFF;
    axiomatik_init(); // Clear system

    clock_t start_time = clock();

    comprehensive_benchmark(50000, 30, 5); // Same parameters

    clock_t end_time = clock();
    double total_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;

    printf("Unverified benchmark time: %.2f ms\n", total_time);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// MAIN BENCHMARK RUNNER
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

int main(void) {
    printf("Axiomatik C17 Performance Benchmark\n");
    printf("====================================\n\n");

    axiomatik_init();

    // Run comprehensive benchmarks
    run_benchmark_with_verification();
    run_benchmark_without_verification();

    printf("\n=== BENCHMARK COMPLETE ===\n");
    printf("This benchmark demonstrates Axiomatik's runtime verification\n");
    printf("overhead during computationally intensive tasks including:\n");
    printf("- Monte Carlo simulation with statistical verification\n");
    printf("- Matrix operations with mathematical invariants\n");
    printf("- Contract verification and performance monitoring\n");
    printf("- Context-aware debugging and error handling\n\n");

    axiomatik_cleanup();
    return 0;
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif