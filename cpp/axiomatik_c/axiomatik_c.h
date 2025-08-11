/**
 * axiomatik_c.h - C17 Runtime Verification Library
 * Transparent Code verification for C without atomics/threads
 */

#ifndef AXIOMATIK_C_H
#define AXIOMATIK_C_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <stdint.h>  // For int64_t

 // Suppress MSVC warnings about "unsafe" functions
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#pragma warning(push)
#pragma warning(disable: 4996) // 'function': This function or variable may be unsafe
#endif

// Maximum sizes for static allocation
#define MAX_PROOF_STEPS 10000
#define MAX_CONTEXT_DEPTH 100
#define MAX_STRING_LENGTH 256
#define MAX_CACHE_ENTRIES 1000
#define MAX_CONTEXT_TYPES 50

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CORE TYPES AND CONFIGURATION
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

typedef enum {
    VERIFICATION_OFF,
    VERIFICATION_CONTRACTS,
    VERIFICATION_INVARIANTS,
    VERIFICATION_FULL,
    VERIFICATION_DEBUG
} verification_level_t;

typedef enum {
    SECURITY_PUBLIC,
    SECURITY_CONFIDENTIAL,
    SECURITY_SECRET,
    SECURITY_TOP_SECRET
} security_label_t;

typedef struct {
    verification_level_t level;
    bool cache_enabled;
    size_t max_proof_steps;
    bool performance_mode;
    bool debug_mode;
} global_config_t;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// PROOF FAILURE AND EVIDENCE TYPES
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

typedef struct {
    char claim[MAX_STRING_LENGTH];
    char context[MAX_STRING_LENGTH];
    char details[MAX_STRING_LENGTH];
    clock_t timestamp;
    bool valid;
} proof_failure_t;

typedef struct {
    char claim[MAX_STRING_LENGTH];
    char context[MAX_STRING_LENGTH];
    clock_t timestamp;
    bool succeeded;
    clock_t verification_time;
} proof_step_t;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// PROOF CACHE
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

typedef struct {
    char key[MAX_STRING_LENGTH];
    bool result;
    clock_t last_access;
    size_t access_count;
    bool valid;
} cache_entry_t;

typedef struct {
    cache_entry_t entries[MAX_CACHE_ENTRIES];
    size_t count;
    size_t max_size;
    size_t hit_count;
    size_t miss_count;
} proof_cache_t;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CORE PROOF SYSTEM
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

typedef struct {
    char name[MAX_STRING_LENGTH];
} context_entry_t;

typedef struct {
    char context[MAX_STRING_LENGTH];
    size_t count;
} context_count_t;

typedef struct {
    proof_step_t steps[MAX_PROOF_STEPS];
    size_t step_count;

    context_entry_t context_stack[MAX_CONTEXT_DEPTH];
    size_t context_depth;

    proof_cache_t cache;

    // Performance metrics
    size_t total_verifications;
    clock_t total_verification_time;

    context_count_t context_counts[MAX_CONTEXT_TYPES];
    size_t context_count_entries;
} proof_system_t;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// FUNCTION CONTRACTS
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

typedef bool (*predicate_func_t)(void* data);

typedef struct {
    char description[MAX_STRING_LENGTH];
    predicate_func_t predicate;
    void* data;
} contract_condition_t;

typedef struct {
    char function_name[MAX_STRING_LENGTH];
    contract_condition_t preconditions[10];
    contract_condition_t postconditions[10];
    size_t precondition_count;
    size_t postcondition_count;
} function_contract_t;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// GLOBAL STATE
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

extern global_config_t global_config;
extern proof_system_t global_proof_system;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CORE API FUNCTIONS
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Configuration
void axiomatik_init(void);
void axiomatik_cleanup(void);
bool should_verify(const char* context_type);

// Core verification
bool require_bool(const char* claim, bool condition);
bool require_int(const char* claim, int value);
bool require_double(const char* claim, double value);
bool require_string(const char* claim, const char* value);
bool require_predicate(const char* claim, predicate_func_t predicate, void* data);

// Context management
void push_context(const char* context);
void pop_context(void);
const char* current_context(void);

// Cache operations
bool cache_get(const char* key);
void cache_set(const char* key, bool value);
void cache_clear(void);
double cache_hit_rate(void);

// Performance analysis
typedef struct {
    size_t total_verifications;
    clock_t total_time;
    double average_time_ms;
    clock_t min_time;
    clock_t max_time;
    double cache_hit_rate;
    size_t context_breakdown_count;
    context_count_t context_breakdown[MAX_CONTEXT_TYPES];
} performance_metrics_t;

performance_metrics_t get_performance_metrics(void);
void print_performance_report(void);

// Contract support
function_contract_t* create_contract(const char* function_name);
void add_precondition(function_contract_t* contract, const char* description,
    predicate_func_t predicate, void* data);
void add_postcondition(function_contract_t* contract, const char* description,
    predicate_func_t predicate, void* data);
bool check_preconditions(function_contract_t* contract);
bool check_postconditions(function_contract_t* contract);
void destroy_contract(function_contract_t* contract);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CONVENIENCE MACROS
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#define PROOF_CONTEXT(name) push_context(name); \
    char _context_name[] = name; \
    (void)_context_name // Suppress unused variable warning

#define PROOF_CONTEXT_END() pop_context()

#define REQUIRE(claim, condition) do { \
    if (!require_bool(claim, condition)) { \
        fprintf(stderr, "PROOF FAILURE: %s in context %s\n", claim, current_context()); \
        return false; \
    } \
} while(0)

#define REQUIRE_RETURN(claim, condition, retval) do { \
    if (!require_bool(claim, condition)) { \
        fprintf(stderr, "PROOF FAILURE: %s in context %s\n", claim, current_context()); \
        return retval; \
    } \
} while(0)

#define CONTRACT_BEGIN(name) function_contract_t* _contract = create_contract(name)
#define PRECONDITION(desc, pred, data) add_precondition(_contract, desc, pred, data)
#define POSTCONDITION(desc, pred, data) add_postcondition(_contract, desc, pred, data)
#define CHECK_PRECONDITIONS() check_preconditions(_contract)
#define CHECK_POSTCONDITIONS() check_postconditions(_contract)
#define CONTRACT_END() destroy_contract(_contract)

#ifdef _MSC_VER
#pragma warning(pop)
#endif

void safe_strcpy(char* dest, const char* src, size_t dest_size);

#endif // AXIOMATIK_C_H