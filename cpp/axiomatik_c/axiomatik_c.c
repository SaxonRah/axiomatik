/**
 * axiomatik_c.c - Implementation of C17 Runtime Verification
 */

#include "axiomatik_c.h"

 // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 // GLOBAL STATE
 // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

global_config_t global_config = {
    .level = VERIFICATION_FULL,
    .cache_enabled = true,
    .max_proof_steps = MAX_PROOF_STEPS,
    .performance_mode = false,
    .debug_mode = false
};

proof_system_t global_proof_system = { 0 };

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// INITIALIZATION AND CLEANUP
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void axiomatik_init(void) {
    memset(&global_proof_system, 0, sizeof(proof_system_t));
    global_proof_system.cache.max_size = global_config.cache_enabled ? MAX_CACHE_ENTRIES : 0;
}

void axiomatik_cleanup(void) {
    // In C17, we don't need to free anything since we use static allocation
    memset(&global_proof_system, 0, sizeof(proof_system_t));
}

bool should_verify(const char* context_type) {
    if (global_config.level == VERIFICATION_OFF) return false;
    if (global_config.level == VERIFICATION_CONTRACTS &&
        context_type && strcmp(context_type, "contract") != 0) return false;
    return true;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CACHE IMPLEMENTATION
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

static int find_cache_entry(const char* key) {
    for (size_t i = 0; i < global_proof_system.cache.count; i++) {
        if (global_proof_system.cache.entries[i].valid &&
            strcmp(global_proof_system.cache.entries[i].key, key) == 0) {
            return (int)i;
        }
    }
    return -1;
}

static void evict_oldest_cache_entry(void) {
    if (global_proof_system.cache.count == 0) return;

    size_t oldest_idx = 0;
    clock_t oldest_time = global_proof_system.cache.entries[0].last_access;

    for (size_t i = 1; i < global_proof_system.cache.count; i++) {
        if (global_proof_system.cache.entries[i].valid &&
            global_proof_system.cache.entries[i].last_access < oldest_time) {
            oldest_time = global_proof_system.cache.entries[i].last_access;
            oldest_idx = i;
        }
    }

    global_proof_system.cache.entries[oldest_idx].valid = false;
}

bool cache_get(const char* key) {
    if (!global_config.cache_enabled) return false;

    int idx = find_cache_entry(key);
    if (idx >= 0) {
        global_proof_system.cache.entries[idx].last_access = clock();
        global_proof_system.cache.entries[idx].access_count++;
        global_proof_system.cache.hit_count++;
        return global_proof_system.cache.entries[idx].result;
    }

    global_proof_system.cache.miss_count++;
    return false;
}

void cache_set(const char* key, bool value) {
    if (!global_config.cache_enabled) return;

    if (global_proof_system.cache.count >= global_proof_system.cache.max_size) {
        evict_oldest_cache_entry();
    }

    // Find first invalid slot or add new entry
    for (size_t i = 0; i < MAX_CACHE_ENTRIES; i++) {
        if (!global_proof_system.cache.entries[i].valid) {
            safe_strcpy(global_proof_system.cache.entries[i].key, key, MAX_STRING_LENGTH - 1);
            global_proof_system.cache.entries[i].key[MAX_STRING_LENGTH - 1] = '\0';
            global_proof_system.cache.entries[i].result = value;
            global_proof_system.cache.entries[i].last_access = clock();
            global_proof_system.cache.entries[i].access_count = 1;
            global_proof_system.cache.entries[i].valid = true;

            if (i >= global_proof_system.cache.count) {
                global_proof_system.cache.count = i + 1;
            }
            break;
        }
    }
}

void cache_clear(void) {
    for (size_t i = 0; i < global_proof_system.cache.count; i++) {
        global_proof_system.cache.entries[i].valid = false;
    }
    global_proof_system.cache.count = 0;
    global_proof_system.cache.hit_count = 0;
    global_proof_system.cache.miss_count = 0;
}

double cache_hit_rate(void) {
    size_t total = global_proof_system.cache.hit_count + global_proof_system.cache.miss_count;
    return total > 0 ? (double)global_proof_system.cache.hit_count / total : 0.0;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CONTEXT MANAGEMENT
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void push_context(const char* context) {
    if (global_proof_system.context_depth < MAX_CONTEXT_DEPTH) {
        safe_strcpy(global_proof_system.context_stack[global_proof_system.context_depth].name,
            context, MAX_STRING_LENGTH - 1);
        global_proof_system.context_stack[global_proof_system.context_depth].name[MAX_STRING_LENGTH - 1] = '\0';
        global_proof_system.context_depth++;
    }
}

void pop_context(void) {
    if (global_proof_system.context_depth > 0) {
        global_proof_system.context_depth--;
    }
}

const char* current_context(void) {
    if (global_proof_system.context_depth > 0) {
        return global_proof_system.context_stack[global_proof_system.context_depth - 1].name;
    }
    return "global";
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CORE VERIFICATION FUNCTIONS
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

static void record_proof_step(const char* claim, bool result, clock_t verification_time) {
    if (global_proof_system.step_count < MAX_PROOF_STEPS) {
        proof_step_t* step = &global_proof_system.steps[global_proof_system.step_count];

        safe_strcpy(step->claim, claim, MAX_STRING_LENGTH - 1);
        step->claim[MAX_STRING_LENGTH - 1] = '\0';

        safe_strcpy(step->context, current_context(), MAX_STRING_LENGTH - 1);
        step->context[MAX_STRING_LENGTH - 1] = '\0';

        step->timestamp = clock();
        step->succeeded = result;
        step->verification_time = verification_time;

        global_proof_system.step_count++;
    }

    global_proof_system.total_verifications++;
    global_proof_system.total_verification_time += verification_time;

    // Update context counts
    const char* ctx = current_context();
    bool found = false;
    for (size_t i = 0; i < global_proof_system.context_count_entries; i++) {
        if (strcmp(global_proof_system.context_counts[i].context, ctx) == 0) {
            global_proof_system.context_counts[i].count++;
            found = true;
            break;
        }
    }
    if (!found && global_proof_system.context_count_entries < MAX_CONTEXT_TYPES) {
        safe_strcpy(global_proof_system.context_counts[global_proof_system.context_count_entries].context,
            ctx, MAX_STRING_LENGTH - 1);
        global_proof_system.context_counts[global_proof_system.context_count_entries].context[MAX_STRING_LENGTH - 1] = '\0';
        global_proof_system.context_counts[global_proof_system.context_count_entries].count = 1;
        global_proof_system.context_count_entries++;
    }
}

bool require_bool(const char* claim, bool condition) {
    if (!should_verify(NULL)) {
        return true;
    }

    clock_t start_time = clock();
    bool result = condition;
    clock_t end_time = clock();
    clock_t verification_time = end_time - start_time;

    record_proof_step(claim, result, verification_time);

    return result;
}

bool require_int(const char* claim, int value) {
    return require_bool(claim, value != 0);
}

/*
bool require_double(const char* claim, double value) {
    return require_bool(claim, value != 0.0 && isfinite(value));
}
*/

bool require_double(const char* claim, double value) {
    return require_bool(claim, isfinite(value));
}

bool require_string(const char* claim, const char* value) {
    return require_bool(claim, value != NULL && strlen(value) > 0);
}

bool require_predicate(const char* claim, predicate_func_t predicate, void* data) {
    if (!should_verify(NULL)) {
        return true;
    }

    clock_t start_time = clock();
    bool result = predicate ? predicate(data) : false;
    clock_t end_time = clock();
    clock_t verification_time = end_time - start_time;

    record_proof_step(claim, result, verification_time);

    return result;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// PERFORMANCE ANALYSIS
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

performance_metrics_t get_performance_metrics(void) {
    performance_metrics_t metrics = { 0 };

    metrics.total_verifications = global_proof_system.total_verifications;
    metrics.total_time = global_proof_system.total_verification_time;
    metrics.cache_hit_rate = cache_hit_rate();

    if (global_proof_system.step_count > 0) {
        clock_t min_time = global_proof_system.steps[0].verification_time;
        clock_t max_time = global_proof_system.steps[0].verification_time;

        for (size_t i = 1; i < global_proof_system.step_count; i++) {
            clock_t time = global_proof_system.steps[i].verification_time;
            if (time < min_time) min_time = time;
            if (time > max_time) max_time = time;
        }

        metrics.min_time = min_time;
        metrics.max_time = max_time;

        if (metrics.total_verifications > 0) {
            metrics.average_time_ms = ((double)metrics.total_time / CLOCKS_PER_SEC * 1000.0) /
                metrics.total_verifications;
        }
    }

    // Copy context breakdown
    metrics.context_breakdown_count = global_proof_system.context_count_entries;
    for (size_t i = 0; i < metrics.context_breakdown_count && i < MAX_CONTEXT_TYPES; i++) {
        metrics.context_breakdown[i] = global_proof_system.context_counts[i];
    }

    return metrics;
}

void print_performance_report(void) {
    performance_metrics_t metrics = get_performance_metrics();

    printf("Axiomatik C17 Performance Report\n");
    printf("================================\n");
    printf("Total verifications: %zu\n", metrics.total_verifications);
    printf("Total time: %.2f ms\n", (double)metrics.total_time / CLOCKS_PER_SEC * 1000.0);
    printf("Average time: %.6f ms\n", metrics.average_time_ms);
    printf("Min time: %.6f ms\n", (double)metrics.min_time / CLOCKS_PER_SEC * 1000.0);
    printf("Max time: %.6f ms\n", (double)metrics.max_time / CLOCKS_PER_SEC * 1000.0);
    printf("Cache hit rate: %.2f%%\n", metrics.cache_hit_rate * 100.0);

    printf("\nContext breakdown:\n");
    for (size_t i = 0; i < metrics.context_breakdown_count; i++) {
        printf("  %s: %zu verifications\n",
            metrics.context_breakdown[i].context,
            metrics.context_breakdown[i].count);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CONTRACT SUPPORT
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function_contract_t* create_contract(const char* function_name) {
    function_contract_t* contract = malloc(sizeof(function_contract_t));
    if (contract) {
        memset(contract, 0, sizeof(function_contract_t));
        safe_strcpy(contract->function_name, function_name, MAX_STRING_LENGTH - 1);
        contract->function_name[MAX_STRING_LENGTH - 1] = '\0';
    }
    return contract;
}

void add_precondition(function_contract_t* contract, const char* description,
    predicate_func_t predicate, void* data) {
    if (contract && contract->precondition_count < 10) {
        contract_condition_t* cond = &contract->preconditions[contract->precondition_count];
        safe_strcpy(cond->description, description, MAX_STRING_LENGTH - 1);
        cond->description[MAX_STRING_LENGTH - 1] = '\0';
        cond->predicate = predicate;
        cond->data = data;
        contract->precondition_count++;
    }
}

void add_postcondition(function_contract_t* contract, const char* description,
    predicate_func_t predicate, void* data) {
    if (contract && contract->postcondition_count < 10) {
        contract_condition_t* cond = &contract->postconditions[contract->postcondition_count];
        safe_strcpy(cond->description, description, MAX_STRING_LENGTH - 1);
        cond->description[MAX_STRING_LENGTH - 1] = '\0';
        cond->predicate = predicate;
        cond->data = data;
        contract->postcondition_count++;
    }
}

bool check_preconditions(function_contract_t* contract) {
    if (!contract) return false;

    char context_name[MAX_STRING_LENGTH];
    snprintf(context_name, sizeof(context_name), "preconditions_%s", contract->function_name);
    push_context(context_name);

    bool all_passed = true;
    for (size_t i = 0; i < contract->precondition_count; i++) {
        char claim[MAX_STRING_LENGTH];
        snprintf(claim, sizeof(claim), "precondition: %s", contract->preconditions[i].description);

        bool result = require_predicate(claim, contract->preconditions[i].predicate,
            contract->preconditions[i].data);
        if (!result) {
            all_passed = false;
        }
    }

    pop_context();
    return all_passed;
}

bool check_postconditions(function_contract_t* contract) {
    if (!contract) return false;

    char context_name[MAX_STRING_LENGTH];
    snprintf(context_name, sizeof(context_name), "postconditions_%s", contract->function_name);
    push_context(context_name);

    bool all_passed = true;
    for (size_t i = 0; i < contract->postcondition_count; i++) {
        char claim[MAX_STRING_LENGTH];
        snprintf(claim, sizeof(claim), "postcondition: %s", contract->postconditions[i].description);

        bool result = require_predicate(claim, contract->postconditions[i].predicate,
            contract->postconditions[i].data);
        if (!result) {
            all_passed = false;
        }
    }

    pop_context();
    return all_passed;
}

void destroy_contract(function_contract_t* contract) {
    free(contract);
}

void safe_strcpy(char* dest, const char* src, size_t dest_size) {
    if (dest && src && dest_size > 0) {
#ifdef _MSC_VER
        strcpy_s(dest, dest_size, src);
#else
        strncpy(dest, src, dest_size - 1);
        dest[dest_size - 1] = '\0';
#endif
    }
}