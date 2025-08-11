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
    if (global_config.level == VERIFICATION_CONTRACTS && context_type && safe_strcmp( context_type, "contract" ) != 0 )
		return false;
    return true;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CACHE IMPLEMENTATION
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

static int find_cache_entry(const char* key) {
    for (size_t i = 0; i < global_proof_system.cache.count; i++) {
        if (global_proof_system.cache.entries[i].valid && safe_strcmp( global_proof_system.cache.entries[i].key, key ) == 0 )
		{
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

void push_context( const char* context ) {
	if ( context != NULL && safe_string_is_valid( context ) && global_proof_system.context_depth < MAX_CONTEXT_DEPTH ) {
		safe_strcpy( global_proof_system.context_stack[global_proof_system.context_depth].name, context, MAX_STRING_LENGTH );
		global_proof_system.context_depth++;
	}
}

void pop_context(void) {
    if (global_proof_system.context_depth > 0) {
        global_proof_system.context_depth--;
    }
}

const char* current_context( void ) {
	if ( global_proof_system.context_depth > 0 && global_proof_system.context_depth <= MAX_CONTEXT_DEPTH ) {
		size_t index = global_proof_system.context_depth - 1;
		return global_proof_system.context_stack[index].name;
	}
	return "global";
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CORE VERIFICATION FUNCTIONS
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

static void record_proof_step( const char* claim, bool result, clock_t verification_time )
{
	if ( global_proof_system.step_count < MAX_PROOF_STEPS )
	{
		proof_step_t* step = &global_proof_system.steps[global_proof_system.step_count];

		safe_strcpy( step->claim, claim, MAX_STRING_LENGTH );
		safe_strcpy( step->context, current_context(), MAX_STRING_LENGTH );

		step->timestamp = clock();
		step->succeeded = result;
		step->verification_time = verification_time;

		global_proof_system.step_count++;
	}

	global_proof_system.total_verifications++;
	global_proof_system.total_verification_time += verification_time;

	// Update context counts - FIXED VERSION
	const char* ctx = current_context();
	if ( ctx != NULL && safe_string_is_valid( ctx ) )
	{
		bool found = false;

		// Search for existing context
		for ( size_t i = 0; i < global_proof_system.context_count_entries; i++ )
		{
			if ( i < MAX_CONTEXT_TYPES && safe_strings_equal( global_proof_system.context_counts[i].context, ctx ) )
			{
				global_proof_system.context_counts[i].count++;
				found = true;
				break;
			}
		}

		// Add new context if not found and we have space
		if ( !found && global_proof_system.context_count_entries < MAX_CONTEXT_TYPES )
		{
			size_t index = global_proof_system.context_count_entries;

			// Use safe_strcpy instead of strncpy + manual null termination
			safe_strcpy( global_proof_system.context_counts[index].context, ctx, MAX_STRING_LENGTH );

			global_proof_system.context_counts[index].count = 1;
			global_proof_system.context_count_entries++;
		}
	}
}

/*
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
*/

bool require_bool( const char* claim, bool condition )
{
	if ( !should_verify( NULL ) )
	{
		return true;
	}

	clock_t start_time = clock();
	bool result = condition;
	clock_t end_time = clock();
	clock_t verification_time = end_time - start_time;

	// Record the step
	record_proof_step( claim, result, verification_time );

	// Optional reporting
	if ( global_report_config.show_all_verifications || ( global_report_config.show_failures_only && !result ) )
	{

		FILE* out = global_report_config.output_file ? global_report_config.output_file : stdout;
		fprintf( out, "[AXIOMATIK] %s: %s - %s\n", current_context(), result ? "PASS" : "FAIL", claim );

		if ( !result )
		{
			fprintf( out, "Verification FAILED in context '%s'\n", current_context() );
		}
	}

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
        printf("%s: %zu verifications\n",
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// VERIFICATION REPORTING
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

verification_report_config_t global_report_config = { .show_all_verifications = false,
													  .show_failures_only = true,
													  .show_context_changes = false,
													  .show_performance_stats = false,
													  .show_summaries = true,
													  .output_file = NULL };

void configure_verification_reporting( bool show_failures, bool show_performance, bool show_context ) {
	global_report_config.show_failures_only = show_failures;
	global_report_config.show_performance_stats = show_performance;
	global_report_config.show_context_changes = show_context;
}

void enable_detailed_verification_output( void ) {
	global_report_config.show_all_verifications = true;
	global_report_config.show_failures_only = true;
	global_report_config.show_context_changes = true;
	global_report_config.show_performance_stats = true;
}

void disable_verification_output( void ) {
	global_report_config.show_all_verifications = false;
	global_report_config.show_failures_only = false;
	global_report_config.show_context_changes = false;
	global_report_config.show_performance_stats = false;
}

void print_verification_summary( void ) {
	printf( "\n=== AXIOMATIK C17 VERIFICATION SUMMARY ===\n" );
	printf( "Total verifications performed: %zu\n", global_proof_system.total_verifications );
	printf( "Total verification time: %.2f ms\n", (double)global_proof_system.total_verification_time / CLOCKS_PER_SEC * 1000.0 );

	if ( global_proof_system.total_verifications > 0 ) {
		printf( "Average time per verification: %.6f ms\n",
				( (double)global_proof_system.total_verification_time / CLOCKS_PER_SEC * 1000.0 ) /
					global_proof_system.total_verifications );
	}

	printf( "Active contexts tracked: %zu\n", global_proof_system.context_count_entries );
	printf( "Cache hit rate: %.2f%%\n", cache_hit_rate() * 100.0 );

	// Show failure count
	size_t failure_count = 0;
	for ( size_t i = 0; i < global_proof_system.step_count; i++ ) {
		if ( !global_proof_system.steps[i].succeeded ) {
			failure_count++;
		}
	}
	printf( "Verification failures: %zu\n", failure_count );

	printf( "==========================================\n\n" );
}

void print_verification_context_tree( void ) {
	printf( "\n=== VERIFICATION CONTEXT BREAKDOWN ===\n" );
	for ( size_t i = 0; i < global_proof_system.context_count_entries; i++ ) {
		printf( "%s: %zu verifications\n", global_proof_system.context_counts[i].context,
				global_proof_system.context_counts[i].count );
	}
	printf( "=====================================\n\n" );
}

void print_recent_verifications( size_t count ) {
	printf( "\n=== RECENT VERIFICATIONS ===\n" );

	size_t start = 0;
	if ( global_proof_system.step_count > count ) {
		start = global_proof_system.step_count - count;
	}

	for ( size_t i = start; i < global_proof_system.step_count; i++ ) {
		proof_step_t* step = &global_proof_system.steps[i];
		printf( "[%s] %s: %s (%.3f ms)\n", step->context, step->succeeded ? "PASS" : "FAIL", step->claim,
				(double)step->verification_time / CLOCKS_PER_SEC * 1000.0 );
	}
	printf( "===========================\n\n" );
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

bool safe_string_is_valid( const char* str )
{
	if ( str == NULL )
	{
		// printf( "String validation: NULL pointer\n" );
		return false;
	}

	// Check if we can safely read the first few characters
	size_t max_check = 256; // Reduced check length
	for ( size_t i = 0; i < max_check; i++ )
	{
		char c = str[i];

		if ( c == '\0' )
		{
			return true; // Found null terminator - string is valid
		}

		// More lenient character validation - only reject obvious garbage
		if ( c < 0 )
		{
			// Negative values in char might indicate garbage (depending on compiler)
			// printf( "String validation failed: negative char value at position %zu\n", i );
			return false;
		}

		// Allow all positive ASCII values (0-127) and basic extended ASCII
		// Only reject extreme values that are clearly garbage
		if ( (unsigned char)c > 250 )
		{
			// printf( "String validation failed: suspicious char value %d at position %zu\n", (int)(unsigned char)c, i );
			return false;
		}
	}

	// String is too long or not null-terminated
	// printf( "String validation failed: no null terminator found in first %zu characters\n", max_check );
	return false;
}

int safe_strcmp( const char* str1, const char* str2 )
{
	// Handle NULL cases
	if ( str1 == NULL && str2 == NULL )
		return 0; // Both NULL = equal
	if ( str1 == NULL )
		return -1; // NULL < non-NULL
	if ( str2 == NULL )
		return 1; // non-NULL > NULL

	// Validate strings before comparing - with debug info
	if ( !safe_string_is_valid( str1 ) )
	{
		// printf( "Warning: safe_strcmp detected invalid str1: %p\n", (void*)str1 );
		if ( str1 != NULL )
		{
			// printf( "First few bytes: " );
			for ( int i = 0; i < 8 && i < 256; i++ )
			{
				// printf( "%02x ", (unsigned char)str1[i] );
				if ( str1[i] == '\0' )
					break;
			}
			// printf( "\n" );
		}
		return -1;
	}
	if ( !safe_string_is_valid( str2 ) )
	{
		// printf( "Warning: safe_strcmp detected invalid str2: %p\n", (void*)str2 );
		if ( str2 != NULL )
		{
			// printf( "First few bytes: " );
			for ( int i = 0; i < 8 && i < 256; i++ )
			{
				// printf( "%02x ", (unsigned char)str2[i] );
				if ( str2[i] == '\0' )
					break;
			}
			// printf( "\n" );
		}
		return 1;
	}

	// Safe to use strcmp now
	return strcmp( str1, str2 );
}

bool safe_strings_equal( const char* str1, const char* str2 ) {
	return safe_strcmp( str1, str2 ) == 0;
}
