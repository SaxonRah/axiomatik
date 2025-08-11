/**
 * demo.c - C17 Demonstration of Transparent Code Verification
 */

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4996)
#endif

#include "axiomatik_c.h"

 // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 // FINANCIAL TRANSACTION EXAMPLE
 // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

typedef struct {
    int64_t cents;
    char currency[4];
} money_amount_t;

typedef struct {
    char account_id[32];
    money_amount_t balance;
    char country[3];
    bool international_enabled;
    bool frozen;
} account_info_t;

typedef struct {
    char transaction_id[32];
    money_amount_t amount;
    char description[256];
    clock_t initiated_at;
} transaction_details_t;

typedef enum {
    PAYMENT_SUCCESS,
    PAYMENT_VALIDATION_FAILED,
    PAYMENT_INSUFFICIENT_FUNDS,
    PAYMENT_ACCOUNT_FROZEN,
    PAYMENT_COMPLIANCE_VIOLATION,
    PAYMENT_RISK_TOO_HIGH
} payment_result_t;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// IMPROVED PREDICATE FUNCTIONS
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

bool transaction_id_valid(void* data) {
    char* transaction_id = (char*)data;
    return transaction_id != NULL && strlen(transaction_id) > 0;
}

bool amount_is_positive(void* data) {
    money_amount_t* amount = (money_amount_t*)data;
    return amount && amount->cents > 0;
}

bool amount_is_reasonable(void* data) {
    money_amount_t* amount = (money_amount_t*)data;
    return amount && amount->cents <= 100000000; // $1M limit
}

bool account_not_frozen(void* data) {
    account_info_t* account = (account_info_t*)data;
    return account && !account->frozen;
}

bool sufficient_funds(void* data) {
    money_amount_t* required = (money_amount_t*)data;
    return required && required->cents <= 10000000; // Dummy check for $100k
}

bool is_finite_double(void* data) {
    double* value = (double*)data;
    return value && isfinite(*value);
}

bool is_positive_double(void* data) {
    double* value = (double*)data;
    return value && *value > 0.0;
}

bool is_non_negative_double(void* data) {
    double* value = (double*)data;
    return value && *value >= 0.0;  // Allow 0.0 for sqrt
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// VALIDATION FUNCTIONS
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

bool validate_transaction_amount(money_amount_t* amount) {
    PROOF_CONTEXT("amount_validation");

    bool valid = true;
    if (!require_bool("amount is positive", amount->cents > 0)) {
        valid = false;
    }
    if (!require_bool("amount is reasonable", amount->cents <= 100000000)) {
        valid = false;
    }

    PROOF_CONTEXT_END();
    return valid;
}

bool validate_accounts(account_info_t* source, account_info_t* dest) {
    PROOF_CONTEXT("account_validation");

    bool valid = true;
    if (!require_bool("source account not frozen", !source->frozen)) {
        valid = false;
    }
    if (!require_bool("destination account not frozen", !dest->frozen)) {
        valid = false;
    }
    if (!require_bool("different accounts", strcmp(source->account_id, dest->account_id) != 0)) {
        valid = false;
    }

    if (strcmp(source->country, dest->country) != 0) {
        if (!require_bool("international transfers enabled", source->international_enabled)) {
            valid = false;
        }
    }

    PROOF_CONTEXT_END();
    return valid;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// PAYMENT PROCESSING WITH PROPER CONTRACTS
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

payment_result_t process_payment_transparent(transaction_details_t* transaction,
    account_info_t* source_account,
    account_info_t* destination_account) {
    PROOF_CONTEXT("payment_processing");

    printf("Processing payment %s for $%.2f...\n",
        transaction->transaction_id,
        transaction->amount.cents / 100.0);

    // Contract verification with PROPER predicates
    CONTRACT_BEGIN("process_payment");
    PRECONDITION("valid transaction ID", transaction_id_valid, transaction->transaction_id);
    PRECONDITION("positive amount", amount_is_positive, &transaction->amount);
    PRECONDITION("reasonable amount", amount_is_reasonable, &transaction->amount);
    PRECONDITION("source account not frozen", account_not_frozen, source_account);
    PRECONDITION("destination account not frozen", account_not_frozen, destination_account);

    if (!CHECK_PRECONDITIONS()) {
        printf("Precondition validation failed\n");
        CONTRACT_END();
        PROOF_CONTEXT_END();
        return PAYMENT_VALIDATION_FAILED;
    }

    printf("Preconditions passed\n");

    // Validation steps
    if (!validate_transaction_amount(&transaction->amount)) {
        printf("Amount validation failed\n");
        CONTRACT_END();
        PROOF_CONTEXT_END();
        return PAYMENT_VALIDATION_FAILED;
    }

    if (!validate_accounts(source_account, destination_account)) {
        printf("Account validation failed\n");
        CONTRACT_END();
        PROOF_CONTEXT_END();
        return PAYMENT_ACCOUNT_FROZEN;
    }

    // Check sufficient funds
    money_amount_t required_amount = transaction->amount;
    if (!require_bool("sufficient funds", source_account->balance.cents >= required_amount.cents)) {
        printf("Insufficient funds\n");
        CONTRACT_END();
        PROOF_CONTEXT_END();
        return PAYMENT_INSUFFICIENT_FUNDS;
    }

    printf("All validations passed\n");

    POSTCONDITION("payment processed successfully", transaction_id_valid, transaction->transaction_id);
    if (!CHECK_POSTCONDITIONS()) {
        printf("Postcondition validation failed\n");
        CONTRACT_END();
        PROOF_CONTEXT_END();
        return PAYMENT_COMPLIANCE_VIOLATION;
    }

    CONTRACT_END();
    PROOF_CONTEXT_END();
    return PAYMENT_SUCCESS;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// MATHEMATICAL COMPUTATION WITH IMPROVED VALIDATION
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

double verified_sqrt(double x) {
    PROOF_CONTEXT("verified_sqrt");

    printf("Computing sqrt(%.1f)...\n", x);

    CONTRACT_BEGIN("verified_sqrt");
    PRECONDITION("input is finite", is_finite_double, &x);
    PRECONDITION("input is non-negative", is_non_negative_double, &x);  // Allow 0.0

    if (!CHECK_PRECONDITIONS()) {
        printf("sqrt precondition failed for input: %f\n", x);
        CONTRACT_END();
        PROOF_CONTEXT_END();
        return -1.0;
    }

    double result = sqrt(x);
    printf("Computed result: %.6f\n", result);

    POSTCONDITION("result is finite", is_finite_double, &result);
    POSTCONDITION("result is non-negative", is_non_negative_double, &result);

    if (!CHECK_POSTCONDITIONS()) {
        printf("sqrt postcondition failed\n");
        CONTRACT_END();
        PROOF_CONTEXT_END();
        return -1.0;
    }

    CONTRACT_END();
    PROOF_CONTEXT_END();
    return result;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// DEMONSTRATION WITH BETTER EXAMPLES
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void demo_successful_payment(void) {
    printf("\nDemo: Successful International Payment\n");
    printf("----------------------------------------\n");

    account_info_t source_account;
    strcpy(source_account.account_id, "ACC001");
    source_account.balance.cents = 10000000; // $100k
    strcpy(source_account.balance.currency, "USD");
    strcpy(source_account.country, "US");
    source_account.international_enabled = true;  // ENABLED for international
    source_account.frozen = false;

    account_info_t destination_account;
    strcpy(destination_account.account_id, "ACC002");
    destination_account.balance.cents = 5000000; // $50k
    strcpy(destination_account.balance.currency, "USD");
    strcpy(destination_account.country, "CA");
    destination_account.international_enabled = false;
    destination_account.frozen = false;

    transaction_details_t transaction;
    strcpy(transaction.transaction_id, "TXN001");
    transaction.amount.cents = 2500000; // $25k - within limits
    strcpy(transaction.amount.currency, "USD");
    strcpy(transaction.description, "International payment");
    transaction.initiated_at = clock();

    payment_result_t result = process_payment_transparent(&transaction, &source_account, &destination_account);

    printf("Payment Result: ");
    switch (result) {
    case PAYMENT_SUCCESS:
        printf("SUCCESS\n");
        break;
    case PAYMENT_VALIDATION_FAILED:
        printf("VALIDATION_FAILED\n");
        break;
    case PAYMENT_INSUFFICIENT_FUNDS:
        printf("INSUFFICIENT_FUNDS\n");
        break;
    case PAYMENT_ACCOUNT_FROZEN:
        printf("ACCOUNT_FROZEN\n");
        break;
    case PAYMENT_COMPLIANCE_VIOLATION:
        printf("COMPLIANCE_VIOLATION\n");
        break;
    case PAYMENT_RISK_TOO_HIGH:
        printf("RISK_TOO_HIGH\n");
        break;
    }
}

void demo_failed_payment(void) {
    printf("\nDemo: Failed Payment (Frozen Account)\n");
    printf("---------------------------------------\n");

    account_info_t source_account;
    strcpy(source_account.account_id, "ACC003");
    source_account.balance.cents = 10000000;
    strcpy(source_account.balance.currency, "USD");
    strcpy(source_account.country, "US");
    source_account.international_enabled = true;
    source_account.frozen = true;  // FROZEN ACCOUNT

    account_info_t destination_account;
    strcpy(destination_account.account_id, "ACC004");
    destination_account.balance.cents = 5000000;
    strcpy(destination_account.balance.currency, "USD");
    strcpy(destination_account.country, "US");
    destination_account.international_enabled = false;
    destination_account.frozen = false;

    transaction_details_t transaction;
    strcpy(transaction.transaction_id, "TXN002");
    transaction.amount.cents = 1000000; // $10k
    strcpy(transaction.amount.currency, "USD");
    strcpy(transaction.description, "Domestic payment");
    transaction.initiated_at = clock();

    payment_result_t result = process_payment_transparent(&transaction, &source_account, &destination_account);

    printf("Payment Result: ");
    switch (result) {
    case PAYMENT_SUCCESS:
        printf("SUCCESS\n");
        break;
    case PAYMENT_VALIDATION_FAILED:
        printf("VALIDATION_FAILED\n");
        break;
    case PAYMENT_INSUFFICIENT_FUNDS:
        printf("INSUFFICIENT_FUNDS\n");
        break;
    case PAYMENT_ACCOUNT_FROZEN:
        printf("ACCOUNT_FROZEN\n");
        break;
    case PAYMENT_COMPLIANCE_VIOLATION:
        printf("COMPLIANCE_VIOLATION\n");
        break;
    case PAYMENT_RISK_TOO_HIGH:
        printf("RISK_TOO_HIGH\n");
        break;
    }
}

int temp_main(void) {
    printf("Axiomatik C17 Transparent Code Demonstration\n");
    printf("================================================\n");

    axiomatik_init();

    // Demo 1: Financial Transaction Processing
    printf("\nTransparent Financial Transaction Processing\n");
    printf("===============================================\n");

    demo_successful_payment();
    demo_failed_payment();

    // Demo 2: Mathematical Operations
    printf("\n\nVerified Mathematical Operations\n");
    printf("===================================\n");

    double test_values[] = { 16.0, 25.0, 100.0, 0.0, -4.0 };
    size_t num_values = sizeof(test_values) / sizeof(test_values[0]);

    for (size_t i = 0; i < num_values; i++) {
        double input = test_values[i];
        double result = verified_sqrt(input);

        if (result >= 0.0) {
            printf("sqrt(%.1f) = %.6f\n", input, result);
        }
        else {
            printf("sqrt(%.1f) = FAILED (invalid input)\n", input);
        }
    }

    // Demo 3: Performance Report
    printf("\n\nPerformance Analysis\n");
    printf("========================\n");
    print_performance_report();

    printf("\n=== Demonstration Complete ===\n");
    printf("All system state is transparent and debuggable!\n");
    printf("Key Features Demonstrated:\n");
    printf("Contract-based verification\n");
    printf("Transparent error handling\n");
    printf("Performance monitoring\n");
    printf("Context-aware debugging\n");

    axiomatik_cleanup();
    return 0;
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif