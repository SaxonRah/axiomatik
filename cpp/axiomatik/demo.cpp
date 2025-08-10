/**
 * demo.cpp - Demonstration of Transparent Code C++ Axiomatik
 */

#include "axiomatik.hpp"
#include <iostream>
#include <thread>
#include <chrono>

using namespace axiomatik;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// TRANSPARENT FINANCIAL TRANSACTION PROCESSING
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

struct MoneyAmount {
    int64_t cents;  // Avoid floating point precision issues
    std::string currency;

    MoneyAmount(int64_t cents_, std::string currency_)
        : cents(cents_), currency(std::move(currency_)) {
    }

    bool operator>=(const MoneyAmount& other) const {
        return currency == other.currency && cents >= other.cents;
    }

    MoneyAmount operator+(const MoneyAmount& other) const {
        require("same currency for addition", currency == other.currency);
        return MoneyAmount(cents + other.cents, currency);
    }
};

struct AccountInfo {
    std::string account_id;
    MoneyAmount balance;
    std::string country;
    bool international_enabled;
    bool frozen;

    AccountInfo(std::string id, MoneyAmount bal, std::string country_, bool intl = false)
        : account_id(std::move(id)), balance(bal), country(std::move(country_)),
        international_enabled(intl), frozen(false) {
    }
};

struct TransactionDetails {
    std::string transaction_id;
    MoneyAmount amount;
    std::string description;
    TimePoint initiated_at;

    TransactionDetails(std::string id, MoneyAmount amt, std::string desc)
        : transaction_id(std::move(id)), amount(amt), description(std::move(desc)),
        initiated_at(std::chrono::steady_clock::now()) {
    }
};

// Rich context structure - everything visible for debugging
struct PaymentProcessingContext {
    TransactionDetails transaction;
    AccountInfo source_account;
    AccountInfo destination_account;

    // Validation state - completely transparent
    std::vector<std::string> validation_errors;
    bool amount_validated = false;
    bool accounts_validated = false;
    bool compliance_checked = false;

    // Risk assessment - explicit factors
    struct RiskFactors {
        bool large_amount = false;
        bool cross_border = false;
        bool new_account = false;
        bool unusual_time = false;
        int total_risk_score = 0;
    } risk_factors;

    // Fee calculation - transparent breakdown
    struct FeeBreakdown {
        MoneyAmount base_fee{ 0, "USD" };
        MoneyAmount cross_border_fee{ 0, "USD" };
        MoneyAmount risk_premium{ 0, "USD" };
        MoneyAmount total{ 0, "USD" };
    } fees;

    // Audit trail - complete visibility
    std::vector<std::string> processing_steps;

    // Timing information
    TimePoint started_at;
    Duration total_processing_time{ 0 };

    PaymentProcessingContext(TransactionDetails trans, AccountInfo src, AccountInfo dest)
        : transaction(std::move(trans)), source_account(std::move(src)), destination_account(std::move(dest)),
        started_at(std::chrono::steady_clock::now()) {
    }
};

enum class PaymentResult {
    SUCCESS,
    VALIDATION_FAILED,
    INSUFFICIENT_FUNDS,
    ACCOUNT_FROZEN,
    COMPLIANCE_VIOLATION,
    RISK_TOO_HIGH
};

// Transparent validation functions - explicit business rules
void validate_transaction_amount(PaymentProcessingContext& ctx) {
    PROOF_CONTEXT("amount_validation");

    ctx.processing_steps.push_back("Starting amount validation");

    require("amount is positive", ctx.transaction.amount.cents > 0);
    require("amount is reasonable", ctx.transaction.amount.cents <= 100000000); // $1M limit

    ctx.amount_validated = true;
    ctx.processing_steps.push_back("Amount validation completed");
}

void validate_accounts(PaymentProcessingContext& ctx) {
    PROOF_CONTEXT("account_validation");

    ctx.processing_steps.push_back("Starting account validation");

    require("source account not frozen", !ctx.source_account.frozen);
    require("destination account not frozen", !ctx.destination_account.frozen);
    require("different accounts", ctx.source_account.account_id != ctx.destination_account.account_id);

    // Cross-border transaction validation
    if (ctx.source_account.country != ctx.destination_account.country) {
        require("international transfers enabled", ctx.source_account.international_enabled);
        ctx.risk_factors.cross_border = true;
    }

    ctx.accounts_validated = true;
    ctx.processing_steps.push_back("Account validation completed");
}

void assess_risk(PaymentProcessingContext& ctx) {
    PROOF_CONTEXT("risk_assessment");

    ctx.processing_steps.push_back("Starting risk assessment");

    // Explicit risk scoring - every factor visible
    ctx.risk_factors.total_risk_score = 0;

    if (ctx.transaction.amount.cents > 5000000) { // $50k
        ctx.risk_factors.large_amount = true;
        ctx.risk_factors.total_risk_score += 25;
    }

    if (ctx.risk_factors.cross_border) {
        ctx.risk_factors.total_risk_score += 15;
    }

    // Time-based risk (using safe localtime_s for Windows compatibility)
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    struct tm tm_buf;

#ifdef _WIN32
    localtime_s(&tm_buf, &time_t);
#else
    localtime_r(&time_t, &tm_buf);
#endif

    if (tm_buf.tm_hour < 6 || tm_buf.tm_hour > 22) {
        ctx.risk_factors.unusual_time = true;
        ctx.risk_factors.total_risk_score += 10;
    }

    ctx.processing_steps.push_back("Risk assessment completed, score: " +
        std::to_string(ctx.risk_factors.total_risk_score));
}

void calculate_fees(PaymentProcessingContext& ctx) {
    PROOF_CONTEXT("fee_calculation");

    ctx.processing_steps.push_back("Starting fee calculation");

    // Base fee: 0.5% of transaction amount
    ctx.fees.base_fee = MoneyAmount(ctx.transaction.amount.cents / 200, ctx.transaction.amount.currency);

    // Cross-border fee: additional 1%
    if (ctx.risk_factors.cross_border) {
        ctx.fees.cross_border_fee = MoneyAmount(ctx.transaction.amount.cents / 100, ctx.transaction.amount.currency);
    }

    // Risk premium: 0.1% for each 10 risk points
    if (ctx.risk_factors.total_risk_score > 20) {
        int premium_basis_points = (ctx.risk_factors.total_risk_score - 20) * 10;
        ctx.fees.risk_premium = MoneyAmount(
            (ctx.transaction.amount.cents * premium_basis_points) / 10000,
            ctx.transaction.amount.currency
        );
    }

    // Calculate total
    ctx.fees.total = ctx.fees.base_fee + ctx.fees.cross_border_fee + ctx.fees.risk_premium;

    // Fix: Convert cents to dollars for display consistency
    ctx.processing_steps.push_back("Fee calculation completed, total: $" +
        std::to_string(ctx.fees.total.cents / 100.0) + " " + ctx.fees.total.currency);
}

// Main processing function - transparent workflow with explicit business logic
PaymentResult process_payment_transparent(PaymentProcessingContext& ctx) {
    PROOF_CONTEXT("payment_processing");

    auto start_time = std::chrono::steady_clock::now();

    // Contract for the entire payment processing function
    CONTRACT("process_payment");
    REQUIRES("valid transaction ID", !ctx.transaction.transaction_id.empty());
    REQUIRES("positive amount", ctx.transaction.amount.cents > 0);
    REQUIRES("valid source account", !ctx.source_account.account_id.empty());
    REQUIRES("valid destination account", !ctx.destination_account.account_id.empty());

    CHECK_PRECONDITIONS();

    try {
        // Step 1: Amount validation
        validate_transaction_amount(ctx);

        // Step 2: Account validation  
        validate_accounts(ctx);

        // Step 3: Risk assessment
        assess_risk(ctx);

        // Step 4: Fee calculation
        calculate_fees(ctx);

        // Step 5: Sufficient funds check
        MoneyAmount total_required = ctx.transaction.amount + ctx.fees.total;
        require("sufficient funds", ctx.source_account.balance >= total_required);

        // Step 6: Risk threshold check
        require("acceptable risk level", ctx.risk_factors.total_risk_score <= 50);

        auto end_time = std::chrono::steady_clock::now();
        ctx.total_processing_time = std::chrono::duration_cast<Duration>(end_time - start_time);

        ctx.processing_steps.push_back("Payment processing completed successfully in " +
            std::to_string(ctx.total_processing_time.count()) + "microseconds");

        // Postconditions - now processing time is guaranteed > 0
        ENSURES("all validation completed", ctx.amount_validated && ctx.accounts_validated);
        ENSURES("fees calculated", ctx.fees.total.cents > 0);
        ENSURES("processing time recorded", ctx.total_processing_time.count() > 0);

        CHECK_POSTCONDITIONS();

        return PaymentResult::SUCCESS;

    }
    catch (const ProofFailure& failure) {
        auto end_time = std::chrono::steady_clock::now();
        ctx.total_processing_time = std::chrono::duration_cast<Duration>(end_time - start_time);

        ctx.processing_steps.push_back("Processing failed: " + failure.claim);

        // Explicit error classification based on context
        if (failure.context.find("amount_validation") != std::string::npos) {
            return PaymentResult::VALIDATION_FAILED;
        }
        else if (failure.context.find("account_validation") != std::string::npos) {
            if (failure.claim.find("frozen") != std::string::npos) {
                return PaymentResult::ACCOUNT_FROZEN;
            }
            return PaymentResult::VALIDATION_FAILED;
        }
        else if (failure.claim.find("sufficient funds") != std::string::npos) {
            return PaymentResult::INSUFFICIENT_FUNDS;
        }
        else if (failure.claim.find("risk level") != std::string::npos) {
            return PaymentResult::RISK_TOO_HIGH;
        }
        else {
            return PaymentResult::COMPLIANCE_VIOLATION;
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// PROTOCOL DEMONSTRATION - FILE-LIKE STATE MACHINE
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

struct FileHandle {
    std::string filename;
    std::string content;

    explicit FileHandle(std::string name) : filename(std::move(name)) {}
};

// Global protocol for file operations
Protocol file_protocol("FileProtocol", "closed");

void init_file_protocol() {
    file_protocol.add_state(ProtocolState("closed", { "open" }));
    file_protocol.add_state(ProtocolState("open", { "read", "write", "closed" }));
    file_protocol.add_state(ProtocolState("read", { "read", "write", "closed" }));
    file_protocol.add_state(ProtocolState("write", { "read", "write", "closed" }));
}

void file_open(FileHandle& file) {
    file_protocol.verify_transition(&file, "open");
    std::cout << "File " << file.filename << " opened\n";
}

std::string file_read(FileHandle& file) {
    file_protocol.verify_transition(&file, "read");
    std::cout << "Reading from file " << file.filename << "\n";
    return file.content;
}

void file_write(FileHandle& file, const std::string& data) {
    file_protocol.verify_transition(&file, "write");
    file.content += data;
    std::cout << "Writing to file " << file.filename << ": " << data << "\n";
}

void file_close(FileHandle& file) {
    file_protocol.verify_transition(&file, "closed");
    std::cout << "File " << file.filename << " closed\n";
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// MAIN DEMONSTRATION
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

int temp_main() {
    std::cout << "Axiomatik C++ Transparent Code Demonstration\n";
    std::cout << "============================================\n\n";

    // Initialize protocols
    init_file_protocol();

    // Demo 1: Transparent Financial Transaction Processing
    std::cout << "1. Transparent Financial Transaction Processing\n";
    std::cout << "-----------------------------------------------\n";

    // Create test accounts and transaction
    AccountInfo source_account("ACC001", MoneyAmount(10000000, "USD"), "US", true); // $100k
    AccountInfo destination_account("ACC002", MoneyAmount(5000000, "USD"), "CA", false); // $50k  
    TransactionDetails transaction("TXN001", MoneyAmount(2500000, "USD"), "International payment"); // $25k

    PaymentProcessingContext payment_ctx(transaction, source_account, destination_account);

    auto result = process_payment_transparent(payment_ctx);

    std::cout << "Payment Result: ";
    switch (result) {
    case PaymentResult::SUCCESS:
        std::cout << "SUCCESS";
        break;
    case PaymentResult::VALIDATION_FAILED:
        std::cout << "VALIDATION_FAILED";
        break;
    case PaymentResult::INSUFFICIENT_FUNDS:
        std::cout << "INSUFFICIENT_FUNDS";
        break;
    case PaymentResult::ACCOUNT_FROZEN:
        std::cout << "ACCOUNT_FROZEN";
        break;
    case PaymentResult::COMPLIANCE_VIOLATION:
        std::cout << "COMPLIANCE_VIOLATION";
        break;
    case PaymentResult::RISK_TOO_HIGH:
        std::cout << "RISK_TOO_HIGH";
        break;
    }
    std::cout << "\n\n";

    // Show complete processing context
    std::cout << "Complete Processing Context (Transparent Debugging):\n";
    std::cout << "Transaction ID: " << payment_ctx.transaction.transaction_id << "\n";
    std::cout << "Amount: $" << (payment_ctx.transaction.amount.cents / 100.0) << "\n";
    std::cout << "Risk Score: " << payment_ctx.risk_factors.total_risk_score << "\n";
    std::cout << "Total Fees: $" << (payment_ctx.fees.total.cents / 100.0) << "\n";
    std::cout << "Processing Time: " << payment_ctx.total_processing_time.count() << "microseconds\n";

    std::cout << "\nAudit Trail:\n";
    for (const auto& step : payment_ctx.processing_steps) {
        std::cout << "  - " << step << "\n";
    }
    std::cout << "\n";

    // Demo 2: Protocol State Machine
    std::cout << "2. Protocol State Machine Verification\n";
    std::cout << "--------------------------------------\n";

    FileHandle file("test.txt");
    std::cout << "Initial file state: " << file_protocol.get_state(&file) << "\n";

    try {
        file_open(file);
        std::cout << "Current state: " << file_protocol.get_state(&file) << "\n";

        file_write(file, "Hello, ");
        file_write(file, "Transparent Code!");
        std::cout << "Current state: " << file_protocol.get_state(&file) << "\n";

        auto content = file_read(file);
        std::cout << "Read content: " << content << "\n";
        std::cout << "Current state: " << file_protocol.get_state(&file) << "\n";

        file_close(file);
        std::cout << "Final state: " << file_protocol.get_state(&file) << "\n";

        // Show transition history
        auto history = file_protocol.get_transition_history(&file);
        std::cout << "Transition history: ";
        for (const auto& state : history) {
            std::cout << state << " -> ";
        }
        std::cout << "END\n\n";

    }
    catch (const ProofFailure& failure) {
        std::cout << "Protocol violation: " << failure.claim << " in context " << failure.context << "\n\n";
    }

    // Demo 3: Refinement Types
    std::cout << "3. Refinement Types Validation\n";
    std::cout << "------------------------------\n";

    try {
        auto positive_num = make_positive_int(42);
        std::cout << "Valid positive number: " << positive_num.get() << "\n";

        auto percentage = make_percentage(85);
        std::cout << "Valid percentage: " << percentage.get() << "%\n";

        auto non_empty_str = make_non_empty_string("Hello, World!");
        std::cout << "Valid non-empty string: '" << non_empty_str.get() << "'\n";

        // This will fail validation
        std::cout << "\nTrying invalid values:\n";
        auto invalid_positive = make_positive_int(-5);
        std::cout << "This should not print: " << invalid_positive.get() << "\n";

    }
    catch (const ProofFailure& failure) {
        std::cout << "Caught validation failure: " << failure.claim << "\n";
    }
    std::cout << "\n";

    // Demo 4: Information Flow Tracking
    std::cout << "4. Information Flow Tracking\n";
    std::cout << "----------------------------\n";

    TaintedValue public_data("Public information", SecurityLabel::PUBLIC, { "web_form" });
    TaintedValue secret_data("Classified information", SecurityLabel::SECRET, { "database" });

    InformationFlowTracker flow_tracker;
    flow_tracker.add_policy(SecurityLabel::PUBLIC, SecurityLabel::CONFIDENTIAL, true, "Public can flow to Confidential");
    flow_tracker.add_policy(SecurityLabel::SECRET, SecurityLabel::PUBLIC, false, "Secret cannot flow to Public");

    try {
        std::cout << "Attempting public -> confidential flow: ";
        flow_tracker.track_flow(public_data, SecurityLabel::CONFIDENTIAL);
        std::cout << "ALLOWED\n";

        std::cout << "Attempting secret -> public flow: ";
        flow_tracker.track_flow(secret_data, SecurityLabel::PUBLIC);
        std::cout << "This should not print\n";

    }
    catch (const ProofFailure& failure) {
        std::cout << "DENIED - " << failure.claim << "\n";
    }

    // Demo declassification
    try {
        std::cout << "Declassifying secret data: ";
        secret_data.declassify(SecurityLabel::CONFIDENTIAL, "Sanitized for internal use");
        std::cout << "SUCCESS\n";
        std::cout << "New security level: " << static_cast<int>(secret_data.label) << "\n";

    }
    catch (const ProofFailure& failure) {
        std::cout << "FAILED - " << failure.claim << "\n";
    }
    std::cout << "\n";

    // Demo 5: Temporal Properties
    std::cout << "5. Temporal Properties Verification\n";
    std::cout << "-----------------------------------\n";

    TemporalVerifier temporal_verifier;

    // Add "eventually consistent" property
    auto eventually_prop = std::make_unique<EventuallyProperty>(
        "data_sync",
        [](const std::deque<TemporalEvent>& history) {
            return std::any_of(history.begin(), history.end(),
                [](const TemporalEvent& e) { return e.event_name == "sync_complete"; });
        },
        Duration(5000) // 5 second timeout
    );

    // Add "always valid" property  
    auto always_prop = std::make_unique<AlwaysProperty>(
        "data_valid",
        [](const TemporalEvent& event) {
            return event.data != "invalid";
        }
    );

    temporal_verifier.add_property(std::move(eventually_prop));
    temporal_verifier.add_property(std::move(always_prop));

    // Record some events
    temporal_verifier.record_event("data_update", "valid");
    temporal_verifier.record_event("validation_check", "valid");
    temporal_verifier.record_event("sync_start", "valid");
    temporal_verifier.record_event("sync_complete", "valid");

    try {
        temporal_verifier.verify_all();
        std::cout << "All temporal properties verified successfully\n";
    }
    catch (const ProofFailure& failure) {
        std::cout << "Temporal verification failed: " << failure.claim << "\n";
    }
    std::cout << "\n";

    // Demo 6: Performance Report
    std::cout << "6. Performance Analysis\n";
    std::cout << "----------------------\n";
    print_performance_report();

    std::cout << "\n=== Demonstration Complete ===\n";
    std::cout << "All system state is transparent and debuggable!\n";

    return 0;
}