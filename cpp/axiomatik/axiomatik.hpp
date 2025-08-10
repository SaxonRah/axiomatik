/**
 * Axiomatik C++: Transparent Runtime Verification
 * A functional/procedural verification system with complete state visibility
 */

#pragma once

#include <vector>
#include <array>
#include <string>
#include <unordered_map>
#include <deque>
#include <chrono>
#include <functional>
#include <thread>
#include <mutex>
#include <optional>
#include <variant>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <memory>
#include <cstring>
#include <type_traits>

namespace axiomatik {

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CORE TYPES AND CONFIGURATION
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    using TimePoint = std::chrono::steady_clock::time_point;
    using Duration = std::chrono::microseconds;
    using ThreadId = std::thread::id;

    using Evidence = std::variant<
        bool,
        int,
        double,
        std::string,
        std::function<bool()>
    >;

    enum class VerificationLevel {
        OFF,
        CONTRACTS,
        INVARIANTS,
        FULL,
        DEBUG
    };

    enum class SecurityLabel {
        PUBLIC,
        CONFIDENTIAL,
        SECRET,
        TOP_SECRET
    };

    struct GlobalConfig {
        VerificationLevel level = VerificationLevel::FULL;
        bool cache_enabled = true;
        size_t max_proof_steps = 10000;
        bool performance_mode = false;
        bool debug_mode = false;

        bool should_verify(const std::string& context_type = "") const {
            if (level == VerificationLevel::OFF) return false;
            if (level == VerificationLevel::CONTRACTS && context_type != "contract") return false;
            return true;
        }
    };

    // Global configuration - declared as extern in header
    extern GlobalConfig global_config;

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // PROOF FAILURE AND EVIDENCE TYPES
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    struct ProofFailure {
        std::string claim;
        std::string context;
        std::string details;
        TimePoint timestamp;
        ThreadId thread_id;

        ProofFailure(const std::string& claim_, const std::string& context_, const std::string& details_ = "")
            : claim(claim_), context(context_), details(details_),
            timestamp(std::chrono::steady_clock::now()),
            thread_id(std::this_thread::get_id()) {
        }
    };

    struct ProofStep {
        std::string claim;
        std::string context;
        TimePoint timestamp;
        ThreadId thread_id;
        bool succeeded;
        Duration verification_time;

        ProofStep(const std::string& claim_, const std::string& context_, bool succeeded_, Duration time_)
            : claim(claim_), context(context_), succeeded(succeeded_), verification_time(time_),
            timestamp(std::chrono::steady_clock::now()),
            thread_id(std::this_thread::get_id()) {
        }
    };

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // PROOF CACHE - TRANSPARENT PERFORMANCE OPTIMIZATION
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    struct CacheEntry {
        bool result;
        TimePoint last_access;
        size_t access_count;

        CacheEntry(bool result_)
            : result(result_), last_access(std::chrono::steady_clock::now()), access_count(1) {
        }
    };

    struct ProofCache {
        std::unordered_map<std::string, CacheEntry> entries;
        size_t max_size;
        size_t hit_count;
        size_t miss_count;

        explicit ProofCache(size_t max_size_ = 1000)
            : max_size(max_size_), hit_count(0), miss_count(0) {
        }

        std::optional<bool> get(const std::string& key) {
            auto it = entries.find(key);
            if (it != entries.end()) {
                it->second.last_access = std::chrono::steady_clock::now();
                it->second.access_count++;
                hit_count++;
                return it->second.result;
            }
            miss_count++;
            return std::nullopt;
        }

        void set(const std::string& key, bool value) {
            if (entries.size() >= max_size) {
                evict_oldest();
            }
            entries.emplace(key, CacheEntry(value));
        }

        void evict_oldest() {
            if (entries.empty()) return;

            auto oldest = std::min_element(entries.begin(), entries.end(),
                [](const auto& a, const auto& b) {
                    return a.second.last_access < b.second.last_access;
                });
            entries.erase(oldest);
        }

        void clear() {
            entries.clear();
            hit_count = 0;
            miss_count = 0;
        }

        double hit_rate() const {
            size_t total = hit_count + miss_count;
            return total > 0 ? static_cast<double>(hit_count) / total : 0.0;
        }
    };

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CORE PROOF SYSTEM - RICH VISIBLE STATE
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    struct ProofSystem {
        std::vector<ProofStep> steps;
        std::vector<std::string> context_stack;
        ProofCache cache;
        mutable std::mutex lock;  // Made mutable for const methods

        // Performance metrics - completely visible
        size_t total_verifications = 0;
        Duration total_verification_time{ 0 };
        std::unordered_map<std::string, size_t> context_counts;
        std::unordered_map<ThreadId, size_t> thread_counts;

        ProofSystem() : cache(global_config.cache_enabled ? 1000 : 0) {}

        // Clear all state - explicit and visible
        void clear() {
            std::lock_guard<std::mutex> guard(lock);
            steps.clear();
            context_stack.clear();
            cache.clear();
            total_verifications = 0;
            total_verification_time = Duration{ 0 };
            context_counts.clear();
            thread_counts.clear();
        }

        void push_context(const std::string& context) {
            std::lock_guard<std::mutex> guard(lock);
            context_stack.push_back(context);
        }

        void pop_context() {
            std::lock_guard<std::mutex> guard(lock);
            if (!context_stack.empty()) {
                context_stack.pop_back();
            }
        }

        std::string current_context() const {
            std::lock_guard<std::mutex> guard(lock);
            return context_stack.empty() ? "global" : context_stack.back();
        }
    };

    // Global proof system - declared as extern in header
    extern ProofSystem global_proof_system;

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // EVIDENCE EVALUATION - EXPLICIT AND TRACEABLE
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    inline bool evaluate_evidence(const Evidence& evidence) {
        return std::visit([](const auto& value) -> bool {
            using T = std::decay_t<decltype(value)>;

            if constexpr (std::is_same_v<T, bool>) {
                return value;
            }
            else if constexpr (std::is_same_v<T, std::function<bool()>>) {
                return value();
            }
            else if constexpr (std::is_same_v<T, int>) {
                return value != 0;
            }
            else if constexpr (std::is_same_v<T, double>) {
                return value != 0.0;
            }
            else if constexpr (std::is_same_v<T, std::string>) {
                return !value.empty();
            }
            else {
                return false;
            }
            }, evidence);
    }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CORE REQUIRE FUNCTION - EXPLICIT VERIFICATION WITH FULL CONTEXT
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CORE REQUIRE FUNCTION - SINGLE OVERLOAD TO AVOID AMBIGUITY
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    template<typename T>
    void require(const std::string& claim, T&& evidence) {
        if (!global_config.should_verify()) {
            return;
        }

        auto start_time = std::chrono::steady_clock::now();
        bool result = false;

        // Evaluate evidence based on type
        if constexpr (std::is_same_v<std::decay_t<T>, bool>) {
            result = evidence;
        }
        else if constexpr (std::is_invocable_r_v<bool, T>) {
            result = evidence();
        }
        else if constexpr (std::is_arithmetic_v<std::decay_t<T>>) {
            result = static_cast<bool>(evidence);
        }
        else if constexpr (std::is_same_v<std::decay_t<T>, std::string>) {
            result = !evidence.empty();
        }
        else if constexpr (std::is_same_v<std::decay_t<T>, Evidence>) {
            result = evaluate_evidence(evidence);
        }
        else {
            result = static_cast<bool>(evidence);
        }

        auto end_time = std::chrono::steady_clock::now();
        auto verification_time = std::chrono::duration_cast<Duration>(end_time - start_time);

        // Get context BEFORE acquiring the lock
        std::string context = global_proof_system.current_context();
        ThreadId thread_id = std::this_thread::get_id();

        // Record proof step with complete context
        {
            std::lock_guard<std::mutex> guard(global_proof_system.lock);

            if (global_proof_system.steps.size() < global_config.max_proof_steps) {
                global_proof_system.steps.emplace_back(
                    claim,
                    context,  // Use the pre-fetched context
                    result,
                    verification_time
                );
            }

            // Update performance metrics
            global_proof_system.total_verifications++;
            global_proof_system.total_verification_time += verification_time;
            global_proof_system.context_counts[context]++;  // Use pre-fetched context
            global_proof_system.thread_counts[thread_id]++;
        }

        if (!result) {
            throw ProofFailure(claim, context);  // Use pre-fetched context
        }
    }

    // Convenience overloads for common cases
    inline void require(const std::string& claim, bool condition) {
        Evidence evidence = condition;
        require(claim, evidence);
    }

    inline void require(const std::string& claim, const std::function<bool()>& predicate) {
        Evidence evidence = predicate;
        require(claim, evidence);
    }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // PROOF CONTEXT MANAGEMENT - EXPLICIT SCOPING
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    struct ProofContext {
        std::string name;
        bool active;

        explicit ProofContext(const std::string& name_) : name(name_), active(true) {
            if (global_config.should_verify()) {
                global_proof_system.push_context(name);
            }
        }

        ~ProofContext() {
            if (active && global_config.should_verify()) {
                global_proof_system.pop_context();
            }
        }

        // Move constructor
        ProofContext(ProofContext&& other) noexcept : name(std::move(other.name)), active(other.active) {
            other.active = false;
        }

        // Disable copy
        ProofContext(const ProofContext&) = delete;
        ProofContext& operator=(const ProofContext&) = delete;
        ProofContext& operator=(ProofContext&&) = delete;
    };

    // Macro for convenient context creation
    #define PROOF_CONTEXT(name) ProofContext _ctx(name)

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // REFINEMENT TYPES - COMPILE-TIME AND RUNTIME VALIDATION
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    template<typename T>
    struct RefinementConstraint {
        std::function<bool(const T&)> predicate;
        std::string description;

        RefinementConstraint(std::function<bool(const T&)> pred, const std::string& desc)
            : predicate(std::move(pred)), description(desc) {
        }
    };

    template<typename T>
    struct RefinedValue {
        T value;
        std::vector<RefinementConstraint<T>> constraints;
        bool validated;

        explicit RefinedValue(const T& val) : value(val), validated(false) {}

        RefinedValue& add_constraint(const RefinementConstraint<T>& constraint) {
            constraints.push_back(constraint);
            validated = false;
            return *this;
        }

        void validate() {
            PROOF_CONTEXT("refinement_validation");

            for (const auto& constraint : constraints) {
                require("refinement constraint: " + constraint.description,
                    [&]() { return constraint.predicate(value); });
            }
            validated = true;
        }

        const T& get() {
            if (!validated) {
                validate();
            }
            return value;
        }

        operator const T& () {
            return get();
        }
    };

    // Common refinement types
    using PositiveInt = RefinedValue<int>;
    using NonEmptyString = RefinedValue<std::string>;
    using Percentage = RefinedValue<int>;

    // Factory functions for common constraints
    inline PositiveInt make_positive_int(int value) {
        PositiveInt result(value);
        result.add_constraint(RefinementConstraint<int>(
            [](int x) { return x > 0; },
            "must be positive"
        ));
        return result;
    }

    inline NonEmptyString make_non_empty_string(const std::string& value) {
        NonEmptyString result(value);
        result.add_constraint(RefinementConstraint<std::string>(
            [](const std::string& s) { return !s.empty(); },
            "must not be empty"
        ));
        return result;
    }

    inline Percentage make_percentage(int value) {
        Percentage result(value);
        result.add_constraint(RefinementConstraint<int>(
            [](int x) { return x >= 0 && x <= 100; },
            "must be between 0 and 100"
        ));
        return result;
    }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // PROTOCOL STATE MACHINE - EXPLICIT STATE TRANSITIONS
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    struct ProtocolState {
        std::string name;
        std::vector<std::string> allowed_transitions;

        ProtocolState() = default;

        ProtocolState(const std::string& name_, std::vector<std::string> transitions = {})
            : name(name_), allowed_transitions(std::move(transitions)) {
        }
    };

    struct ProtocolInstance {
        std::string current_state;
        std::vector<std::string> transition_history;
        TimePoint last_transition;

        ProtocolInstance() : last_transition(std::chrono::steady_clock::now()) {}

        explicit ProtocolInstance(const std::string& initial_state)
            : current_state(initial_state), last_transition(std::chrono::steady_clock::now()) {
        }
    };

    struct Protocol {
        std::string name;
        std::string initial_state;
        std::unordered_map<std::string, ProtocolState> states;
        std::unordered_map<void*, ProtocolInstance> instances;  // Object pointer -> state

        Protocol(const std::string& name_, const std::string& initial_state_)
            : name(name_), initial_state(initial_state_) {
        }

        void add_state(const ProtocolState& state) {
            states[state.name] = state;
        }

        void verify_transition(void* object, const std::string& new_state) {
            PROOF_CONTEXT("protocol_" + name);

            auto it = instances.find(object);
            std::string current = (it != instances.end()) ? it->second.current_state : initial_state;

            require("target state exists", states.find(new_state) != states.end());

            auto current_state_it = states.find(current);
            require("current state exists", current_state_it != states.end());

            const auto& allowed = current_state_it->second.allowed_transitions;
            require("transition is allowed",
                std::find(allowed.begin(), allowed.end(), new_state) != allowed.end());

            // Update state
            if (it != instances.end()) {
                it->second.current_state = new_state;
                it->second.transition_history.push_back(new_state);
                it->second.last_transition = std::chrono::steady_clock::now();
            }
            else {
                ProtocolInstance instance(new_state);
                instance.transition_history.push_back(new_state);
                instances[object] = std::move(instance);
            }
        }

        std::string get_state(void* object) const {
            auto it = instances.find(object);
            return (it != instances.end()) ? it->second.current_state : initial_state;
        }

        std::vector<std::string> get_transition_history(void* object) const {
            auto it = instances.find(object);
            return (it != instances.end()) ? it->second.transition_history : std::vector<std::string>{};
        }
    };

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // INFORMATION FLOW TRACKING - EXPLICIT SECURITY BOUNDARIES
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    struct TaintedValue {
        std::string data;  // Could be templated for different types
        SecurityLabel label;
        std::vector<std::string> provenance;
        TimePoint created_at;

        TaintedValue(std::string data_, SecurityLabel label_, std::vector<std::string> prov = {})
            : data(std::move(data_)), label(label_), provenance(std::move(prov)),
            created_at(std::chrono::steady_clock::now()) {
        }

        bool can_flow_to(SecurityLabel target_label) const {
            return static_cast<int>(label) <= static_cast<int>(target_label);
        }

        void declassify(SecurityLabel new_label, const std::string& justification) {
            PROOF_CONTEXT("information_flow_declassify");

            require("declassification has justification", !justification.empty());
            require("declassification reduces security level",
                static_cast<int>(new_label) < static_cast<int>(label));

            label = new_label;
            provenance.push_back("declassified: " + justification);
        }

        TaintedValue combine_with(const TaintedValue& other) const {
            SecurityLabel max_label = (static_cast<int>(label) > static_cast<int>(other.label)) ? label : other.label;

            std::vector<std::string> combined_prov = provenance;
            combined_prov.insert(combined_prov.end(), other.provenance.begin(), other.provenance.end());

            return TaintedValue(data + "+" + other.data, max_label, combined_prov);
        }
    };

    struct InformationFlowPolicy {
        SecurityLabel from;
        SecurityLabel to;
        bool allowed;
        std::string reason;
    };

    struct InformationFlowTracker {
        std::vector<InformationFlowPolicy> policies;
        std::vector<TaintedValue> flow_history;

        void add_policy(SecurityLabel from, SecurityLabel to, bool allowed, const std::string& reason = "") {
            policies.push_back({ from, to, allowed, reason });
        }

        void track_flow(const TaintedValue& source, SecurityLabel target_label) {
            PROOF_CONTEXT("information_flow_track");

            // Check explicit policies first
            for (const auto& policy : policies) {
                if (policy.from == source.label && policy.to == target_label) {
                    require("flow allowed by policy: " + policy.reason, policy.allowed);
                    return;
                }
            }

            // Default policy: can only flow to equal or higher security levels
            require("flow satisfies default policy", source.can_flow_to(target_label));
        }
    };

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // TEMPORAL PROPERTIES - EXPLICIT TIME-BASED VERIFICATION
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    struct TemporalEvent {
        std::string event_name;
        std::string data;
        TimePoint timestamp;
        ThreadId thread_id;

        TemporalEvent(std::string name, std::string data_ = "")
            : event_name(std::move(name)), data(std::move(data_)),
            timestamp(std::chrono::steady_clock::now()),
            thread_id(std::this_thread::get_id()) {
        }
    };

    struct TemporalProperty {
        std::string name;
        std::deque<TemporalEvent> history;
        size_t max_history_size;

        explicit TemporalProperty(std::string name_, size_t max_history = 1000)
            : name(std::move(name_)), max_history_size(max_history) {
        }

        void record_event(const std::string& event_name, const std::string& data = "") {
            history.emplace_back(event_name, data);
            while (history.size() > max_history_size) {
                history.pop_front();
            }
        }

        virtual bool check() = 0;
        virtual ~TemporalProperty() = default;
    };

    struct EventuallyProperty : public TemporalProperty {
        std::function<bool(const std::deque<TemporalEvent>&)> condition;
        Duration timeout;
        TimePoint start_time;

        EventuallyProperty(std::string name, std::function<bool(const std::deque<TemporalEvent>&)> cond, Duration timeout_)
            : TemporalProperty(std::move(name)), condition(std::move(cond)), timeout(timeout_),
            start_time(std::chrono::steady_clock::now()) {
        }

        bool check() override {
            if (condition(history)) {
                return true;
            }

            auto now = std::chrono::steady_clock::now();
            if (now - start_time > timeout) {
                throw ProofFailure("temporal_eventually_timeout",
                    "Eventually property '" + name + "' timed out",
                    "Timeout after " + std::to_string(timeout.count()) + "ms");
            }

            return false;
        }
    };

    struct AlwaysProperty : public TemporalProperty {
        std::function<bool(const TemporalEvent&)> invariant;

        AlwaysProperty(std::string name, std::function<bool(const TemporalEvent&)> inv)
            : TemporalProperty(std::move(name)), invariant(std::move(inv)) {
        }

        bool check() override {
            for (const auto& event : history) {
                if (!invariant(event)) {
                    return false;
                }
            }
            return true;
        }
    };

    struct TemporalVerifier {
        std::vector<std::unique_ptr<TemporalProperty>> properties;

        void add_property(std::unique_ptr<TemporalProperty> prop) {
            properties.push_back(std::move(prop));
        }

        void record_event(const std::string& event_name, const std::string& data = "") {
            for (auto& prop : properties) {
                prop->record_event(event_name, data);
            }
        }

        void verify_all() {
            PROOF_CONTEXT("temporal_verification");

            for (const auto& prop : properties) {
                require("temporal property '" + prop->name + "' holds",
                    [&]() { return prop->check(); });
            }
        }
    };

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // PERFORMANCE ANALYSIS - TRANSPARENT METRICS
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    struct PerformanceMetrics {
        size_t total_verifications;
        Duration total_time;
        Duration average_time;
        Duration min_time;
        Duration max_time;
        double cache_hit_rate;
        std::unordered_map<std::string, size_t> context_breakdown;
        std::unordered_map<ThreadId, size_t> thread_breakdown;

        PerformanceMetrics() : total_verifications(0), total_time(0), average_time(0),
            min_time(Duration::max()), max_time(Duration::zero()),
            cache_hit_rate(0.0) {
        }
    };

    inline PerformanceMetrics get_performance_metrics() {
        std::lock_guard<std::mutex> guard(global_proof_system.lock);

        PerformanceMetrics metrics;
        metrics.total_verifications = global_proof_system.total_verifications;
        metrics.total_time = global_proof_system.total_verification_time;
        metrics.context_breakdown = global_proof_system.context_counts;
        metrics.thread_breakdown = global_proof_system.thread_counts;
        metrics.cache_hit_rate = global_proof_system.cache.hit_rate();

        if (!global_proof_system.steps.empty()) {
            auto times = std::vector<Duration>();
            for (const auto& step : global_proof_system.steps) {
                times.push_back(step.verification_time);
            }

            metrics.min_time = *std::min_element(times.begin(), times.end());
            metrics.max_time = *std::max_element(times.begin(), times.end());

            if (metrics.total_verifications > 0) {
                metrics.average_time = Duration(metrics.total_time.count() / metrics.total_verifications);
            }
        }

        return metrics;
    }

    inline void print_performance_report() {
        auto metrics = get_performance_metrics();

        std::cout << "Axiomatik C++ Performance Report\n";
        std::cout << "================================\n";
        std::cout << "Total verifications: " << metrics.total_verifications << "\n";
        std::cout << "Total time: " << metrics.total_time.count() << "microseconds\n";
        std::cout << "Average time: " << metrics.average_time.count() << "microseconds\n";
        std::cout << "Min time: " << metrics.min_time.count() << "microseconds\n";
        std::cout << "Max time: " << metrics.max_time.count() << "microseconds\n";
        std::cout << "Cache hit rate: " << (metrics.cache_hit_rate * 100) << "%\n";

        std::cout << "\nContext breakdown:\n";
        for (const auto& [context, count] : metrics.context_breakdown) {
            std::cout << "  " << context << ": " << count << " verifications\n";
        }

        std::cout << "\nThread breakdown:\n";
        for (const auto& [thread_id, count] : metrics.thread_breakdown) {
            std::cout << "  Thread " << std::hash<ThreadId>{}(thread_id) << ": " << count << " verifications\n";
        }
    }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // FUNCTION CONTRACTS - TRANSPARENT PRE/POST CONDITIONS
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    struct ContractCondition {
        std::string description;
        std::function<bool()> predicate;

        ContractCondition(std::string desc, std::function<bool()> pred)
            : description(std::move(desc)), predicate(std::move(pred)) {
        }
    };

    struct FunctionContract {
        std::string function_name;
        std::vector<ContractCondition> preconditions;
        std::vector<ContractCondition> postconditions;

        explicit FunctionContract(std::string name) : function_name(std::move(name)) {}

        FunctionContract& requires(const std::string& desc, std::function<bool()> pred) {
            preconditions.emplace_back(desc, std::move(pred));
            return *this;
        }

        FunctionContract& ensures(const std::string& desc, std::function<bool()> pred) {
            postconditions.emplace_back(desc, std::move(pred));
            return *this;
        }

        void check_preconditions() {
            PROOF_CONTEXT("preconditions_" + function_name);

            for (const auto& condition : preconditions) {
                require("precondition: " + condition.description, condition.predicate);
            }
        }

        void check_postconditions() {
            PROOF_CONTEXT("postconditions_" + function_name);

            for (const auto& condition : postconditions) {
                require("postcondition: " + condition.description, condition.predicate);
            }
        }
    };

    // Fixed macros for convenient contract definition
    #define CONTRACT(name) FunctionContract _contract(name)
    #define REQUIRES(desc, expr) _contract.requires(desc, [&]() { return (expr); })
    #define ENSURES(desc, expr) _contract.ensures(desc, [&]() { return (expr); })
    #define CHECK_PRECONDITIONS() _contract.check_preconditions()
    #define CHECK_POSTCONDITIONS() _contract.check_postconditions()

} // namespace axiomatik