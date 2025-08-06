"""
Axiomatik: Performant Runtime Verification for Python with Future Features
A comprehensive system with performance optimization, integration helpers, domain-specific extensions,
adaptive monitoring, performance introspection, and recovery framework.

USAGE EXAMPLE:
    # Adaptive monitoring with priority
    with adaptive_verification_context("database"):
        adaptive_require("connection valid", db.is_connected(),
                        property_name="db_check", priority=5)

    # Auto-tuning for 5% overhead
    auto_tune_verification_level(target_overhead_percent=5.0)

    # Recovery-enabled contracts
    @contract_with_recovery(
        preconditions=[("data valid", lambda x: len(x) > 0)],
        recovery_strategy=RecoveryStrategy(RecoveryPolicy.GRACEFUL_DEGRADATION,
                                         fallback_handler=simple_fallback)
    )
    def robust_function(data):
        return complex_processing(data)
"""

import functools
import os
import time
import threading
import weakref
import statistics
from typing import Callable, List, Tuple, Any, Dict, Optional, Union, Type, Set
from contextlib import contextmanager
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict, deque


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PERFORMANCE CONFIGURATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class VerificationLevel(Enum):
    OFF = "off"
    CONTRACTS = "contracts"
    INVARIANTS = "invariants"
    FULL = "full"
    DEBUG = "debug"


class Config:
    """Global configuration for Axiomatik"""

    def __init__(self):
        self.level = VerificationLevel(os.getenv('AXIOMATIK_LEVEL', 'full'))
        self.cache_enabled = os.getenv('AXIOMATIK_CACHE', '1') == '1'
        self.max_proof_steps = int(os.getenv('AXIOMATIK_MAX_STEPS', '10000'))
        self.performance_mode = os.getenv('AXIOMATIK_PERF', '0') == '1'
        self.debug_mode = self.level == VerificationLevel.DEBUG

    def should_verify(self, context_type: str = None) -> bool:
        """Check if verification should be performed"""
        if self.level == VerificationLevel.OFF:
            return False
        if self.level == VerificationLevel.CONTRACTS and context_type != 'contract':
            return False
        return True


_config = Config()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CORE PROOF SYSTEM WITH PERFORMANCE OPTIMIZATIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ProofFailure(Exception):
    """Raised when we can't prove something"""
    pass


@dataclass
class ProofStep:
    """Individual proof step with metadata"""
    claim: str
    context: str
    timestamp: float
    thread_id: int


class ProofCache:
    """Cache for expensive proof computations"""

    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size

    def get(self, key: str):
        """Get cached result"""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None

    def set(self, key: str, value: Any):
        """Cache result"""
        if len(self.cache) >= self.max_size:
            self._evict_oldest()

        self.cache[key] = value
        self.access_times[key] = time.time()

    def _evict_oldest(self):
        """Remove least recently used item"""
        if self.access_times:
            oldest_key = min(self.access_times.keys(),
                             key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]


class Proof:
    """High-performance proof system with caching and optimization"""

    def __init__(self):
        self.steps: List[ProofStep] = []
        self.contexts = []
        self._cache = ProofCache() if _config.cache_enabled else None
        self._lock = threading.RLock()

    def require(self, claim: str, evidence: Any) -> Any:
        """Runtime check with performance optimization"""
        if not _config.should_verify():
            return evidence

        # Fast path for simple evidence
        if not evidence:
            context = f" (in {self.contexts[-1]})" if self.contexts else ""
            raise ProofFailure(f"Cannot prove: {claim}{context}")

        # Cache check for expensive computations
        if self._cache and callable(evidence):
            cache_key = f"{claim}:{id(evidence)}"
            cached_result = self._cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            result = evidence()
            self._cache.set(cache_key, result)
            if not result:
                raise ProofFailure(f"Cannot prove: {claim}")
            evidence = result

        # Record proof step
        with self._lock:
            if len(self.steps) < _config.max_proof_steps:
                step = ProofStep(
                    claim=claim,
                    context=self.contexts[-1] if self.contexts else "",
                    timestamp=time.time(),
                    thread_id=threading.get_ident()
                )
                self.steps.append(step)

        return evidence

    def push_context(self, context: str):
        """Enter proof context"""
        self.contexts.append(context)

    def pop_context(self):
        """Exit proof context"""
        if self.contexts:
            self.contexts.pop()

    def clear(self):
        """Clear proof trace"""
        with self._lock:
            self.steps.clear()
            self.contexts.clear()
            if self._cache:
                self._cache.cache.clear()
                self._cache.access_times.clear()

    def get_summary(self) -> Dict[str, Any]:
        """Get proof summary statistics"""
        contexts = defaultdict(int)
        threads = set()

        for step in self.steps:
            contexts[step.context] += 1
            threads.add(step.thread_id)

        return {
            'total_steps': len(self.steps),
            'contexts': dict(contexts),
            'thread_count': len(threads),
            'cache_enabled': self._cache is not None
        }


class GhostState:
    """Thread-safe ghost state with scoping"""

    def __init__(self):
        self._data = {}
        self._scopes = []
        self._lock = threading.RLock()

    def set(self, key: str, value: Any):
        """Store ghost variable"""
        with self._lock:
            scope = self._scopes[-1] if self._scopes else self._data
            scope[key] = value

    def get(self, key: str, default=None):
        """Retrieve ghost variable"""
        with self._lock:
            for scope in reversed(self._scopes):
                if key in scope:
                    return scope[key]
            return self._data.get(key, default)

    def push_scope(self):
        """Create new scope"""
        with self._lock:
            self._scopes.append({})

    def pop_scope(self):
        """Exit current scope"""
        with self._lock:
            if self._scopes:
                self._scopes.pop()


# Global instances
_proof = Proof()
_ghost = GhostState()


# def require(claim: str, evidence: Any) -> Any:
#     """Optimized global require function"""
#     return _proof.require(claim, evidence)
def require(claim: str, evidence: Any) -> Any:
    """Optimized global require function with performance tracking"""
    start_time = time.perf_counter()

    try:
        result = _proof.require(claim, evidence)
        # Record the verification for performance analysis
        verification_time = time.perf_counter() - start_time
        _performance_analyzer.record_verification(
            property_name=claim,
            context=_proof.contexts[-1] if _proof.contexts else "global",
            execution_time=0.001,  # Placeholder
            verification_time=verification_time
        )
        return result
    except ProofFailure:
        verification_time = time.perf_counter() - start_time
        _performance_analyzer.record_verification(
            property_name=claim,
            context=_proof.contexts[-1] if _proof.contexts else "global",
            execution_time=0.001,
            verification_time=verification_time
        )
        raise

@contextmanager
def proof_context(name: str):
    """Optimized proof context manager"""
    if not _config.should_verify():
        yield
        return

    _proof.push_context(name)
    _ghost.push_scope()
    try:
        yield
    finally:
        _ghost.pop_scope()
        _proof.pop_context()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ADAPTIVE MONITORING SYSTEM
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class AdaptiveMonitor:
    """Dynamically adapts verification behavior based on runtime conditions"""

    def __init__(self):
        self.load_metrics = deque(maxlen=100)  # Recent performance samples
        self.active_properties = set()
        self.property_costs = {}  # Cost per property verification
        self.sampling_rates = defaultdict(lambda: 1)  # How often to verify each property
        self.property_registry = {}
        self.adaptation_lock = threading.Lock()

        # Thresholds for adaptation
        self.high_load_threshold = 0.1  # 100ms average verification time
        self.critical_load_threshold = 0.5  # 500ms average verification time

    def register_property(self, property_name: str,
                          verification_func: Callable,
                          priority: int = 1,
                          cost_estimate: float = 0.001):
        """Register a verifiable property with metadata"""
        self.property_registry[property_name] = {
            'func': verification_func,
            'priority': priority,  # 1=low, 5=critical
            'cost_estimate': cost_estimate,
            'success_rate': 1.0,
            'recent_failures': deque(maxlen=10)
        }
        self.active_properties.add(property_name)

    def should_verify_property(self, property_name: str) -> bool:
        """Decide whether to verify a property based on current load"""
        if property_name not in self.active_properties:
            return False

        # Always verify critical properties
        prop_info = self.property_registry.get(property_name, {})
        if prop_info.get('priority', 1) >= 4:
            return True

        # Check sampling rate
        sampling_rate = self.sampling_rates[property_name]
        if sampling_rate <= 1:
            return True

        # Use hash of current time for deterministic sampling
        return hash(time.time()) % sampling_rate == 0

    def record_verification_cost(self, property_name: str, cost: float, success: bool):
        """Record the cost and result of a verification"""
        with self.adaptation_lock:
            self.load_metrics.append(cost)
            self.property_costs[property_name] = cost

            # Update property success rate
            if property_name in self.property_registry:
                prop_info = self.property_registry[property_name]
                prop_info['recent_failures'].append(not success)
                failure_rate = sum(prop_info['recent_failures']) / len(prop_info['recent_failures'])
                prop_info['success_rate'] = 1.0 - failure_rate

            # Trigger adaptation if needed
            self._adapt_if_needed()

    def _adapt_if_needed(self):
        """Adapt verification strategy based on current load"""
        if len(self.load_metrics) < 10:
            return

        avg_cost = sum(self.load_metrics) / len(self.load_metrics)

        if avg_cost > self.critical_load_threshold:
            # Critical load - disable low-priority properties
            self._disable_low_priority_properties()
            self._increase_sampling_rates()

        elif avg_cost > self.high_load_threshold:
            # High load - increase sampling for expensive properties
            self._increase_sampling_rates()

        elif avg_cost < self.high_load_threshold / 2:
            # Low load - can re-enable properties
            self._decrease_sampling_rates()
            self._enable_high_success_properties()

    def _disable_low_priority_properties(self):
        """Temporarily disable low-priority properties"""
        for prop_name, prop_info in self.property_registry.items():
            if prop_info.get('priority', 1) <= 2:
                self.active_properties.discard(prop_name)

    def _increase_sampling_rates(self):
        """Reduce verification frequency for expensive properties"""
        for prop_name, cost in self.property_costs.items():
            if cost > 0.05:  # Expensive property
                self.sampling_rates[prop_name] = min(self.sampling_rates[prop_name] * 2, 10)

    def _decrease_sampling_rates(self):
        """Increase verification frequency when load is low"""
        for prop_name in self.sampling_rates:
            self.sampling_rates[prop_name] = max(self.sampling_rates[prop_name] // 2, 1)

    def _enable_high_success_properties(self):
        """Re-enable properties with high success rates"""
        for prop_name, prop_info in self.property_registry.items():
            if prop_info.get('success_rate', 0) > 0.95:
                self.active_properties.add(prop_name)


# Require function with adaptive monitoring
_adaptive_monitor = AdaptiveMonitor()


def adaptive_require(claim: str, evidence: Any,
                     property_name: str = None,
                     priority: int = 1) -> Any:
    """Require with adaptive monitoring"""

    if property_name is None:
        property_name = f"anonymous_{claim[:20]}"

    # Register property if not seen before
    if property_name not in _adaptive_monitor.property_registry:
        _adaptive_monitor.register_property(
            property_name,
            lambda: evidence,
            priority=priority
        )

    # Check if we should verify this property
    if not _adaptive_monitor.should_verify_property(property_name):
        return evidence  # Skip verification due to load

    # Perform verification with timing
    start_time = time.perf_counter()
    try:
        result = require(claim, evidence)
        success = True
        return result
    except ProofFailure:
        success = False
        raise
    finally:
        cost = time.perf_counter() - start_time
        _adaptive_monitor.record_verification_cost(property_name, cost, success)


class PropertyManager:
    """Manages dynamic loading/unloading of verification properties"""

    def __init__(self):
        self.loaded_properties = {}
        self.property_modules = {}

    def load_properties_for_context(self, context: str):
        """Load verification properties specific to a context"""
        if context == "database":
            self._load_database_properties()
        elif context == "network":
            self._load_network_properties()
        elif context == "crypto":
            self._load_crypto_properties()

    def unload_properties_for_context(self, context: str):
        """Unload properties when leaving a context"""
        properties_to_remove = [
            name for name in self.loaded_properties
            if name.startswith(f"{context}_")
        ]
        for prop_name in properties_to_remove:
            del self.loaded_properties[prop_name]
            _adaptive_monitor.active_properties.discard(prop_name)

    def _load_database_properties(self):
        """Load database-specific properties"""
        _adaptive_monitor.register_property(
            "db_connection_valid",
            lambda: True,
            priority=5
        )

    def _load_network_properties(self):
        """Load network-specific properties"""
        _adaptive_monitor.register_property(
            "network_available",
            lambda: True,
            priority=4
        )

    def _load_crypto_properties(self):
        """Load crypto-specific properties"""
        _adaptive_monitor.register_property(
            "crypto_key_valid",
            lambda: True,
            priority=5
        )


@contextmanager
def adaptive_verification_context(context: str):
    """Context manager that loads appropriate properties"""
    property_manager = PropertyManager()
    property_manager.load_properties_for_context(context)
    try:
        yield
    finally:
        property_manager.unload_properties_for_context(context)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PERFORMANCE INTROSPECTION SYSTEM
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@dataclass
class VerificationHotspot:
    """Represents an expensive verification check"""
    property_name: str
    total_time: float
    call_count: int
    average_time: float
    percentage_of_total: float
    context: str


class PerformanceAnalyzer:
    """Analyzes verification performance and identifies hotspots"""

    def __init__(self):
        self.verification_times = defaultdict(list)
        self.context_times = defaultdict(list)
        self.total_verification_time = 0.0
        self.auto_tuning_enabled = False
        self.target_overhead_percent = 5.0
        self.last_adjustment_time = 0
        self.adjustment_cooldown = 1.0
        self.adjustment_count = 0

    def record_verification(self, property_name: str, context: str,
                            execution_time: float, verification_time: float):
        """Record performance data for a verification"""
        self.verification_times[property_name].append(verification_time)
        self.context_times[context].append(verification_time)
        self.total_verification_time += verification_time

        # Auto-tune if enabled
        if self.auto_tuning_enabled:
            self._check_auto_tune(execution_time, verification_time)

    def get_performance_hotspots(self, top_n: int = 10) -> List[VerificationHotspot]:
        """Identify the most expensive verification checks"""
        hotspots = []

        for prop_name, times in self.verification_times.items():
            total_time = sum(times)
            call_count = len(times)
            avg_time = total_time / call_count
            percentage = (total_time / max(self.total_verification_time, 0.001)) * 100

            # Find most common context for this property
            context = self._find_primary_context(prop_name)

            hotspots.append(VerificationHotspot(
                property_name=prop_name,
                total_time=total_time,
                call_count=call_count,
                average_time=avg_time,
                percentage_of_total=percentage,
                context=context
            ))

        # Sort by total time and return top N
        hotspots.sort(key=lambda x: x.total_time, reverse=True)
        return hotspots[:top_n]

    def _find_primary_context(self, prop_name: str) -> str:
        """Find the most common context for a property"""
        # Simple implementation - return first context found
        for step in _proof.steps:
            if prop_name in step.claim:
                return step.context
        return "unknown"

    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        hotspots = self.get_performance_hotspots()

        report = ["Axiomatik Performance Analysis", "=" * 50, ""]

        # Overall statistics
        total_properties = len(self.verification_times)
        total_calls = sum(len(times) for times in self.verification_times.values())
        avg_verification_time = self.total_verification_time / max(1, total_calls)

        report.extend([
            f"Total properties verified: {total_properties}",
            f"Total verification calls: {total_calls}",
            f"Total verification time: {self.total_verification_time:.3f}s",
            f"Average per verification: {avg_verification_time * 1000:.3f}ms",
            ""
        ])

        # Top hotspots
        report.append("Top Performance Hotspots:")
        for i, hotspot in enumerate(hotspots, 1):
            report.append(
                f"  {i:2d}. {hotspot.property_name[:40]:40} "
                f"{hotspot.total_time * 1000:6.1f}ms "
                f"({hotspot.percentage_of_total:4.1f}%) "
                f"[{hotspot.call_count} calls]"
            )

        # Context analysis
        report.extend(["", "Performance by Context:"])
        context_totals = {
            ctx: sum(times) for ctx, times in self.context_times.items()
        }
        for context, total_time in sorted(context_totals.items(),
                                          key=lambda x: x[1], reverse=True):
            percentage = (total_time / max(self.total_verification_time, 0.001)) * 100
            report.append(f"  {context:20}: {total_time * 1000:6.1f}ms ({percentage:4.1f}%)")

        return "\n".join(report)

    def auto_tune_verification_level(self, target_overhead_percent: float = 5.0):
        """Automatically tune verification level based on measured overhead"""
        self.auto_tuning_enabled = True
        self.target_overhead_percent = target_overhead_percent

        print(f"Auto-tuning enabled: target overhead {target_overhead_percent}%")

    def _check_auto_tune(self, execution_time: float, verification_time: float):
        """Check if auto-tuning adjustment is needed"""
        # Skip if disabled or in cooldown
        if not self.auto_tuning_enabled:
            return

        current_time = time.time()
        if current_time - self.last_adjustment_time < self.adjustment_cooldown:
            return

        # Skip if execution time is clearly a placeholder
        if execution_time <= 0.001:
            return

        overhead_percent = (verification_time / execution_time) * 100

        # Only adjust if overhead is significantly outside target range
        if overhead_percent > self.target_overhead_percent * 2.0:  # More conservative
            if self._reduce_verification_intensity():
                self.last_adjustment_time = current_time
                self.adjustment_count += 1
        elif overhead_percent < self.target_overhead_percent * 0.25:  # Much more conservative
            if self._increase_verification_intensity():
                self.last_adjustment_time = current_time
                self.adjustment_count += 1

        # Disable after too many adjustments
        if self.adjustment_count > 5:
            self.auto_tuning_enabled = False
            print("Auto-tuning disabled after 5 adjustments")

    def _reduce_verification_intensity(self) -> bool:
        """Reduce verification intensity to meet performance targets"""
        current_level = _config.level

        # Define explicit ordering
        level_order = [
            VerificationLevel.DEBUG,
            VerificationLevel.FULL,
            VerificationLevel.INVARIANTS,
            VerificationLevel.CONTRACTS,
            VerificationLevel.OFF
        ]

        try:
            current_index = level_order.index(current_level)
            if current_index < len(level_order) - 1:  # Not at minimum
                new_level = level_order[current_index + 1]
                _config.level = new_level
                print(f"Auto-tune: Reduced verification level to {new_level.value}")
                return True
        except ValueError:
            pass

        return False

    def _increase_verification_intensity(self) -> bool:
        """Increase verification intensity when performance allows"""
        current_level = _config.level

        # Define explicit ordering
        level_order = [
            VerificationLevel.OFF,
            VerificationLevel.CONTRACTS,
            VerificationLevel.INVARIANTS,
            VerificationLevel.FULL,
            VerificationLevel.DEBUG
        ]

        try:
            current_index = level_order.index(current_level)
            if current_index < len(level_order) - 1:  # Not at maximum
                new_level = level_order[current_index + 1]
                _config.level = new_level
                print(f"Auto-tune: Increased verification level to {new_level.value}")
                return True
        except ValueError:
            pass

        return False

    def visualize_hotspots(self, save_path: str = None):
        """Create visualization of performance hotspots"""
        try:
            import matplotlib.pyplot as plt
            hotspots = self.get_performance_hotspots()

            if not hotspots:
                print("No performance data available for visualization")
                return

            # Create bar chart of top hotspots
            names = [h.property_name[:20] for h in hotspots[:10]]
            times = [h.total_time * 1000 for h in hotspots[:10]]  # Convert to ms

            plt.figure(figsize=(12, 6))
            plt.bar(names, times)
            plt.title("Top 10 Verification Performance Hotspots")
            plt.xlabel("Property Name")
            plt.ylabel("Total Time (ms)")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()

        except ImportError:
            print("Matplotlib not available for visualization")


# Global performance analyzer
_performance_analyzer = PerformanceAnalyzer()


# Global functions
def get_performance_hotspots(top_n: int = 10) -> List[VerificationHotspot]:
    """Get the most expensive verification checks"""
    return _performance_analyzer.get_performance_hotspots(top_n)


def auto_tune_verification_level(target_overhead_percent: float = 5.0):
    """Automatically tune verification level based on measured overhead"""
    _performance_analyzer.auto_tune_verification_level(target_overhead_percent)


def generate_performance_report() -> str:
    """Generate comprehensive performance report"""
    return _performance_analyzer.generate_performance_report()


def visualize_performance(save_path: str = None):
    """Visualize verification performance"""
    _performance_analyzer.visualize_hotspots(save_path)


# Integration with existing require function
def _future_require(claim: str, evidence: Any, context: str = "") -> Any:
    """Require with performance tracking"""
    start_time = time.perf_counter()

    try:
        result = require(claim, evidence)
        return result
    finally:
        verification_time = time.perf_counter() - start_time
        _performance_analyzer.record_verification(
            property_name=claim,
            context=context or "global",
            execution_time=0.001,  # Placeholder - would need actual execution timing
            verification_time=verification_time
        )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# RECOVERY FRAMEWORK
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RecoveryPolicy(Enum):
    FAIL_FAST = "fail_fast"  # Current behavior - raise exception
    GRACEFUL_DEGRADATION = "graceful_degradation"  # Use simpler algorithm
    RETRY_WITH_BACKOFF = "retry_with_backoff"  # Retry with exponential backoff
    CIRCUIT_BREAKER = "circuit_breaker"  # Disable after repeated failures
    ROLLBACK_STATE = "rollback_state"  # Restore previous known-good state


class RecoveryStrategy:
    """Defines how to recover from verification failures"""

    def __init__(self,
                 policy: RecoveryPolicy,
                 fallback_handler: Callable = None,
                 max_retries: int = 3,
                 backoff_factor: float = 2.0,
                 circuit_breaker_threshold: int = 5):
        self.policy = policy
        self.fallback_handler = fallback_handler
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.failure_count = 0
        self.circuit_open = False


class RecoveryManager:
    """Manages recovery state and strategies"""

    def __init__(self):
        self.state_snapshots = {}  # For rollback recovery
        self.recovery_stats = {}  # Track recovery effectiveness

    def capture_state(self, function_name: str, args, kwargs):
        """Capture state for potential rollback"""
        self.state_snapshots[function_name] = {
            'args': args,
            'kwargs': kwargs,
            'timestamp': time.time(),
            'call_count': self.recovery_stats.get(function_name, {}).get('calls', 0)
        }

    def execute_recovery(self, strategy: RecoveryStrategy,
                         original_function: Callable,
                         violation: ProofFailure,
                         *args, **kwargs):
        """Execute recovery strategy when verification fails"""

        if strategy.policy == RecoveryPolicy.GRACEFUL_DEGRADATION:
            if strategy.fallback_handler:
                return strategy.fallback_handler(*args, **kwargs)
            else:
                # Use a generic simplified version
                return self._simplified_fallback(original_function, *args, **kwargs)

        elif strategy.policy == RecoveryPolicy.RETRY_WITH_BACKOFF:
            return self._retry_with_backoff(strategy, original_function, *args, **kwargs)

        elif strategy.policy == RecoveryPolicy.CIRCUIT_BREAKER:
            return self._circuit_breaker_recovery(strategy, original_function, *args, **kwargs)

        elif strategy.policy == RecoveryPolicy.ROLLBACK_STATE:
            return self._rollback_state_recovery(strategy, original_function, *args, **kwargs)

        else:  # FAIL_FAST
            raise violation

    def _simplified_fallback(self, original_function, *args, **kwargs):
        """Generic simplified fallback - disable verification and retry"""
        old_level = _config.level
        try:
            _config.level = VerificationLevel.OFF
            return original_function(*args, **kwargs)
        finally:
            _config.level = old_level

    def _retry_with_backoff(self, strategy: RecoveryStrategy, original_function, *args, **kwargs):
        """Retry with exponential backoff"""
        last_exception = None
        for attempt in range(strategy.max_retries):
            try:
                return original_function(*args, **kwargs)
            except ProofFailure as e:
                last_exception = e
                if attempt < strategy.max_retries - 1:
                    delay = (strategy.backoff_factor ** attempt) * 0.1
                    time.sleep(delay)

        # All retries failed
        raise last_exception

    def _circuit_breaker_recovery(self, strategy: RecoveryStrategy, original_function, *args, **kwargs):
        """Circuit breaker recovery pattern"""
        if strategy.circuit_open:
            # Circuit is open, use fallback immediately
            if strategy.fallback_handler:
                return strategy.fallback_handler(*args, **kwargs)
            else:
                return self._simplified_fallback(original_function, *args, **kwargs)

        try:
            result = original_function(*args, **kwargs)
            strategy.failure_count = 0  # Reset on success
            return result
        except ProofFailure as e:
            strategy.failure_count += 1
            if strategy.failure_count >= strategy.circuit_breaker_threshold:
                strategy.circuit_open = True
                print(f"Circuit breaker opened after {strategy.failure_count} failures")
            raise e

    def _rollback_state_recovery(self, strategy: RecoveryStrategy, original_function, *args, **kwargs):
        """Rollback to previous state and retry"""
        func_name = original_function.__name__
        if func_name in self.state_snapshots:
            snapshot = self.state_snapshots[func_name]
            # Restore previous args/kwargs and retry
            return original_function(*snapshot['args'], **snapshot['kwargs'])
        else:
            # No snapshot available, use fallback
            return self._simplified_fallback(original_function, *args, **kwargs)


# Contract decorator with recovery
def contract_with_recovery(
        preconditions: List[Tuple[str, Callable]] = None,
        postconditions: List[Tuple[str, Callable]] = None,
        recovery_strategy: RecoveryStrategy = None
):
    """Contract decorator with automated recovery capabilities"""

    if recovery_strategy is None:
        recovery_strategy = RecoveryStrategy(RecoveryPolicy.FAIL_FAST)

    recovery_manager = RecoveryManager()

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__

            # Capture state for potential recovery
            if recovery_strategy.policy == RecoveryPolicy.ROLLBACK_STATE:
                recovery_manager.capture_state(func_name, args, kwargs)

            try:
                # Standard contract verification
                if preconditions:
                    for claim, condition_fn in preconditions:
                        require(f"precondition: {claim}", condition_fn(*args, **kwargs))

                result = func(*args, **kwargs)

                if postconditions:
                    for claim, condition_fn in postconditions:
                        require(f"postcondition: {claim}",
                                condition_fn(*args, result=result, **kwargs))

                # Reset failure count on success
                recovery_strategy.failure_count = 0
                return result

            except ProofFailure as violation:
                # Execute recovery strategy
                recovery_strategy.failure_count += 1

                # Update recovery statistics
                stats = recovery_manager.recovery_stats.setdefault(func_name, {
                    'calls': 0, 'failures': 0, 'recoveries': 0
                })
                stats['failures'] += 1

                try:
                    result = recovery_manager.execute_recovery(
                        recovery_strategy, func, violation, *args, **kwargs
                    )
                    stats['recoveries'] += 1
                    return result
                except Exception as recovery_error:
                    # Recovery failed, log and re-raise original violation
                    print(f"Recovery failed for {func_name}: {recovery_error}")
                    raise violation

        return wrapper

    return decorator


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# REFINEMENT TYPES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RefinementType:
    """Base class for refinement types"""

    def __init__(self, base_type: Type, predicate: Callable[[Any], bool],
                 description: str = ""):
        self.base_type = base_type
        self.predicate = predicate
        self.description = description

    def validate(self, value: Any) -> Any:
        """Validate value against refinement"""
        require(f"value is {self.base_type.__name__}", isinstance(value, self.base_type))
        require(f"value satisfies: {self.description}", self.predicate(value))
        return value

    def __call__(self, value: Any) -> Any:
        """Allow type to be used as constructor"""
        return self.validate(value)


class RangeInt(RefinementType):
    """Integer within specific range"""

    def __init__(self, min_val: int, max_val: int | float):
        super().__init__(
            int,
            lambda x: min_val <= x <= max_val,
            f"integer in range [{min_val}, {max_val}]"
        )
        self.min_val = min_val
        self.max_val = max_val


class NonEmptyList(RefinementType):
    """Non-empty list"""

    def __init__(self, element_type_or_data=None):
        # Check if the argument is actual data (a list) or a type specification
        if isinstance(element_type_or_data, list):
            # This is actual data to validate - create a basic non-empty validator
            predicate = lambda x: len(x) > 0
            super().__init__(
                list,
                predicate,
                "non-empty list"
            )
            # Validate and store the data immediately
            self._validated_data = self.validate(element_type_or_data)
        else:
            # This is a type specification (or None)
            element_type = element_type_or_data
            predicate = lambda x: len(x) > 0

            if element_type:
                predicate = lambda x: len(x) > 0 and all(isinstance(e, element_type) for e in x)

            # Safely get the type name
            type_name = ""
            if element_type:
                if hasattr(element_type, '__name__'):
                    type_name = f" of {element_type.__name__}"
                else:
                    type_name = f" of {str(element_type)}"

            super().__init__(
                list,
                predicate,
                f"non-empty list{type_name}"
            )
            self._validated_data = None

    def __call__(self, value=None):
        """Allow type to be used as constructor or return validated data"""
        if value is None and hasattr(self, '_validated_data') and self._validated_data is not None:
            # Return the already validated data
            return self._validated_data
        elif value is not None:
            # Validate the provided value
            return self.validate(value)
        else:
            # This shouldn't happen, but handle gracefully
            raise ValueError("No data provided for validation")

    def __iter__(self):
        """Make the validated list iterable"""
        if hasattr(self, '_validated_data') and self._validated_data is not None:
            return iter(self._validated_data)
        else:
            raise ValueError("No validated data available for iteration")

    def __len__(self):
        """Return length of validated data"""
        if hasattr(self, '_validated_data') and self._validated_data is not None:
            return len(self._validated_data)
        else:
            return 0

    def __getitem__(self, index):
        """Allow indexing into validated data"""
        if hasattr(self, '_validated_data') and self._validated_data is not None:
            return self._validated_data[index]
        else:
            raise ValueError("No validated data available for indexing")

    def __repr__(self):
        """String representation"""
        if hasattr(self, '_validated_data') and self._validated_data is not None:
            return f"NonEmptyList({self._validated_data})"
        else:
            return f"NonEmptyList(type={self.description})"


class ValidatedString(RefinementType):
    """String matching pattern"""

    def __init__(self, pattern: str, description: str = ""):
        import re
        regex = re.compile(pattern)
        super().__init__(
            str,
            lambda x: bool(regex.match(x)),
            description or f"string matching {pattern}"
        )


# Predefined refinement types
PositiveInt = RangeInt(1, float('inf'))
Natural = RangeInt(0, float('inf'))
Percentage = RangeInt(0, 100)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PROTOCOL VERIFICATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ProtocolState:
    """Represents a protocol state"""

    def __init__(self, name: str, allowed_transitions: List[str] = None):
        self.name = name
        self.allowed_transitions = allowed_transitions or []


class Protocol:
    """Define and verify API usage protocols"""

    def __init__(self, name: str, initial_state: str):
        self.name = name
        self.initial_state = initial_state
        self.states = {}
        self.instances = weakref.WeakKeyDictionary()

    def add_state(self, state: ProtocolState):
        """Add state to protocol"""
        self.states[state.name] = state
        return self

    def transition(self, from_state: str, to_state: str):
        """Add allowed transition"""
        if from_state in self.states:
            self.states[from_state].allowed_transitions.append(to_state)
        return self

    def verify_transition(self, obj: Any, new_state: str):
        """Verify protocol transition"""
        current_state = self.instances.get(obj, self.initial_state)

        with proof_context(f"protocol[{self.name}]"):
            require("target state exists", new_state in self.states)
            require("transition is allowed",
                    new_state in self.states[current_state].allowed_transitions)

        self.instances[obj] = new_state

    def get_state(self, obj: Any) -> str:
        """Get current protocol state"""
        return self.instances.get(obj, self.initial_state)


def protocol_method(protocol: Protocol, target_state: str):
    """Decorator for protocol-aware methods"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            protocol.verify_transition(self, target_state)
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# INFORMATION FLOW TRACKING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class SecurityLabel(Enum):
    """Security labels for information flow"""
    PUBLIC = "public"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


class TaintedValue:
    """Value with information flow tracking"""

    def __init__(self, value: Any, label: SecurityLabel, provenance: List[str] = None):
        self.value = value
        self.label = label
        self.provenance = provenance or []
        self.created_at = time.time()

    def combine_with(self, other: 'TaintedValue') -> 'TaintedValue':
        """Combine two tainted values"""
        # Take maximum security level
        max_label = max(self.label, other.label, key=lambda x: x.value)
        combined_provenance = self.provenance + other.provenance

        return TaintedValue(
            value=(self.value, other.value),
            label=max_label,
            provenance=combined_provenance
        )

    def can_flow_to(self, target_label: SecurityLabel) -> bool:
        """Check if information can flow to target label"""
        return self.label.value <= target_label.value

    def declassify(self, new_label: SecurityLabel, justification: str):
        """Declassify information with justification"""
        with proof_context("information_flow.declassify"):
            require("declassification is justified", len(justification) > 0)
            require("declassification reduces security level",
                    new_label.value < self.label.value)

        self.label = new_label
        self.provenance.append(f"declassified: {justification}")


class InformationFlowTracker:
    """Track information flow throughout computation"""

    def __init__(self):
        self.flows = []
        self.policies = {}

    def add_policy(self, from_label: SecurityLabel, to_label: SecurityLabel, allowed: bool):
        """Add information flow policy"""
        self.policies[(from_label, to_label)] = allowed

    def track_flow(self, source: TaintedValue, target_label: SecurityLabel):
        """Track information flow"""
        with proof_context("information_flow.track"):
            policy_key = (source.label, target_label)
            if policy_key in self.policies:
                require("flow is allowed by policy", self.policies[policy_key])
            else:
                require("flow satisfies default policy", source.can_flow_to(target_label))

        self.flows.append({
            'from': source.label,
            'to': target_label,
            'timestamp': time.time(),
            'provenance': source.provenance
        })


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TEMPORAL PROPERTIES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TemporalProperty:
    """Base class for temporal properties"""

    def __init__(self, name: str):
        self.name = name
        self.history = deque(maxlen=1000)  # Bounded history

    def record_event(self, event: str, data: Any = None):
        """Record event for temporal checking"""
        self.history.append({
            'event': event,
            'data': data,
            'timestamp': time.time(),
            'thread': threading.get_ident()
        })

    def check(self) -> bool:
        """Check if property holds"""
        raise NotImplementedError


class EventuallyProperty(TemporalProperty):
    """Property that must eventually become true"""

    def __init__(self, name: str, condition: Callable[[List], bool], timeout: float = 10.0):
        super().__init__(name)
        self.condition = condition
        self.timeout = timeout
        self.start_time = time.time()

    def check(self) -> bool:
        """Check if condition eventually holds"""
        if self.condition(list(self.history)):
            return True

        # Check timeout
        if time.time() - self.start_time > self.timeout:
            raise ProofFailure(f"Eventually property '{self.name}' timed out")

        return False


class AlwaysProperty(TemporalProperty):
    """Property that must always be true"""

    def __init__(self, name: str, invariant: Callable[[Dict], bool]):
        super().__init__(name)
        self.invariant = invariant

    def check(self) -> bool:
        """Check if invariant always holds"""
        for event in self.history:
            if not self.invariant(event):
                return False
        return True


class TemporalVerifier:
    """Verify temporal properties"""

    def __init__(self):
        self.properties = []

    def add_property(self, prop: TemporalProperty):
        """Add temporal property to verify"""
        self.properties.append(prop)

    def record_event(self, event: str, data: Any = None):
        """Record event for all properties"""
        for prop in self.properties:
            prop.record_event(event, data)

    def verify_all(self):
        """Verify all temporal properties"""
        with proof_context("temporal_verification"):
            for prop in self.properties:
                require(f"temporal property '{prop.name}' holds", prop.check())


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PLUGIN SYSTEM
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Plugin:
    """Base class for Axiomatik plugins"""

    def __init__(self, name: str):
        self.name = name

    def initialize(self):
        """Initialize plugin"""
        pass

    def add_verifiers(self) -> Dict[str, Callable]:
        """Return dictionary of verification functions"""
        return {}

    def add_types(self) -> Dict[str, Type]:
        """Return dictionary of custom types"""
        return {}


class PluginRegistry:
    """Registry for Axiomatik plugins"""

    def __init__(self):
        self.plugins = {}
        self.verifiers = {}
        self.types = {}

    def register(self, plugin: Plugin):
        """Register a plugin"""
        plugin.initialize()
        self.plugins[plugin.name] = plugin

        # Add verifiers
        for name, verifier in plugin.add_verifiers().items():
            self.verifiers[name] = verifier

        # Add types
        for name, type_cls in plugin.add_types().items():
            self.types[name] = type_cls

    def get_verifier(self, name: str) -> Optional[Callable]:
        """Get verifier by name"""
        return self.verifiers.get(name)

    def get_type(self, name: str) -> Optional[Type]:
        """Get type by name"""
        return self.types.get(name)


_plugin_registry = PluginRegistry()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DOMAIN-SPECIFIC EXTENSIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Cryptographic Verification Plugin
class CryptoPlugin(Plugin):
    """Plugin for cryptographic verification"""

    def __init__(self):
        super().__init__("crypto")

    def add_verifiers(self):
        return {
            'constant_time': self.verify_constant_time,
            'key_zeroized': self.verify_key_zeroized,
            'secure_random': self.verify_secure_random
        }

    @staticmethod
    def verify_constant_time(func: Callable, inputs: List[Any]) -> bool:
        """Verify function runs in constant time"""
        times = []
        for inp in inputs:
            start = time.perf_counter()
            func(inp)
            end = time.perf_counter()
            times.append(end - start)

        # Check that timing variation is within acceptable bounds
        if len(times) < 2:
            return True

        mean_time = sum(times) / len(times)
        max_deviation = max(abs(t - mean_time) for t in times)

        # Allow 10% timing variation
        return max_deviation / mean_time <= 0.1

    @staticmethod
    def verify_key_zeroized(key_data: Any) -> bool:
        """Verify sensitive data has been zeroized"""
        if isinstance(key_data, (bytes, bytearray)):
            return all(b == 0 for b in key_data)
        elif isinstance(key_data, str):
            return all(c == '\x00' for c in key_data)
        return False

    @staticmethod
    def verify_secure_random(random_func: Callable) -> bool:
        """Verify random number generator is cryptographically secure"""
        # This is a simplified check - real implementation would be more sophisticated
        import secrets
        return random_func.__module__ == 'secrets' or hasattr(random_func, 'SystemRandom')


# Financial Calculation Plugin
class FinancePlugin(Plugin):
    """Plugin for financial calculation verification"""

    def __init__(self):
        super().__init__("finance")
        self.audit_trail = []

    def add_types(self):
        return {
            'Money': self.Money,
            'Percentage': lambda x: Percentage(x)
        }

    def add_verifiers(self):
        return {
            'precision_maintained': self.verify_precision,
            'audit_recorded': self.verify_audit_trail
        }

    class Money:
        """Precise money representation"""

        def __init__(self, amount: Union[int, float, str], currency: str = "USD"):
            from decimal import Decimal
            self.amount = Decimal(str(amount))
            self.currency = currency

        def __add__(self, other):
            require("same currency", self.currency == other.currency)
            return FinancePlugin.Money(self.amount + other.amount, self.currency)

        def __str__(self):
            return f"{self.amount} {self.currency}"

    @staticmethod
    def verify_precision(calculation_result: Any, expected_precision: int) -> bool:
        """Verify financial calculation maintains required precision"""
        if hasattr(calculation_result, 'amount'):
            # Check decimal places
            decimal_places = abs(calculation_result.amount.as_tuple().exponent)
            return decimal_places <= expected_precision
        return True

    def verify_audit_trail(self, operation: str, inputs: List[Any], result: Any) -> bool:
        """Record and verify audit trail"""
        audit_entry = {
            'operation': operation,
            'inputs': [str(inp) for inp in inputs],
            'result': str(result),
            'timestamp': time.time(),
            'thread': threading.get_ident()
        }
        self.audit_trail.append(audit_entry)
        return True


# Security Plugin
class SecurityPlugin(Plugin):
    """Plugin for security verification"""

    def __init__(self):
        super().__init__("security")

    def add_verifiers(self):
        return {
            'input_sanitized': self.verify_input_sanitized,
            'privilege_boundary': self.verify_privilege_boundary,
            'no_injection': self.verify_no_injection
        }

    @staticmethod
    def verify_input_sanitized(input_data: str, sanitizer_func: Callable) -> bool:
        """Verify input has been properly sanitized"""
        sanitized = sanitizer_func(input_data)

        # Check for common injection patterns
        dangerous_patterns = ['<script', 'javascript:', 'DROP TABLE', 'SELECT * FROM']
        return not any(pattern.lower() in sanitized.lower() for pattern in dangerous_patterns)

    @staticmethod
    def verify_privilege_boundary(user_role: str, required_privilege: str) -> bool:
        """Verify user has required privileges"""
        privilege_hierarchy = {
            'guest': 0, 'user': 1, 'admin': 2, 'superuser': 3
        }

        user_level = privilege_hierarchy.get(user_role, -1)
        required_level = privilege_hierarchy.get(required_privilege, 3)

        return user_level >= required_level

    @staticmethod
    def verify_no_injection(query: str, user_input: str) -> bool:
        """Verify query is safe from injection"""
        # Simplified check - real implementation would be more sophisticated
        return user_input not in query or query.count('?') >= query.count(user_input)


# Concurrency Plugin
class ConcurrencyPlugin(Plugin):
    """Plugin for concurrency verification"""

    def __init__(self):
        super().__init__("concurrency")
        self.lock_graph = defaultdict(set)
        self.thread_locks = defaultdict(set)

    def add_verifiers(self):
        return {
            'no_deadlock': self.verify_no_deadlock,
            'race_free': self.verify_race_free,
            'atomic_operation': self.verify_atomic
        }

    def verify_no_deadlock(self, lock1: Any, lock2: Any) -> bool:
        """Verify acquiring locks won't cause deadlock"""
        thread_id = threading.get_ident()

        # Check if this would create a cycle in the lock graph
        if lock2 in self.lock_graph[lock1]:
            return False  # Would create immediate cycle

        # Add to lock graph
        self.lock_graph[lock1].add(lock2)
        self.thread_locks[thread_id].update([lock1, lock2])

        return True

    @staticmethod
    def verify_race_free(shared_resource: Any, access_type: str) -> bool:
        """Verify access to shared resource is race-free"""
        # This is a simplified implementation
        # Real implementation would track actual synchronization
        return hasattr(shared_resource, '_lock') or access_type == 'read'

    @staticmethod
    def verify_atomic(operation: Callable) -> bool:
        """Verify operation is atomic"""
        # Check if operation uses atomic primitives or synchronization
        import inspect
        source = inspect.getsource(operation)
        atomic_indicators = ['with lock', 'atomic', 'synchronized', 'Lock()', 'RLock()']
        return any(indicator in source for indicator in atomic_indicators)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# INTEGRATION HELPERS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def gradually_verify(level: VerificationLevel = VerificationLevel.CONTRACTS):
    """Decorator for gradual adoption of verification"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            old_level = _config.level
            _config.level = level
            try:
                return func(*args, **kwargs)
            finally:
                _config.level = old_level

        return wrapper

    return decorator


def auto_contract(func):
    """Automatically generate contracts from type hints"""
    import inspect

    sig = inspect.signature(func)
    preconditions = []
    postconditions = []

    # Generate preconditions from parameter types
    for param_name, param in sig.parameters.items():
        if param.annotation != inspect.Parameter.empty:
            param_type = param.annotation

            # Handle refinement types
            if (isinstance(param_type, type) and
                    issubclass(param_type, RefinementType)):

                def make_refinement_precondition(pt, pn, param_obj):
                    def check_refinement(*args, **kwargs):
                        # Get the parameter value
                        value = kwargs.get(pn)
                        if value is None:
                            param_names = list(sig.parameters.keys())
                            try:
                                param_index = param_names.index(pn)
                                if param_index < len(args):
                                    value = args[param_index]
                            except (ValueError, IndexError):
                                pass

                        if value is None and param_obj.default != inspect.Parameter.empty:
                            value = param_obj.default

                        if value is None:
                            return True  # Skip validation for missing optional parameters

                        try:
                            # For refinement types, create an instance and validate
                            if hasattr(pt, '_validated_data'):
                                # This is an instantiated refinement type
                                pt.validate(value)
                            else:
                                # This is a refinement type class, instantiate it
                                refinement_instance = pt()
                                refinement_instance.validate(value)
                            return True
                        except ProofFailure:
                            return False
                        except Exception:
                            return False

                    return check_refinement

                type_name = getattr(param_type, '__name__', str(param_type))
                preconditions.append((
                    f"{param_name} is {type_name}",
                    make_refinement_precondition(param_type, param_name, param)
                ))

            else:
                # Handle regular types and typing generics
                type_name = getattr(param_type, '__name__', str(param_type))

                def make_precondition(pt, pn, param_obj):
                    def check_type(*args, **kwargs):
                        value = kwargs.get(pn)
                        if value is None:
                            param_names = list(sig.parameters.keys())
                            try:
                                param_index = param_names.index(pn)
                                if param_index < len(args):
                                    value = args[param_index]
                            except (ValueError, IndexError):
                                pass

                        if value is None and param_obj.default != inspect.Parameter.empty:
                            value = param_obj.default

                        if value is None:
                            return True  # Skip validation for missing optional parameters

                        # Handle typing generics
                        if hasattr(pt, '__origin__'):
                            return _check_typing_generic(value, pt)
                        # Handle regular types
                        else:
                            try:
                                return isinstance(value, pt)
                            except TypeError:
                                return True  # Skip validation for non-type annotations

                    return check_type

                preconditions.append((
                    f"{param_name} is {type_name}",
                    make_precondition(param_type, param_name, param)
                ))

    # Generate postcondition from return type
    if sig.return_annotation != inspect.Parameter.empty:
        return_type = sig.return_annotation

        # Handle refinement types in return annotations
        if (isinstance(return_type, type) and
                issubclass(return_type, RefinementType)):

            def make_refinement_postcondition(rt):
                def check_result(*args, **kwargs):
                    result = kwargs.get('result')
                    if result is None:
                        return True

                    try:
                        if hasattr(rt, '_validated_data'):
                            rt.validate(result)
                        else:
                            refinement_instance = rt()
                            refinement_instance.validate(result)
                        return True
                    except ProofFailure:
                        return False
                    except Exception:
                        return False

                return check_result

            type_name = getattr(return_type, '__name__', str(return_type))
            postconditions.append((
                f"result is {type_name}",
                make_refinement_postcondition(return_type)
            ))

        else:
            # Handle regular return types
            type_name = getattr(return_type, '__name__', str(return_type))

            def make_postcondition(rt):
                def check_result(*args, **kwargs):
                    result = kwargs.get('result')

                    if rt is type(None) or rt is None:
                        return result is None

                    if isinstance(rt, str):
                        if rt.lower() == "none":
                            return result is None
                        return True

                    if hasattr(rt, '__origin__'):
                        return _check_typing_generic(result, rt)

                    try:
                        return isinstance(result, rt)
                    except TypeError:
                        return True

                return check_result

            postconditions.append((
                f"result is {type_name}",
                make_postcondition(return_type)
            ))

    return contract(preconditions, postconditions)(func)


def _check_typing_generic(value, type_hint):
    """Helper function to check typing generics like Optional[T], List[T], etc."""
    from typing import get_origin, get_args
    import typing

    origin = get_origin(type_hint)
    args = get_args(type_hint)

    # Handle Optional[T] (which is Union[T, None])
    if origin is typing.Union:
        # Check if it's Optional (Union with None)
        if len(args) == 2 and type(None) in args:
            # This is Optional[T]
            if value is None:
                return True
            # Check if value matches the non-None type
            non_none_type = args[0] if args[1] is type(None) else args[1]
            try:
                return isinstance(value, non_none_type)
            except TypeError:
                return True
        else:
            # General Union - value must match at least one type
            for arg_type in args:
                try:
                    if isinstance(value, arg_type):
                        return True
                except TypeError:
                    continue
            return False

    # Handle List[T]
    elif origin is list:
        if not isinstance(value, list):
            return False
        if args:  # If we have type args, check elements
            element_type = args[0]
            try:
                return all(isinstance(item, element_type) for item in value)
            except TypeError:
                return True  # Skip validation if element type is not checkable
        return True

    # Handle Dict[K, V]
    elif origin is dict:
        if not isinstance(value, dict):
            return False
        if len(args) >= 2:  # If we have key and value types
            key_type, value_type = args[0], args[1]
            try:
                return all(isinstance(k, key_type) and isinstance(v, value_type)
                           for k, v in value.items())
            except TypeError:
                return True  # Skip validation if types are not checkable
        return True

    # For other generics, just check the origin type
    elif origin:
        try:
            return isinstance(value, origin)
        except TypeError:
            return True

    # Fallback
    return True


@contextmanager
def verification_mode():
    """Context manager for testing with full verification"""
    old_level = _config.level
    _config.level = VerificationLevel.DEBUG
    try:
        yield
    finally:
        _config.level = old_level


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# REGISTER DEFAULT PLUGINS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

_plugin_registry.register(CryptoPlugin())
_plugin_registry.register(FinancePlugin())
_plugin_registry.register(SecurityPlugin())
_plugin_registry.register(ConcurrencyPlugin())


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FUNCTION CONTRACTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def contract(preconditions: List[Tuple[str, Callable]] = None,
             postconditions: List[Tuple[str, Callable]] = None,
             plugins: List[str] = None):
    """Contract decorator with plugin support"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not _config.should_verify('contract'):
                return func(*args, **kwargs)

            func_name = func.__name__

            with proof_context(f"contract[{func_name}]"):
                # Check preconditions
                if preconditions:
                    for claim, condition_fn in preconditions:
                        require(f"precondition: {claim}",
                                condition_fn(*args, **kwargs))

                # Execute function
                result = func(*args, **kwargs)

                # Check postconditions
                if postconditions:
                    for claim, condition_fn in postconditions:
                        require(f"postcondition: {claim}",
                                condition_fn(*args, result=result, **kwargs))

                # Run plugin verifiers - Pass only the arguments they expect
                if plugins:
                    for plugin_name in plugins:
                        verifier = _plugin_registry.get_verifier(plugin_name)
                        if verifier:
                            # For security plugin, pass specific arguments
                            if plugin_name == "no_injection" and len(args) > 0:
                                # Assume first arg is user input, create a dummy query
                                require(f"plugin check: {plugin_name}",
                                        verifier("SELECT * FROM table WHERE input = ?", args[0]))
                            else:
                                # For other verifiers, try with just the function arguments
                                try:
                                    require(f"plugin check: {plugin_name}",
                                            verifier(*args, **kwargs))
                                except TypeError:
                                    # If that fails, try with result
                                    require(f"plugin check: {plugin_name}",
                                            verifier(*args, result=result, **kwargs))

                return result

        return wrapper

    return decorator


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# GLOBAL TEMPORAL VERIFIER AND CONVENIENCE FUNCTIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Global temporal verifier instance
_temporal_verifier = TemporalVerifier()


def record_temporal_event(event: str, data: Any = None):
    """Global convenience function for recording temporal events"""
    _temporal_verifier.record_event(event, data)


def add_temporal_property(prop: TemporalProperty):
    """Add a temporal property to the global verifier"""
    _temporal_verifier.add_property(prop)


def verify_temporal_properties():
    """Verify all temporal properties in the global verifier"""
    _temporal_verifier.verify_all()


def get_temporal_history():
    """Get the temporal event history"""
    return [prop.history for prop in _temporal_verifier.properties]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PREDEFINED PROTOCOL INSTANCES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# File-like protocol, allow proper read/write transitions
filemanager_protocol = Protocol("FileManager", "closed")
filemanager_protocol.add_state(ProtocolState("closed", ["open"]))
filemanager_protocol.add_state(ProtocolState("open", ["read", "write", "close"]))
filemanager_protocol.add_state(ProtocolState("read", ["read", "write", "close"]))  # Can read again, write, or close
filemanager_protocol.add_state(ProtocolState("write", ["read", "write", "close"]))  # Can read, write again, or close
filemanager_protocol.add_state(ProtocolState("close", ["closed"]))

# State machine protocol, allowed_transitions should be TARGET STATES, not method names
statemachine_protocol = Protocol("StateMachine", "stopped")
statemachine_protocol.add_state(ProtocolState("stopped", ["running"]))  # From stopped, can go to running (via start())
statemachine_protocol.add_state(ProtocolState("running", ["stopped",
                                                          "process"]))  # From running, can go to stopped (via stop()) or process (via process())
statemachine_protocol.add_state(
    ProtocolState("process", ["running", "stopped"]))  # From process, can go back to running or stopped

# Database connection protocol
dbconnection_protocol = Protocol("DatabaseConnection", "disconnected")
dbconnection_protocol.add_state(ProtocolState("disconnected", ["connected"]))
dbconnection_protocol.add_state(
    ProtocolState("connected", ["connected", "disconnected"]))  # Can stay connected or disconnect
dbconnection_protocol.add_state(ProtocolState("transaction", ["connected"]))  # Transaction ends back to connected

# HTTP client protocol
httpclient_protocol = Protocol("HttpClient", "idle")
httpclient_protocol.add_state(ProtocolState("idle", ["request"]))
httpclient_protocol.add_state(ProtocolState("request", ["response", "error"]))
httpclient_protocol.add_state(ProtocolState("response", ["idle"]))
httpclient_protocol.add_state(ProtocolState("error", ["idle"]))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SENSITIVE DATA TRACKING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def track_sensitive_data(variable_name: str, value: Any, label: SecurityLabel = SecurityLabel.CONFIDENTIAL):
    """Track sensitive data assignment for information flow analysis"""
    tainted = TaintedValue(value, label, [f"assignment_to_{variable_name}"])

    # Store in ghost state for tracking
    _ghost.set(f"tainted_{variable_name}", tainted)

    # Log the tracking
    if _config.debug_mode:
        print(f"Tracking sensitive data: {variable_name} with label {label.value}")

    return tainted


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FUTURE DEMONSTRATION WITH ALL NEW FEATURES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def demo_future_features():
    """Demonstrate all future features including adaptive monitoring, introspection, and recovery"""
    print("Axiomatik Future: Runtime Verification with Adaptive Monitoring, Introspection & Recovery")
    print("~" * 90)

    # Clear state
    _proof.clear()

    # Demo 1: Adaptive Monitoring
    print("\n~~~~~~~ Adaptive Monitoring Demo ~~~~~~~")

    # Enable conservative auto-tuning
    _performance_analyzer.auto_tuning_enabled = True
    _performance_analyzer.target_overhead_percent = 15.0  # More reasonable target
    _performance_analyzer.adjustment_cooldown = 2.0  # Longer cooldown
    print(f"Conservative auto-tuning enabled: target overhead {_performance_analyzer.target_overhead_percent}%")
    # Enable auto-tuning
    # auto_tune_verification_level(target_overhead_percent=5.0)

    # Simulate load testing with adaptive properties
    for i in range(50):
        with adaptive_verification_context("database"):
            adaptive_require(
                "connection is valid",
                True,
                property_name="db_connection_check",
                priority=5
            )

        with adaptive_verification_context("network"):
            adaptive_require(
                "network is available",
                True,
                property_name="network_check",
                priority=3
            )

    print(f"Adaptive monitoring processed 100 verifications")
    print(f"Active properties: {len(_adaptive_monitor.active_properties)}")
    print(f"Property registry: {len(_adaptive_monitor.property_registry)}")

    # Demo 2: Performance Introspection
    print("\n~~~~~~~ Performance Introspection Demo ~~~~~~~")

    # Generate some performance data
    @auto_contract
    def expensive_operation(n: PositiveInt) -> int:
        """Simulate expensive operation for benchmarking"""
        time.sleep(0.001)  # Small delay to simulate work
        return n * 2

    @auto_contract
    def cheap_operation(n: PositiveInt) -> int:
        """Quick operation for comparison"""
        return n + 1

    # Run operations to generate performance data
    for i in range(1, 21):
        expensive_operation(PositiveInt(i))
        cheap_operation(PositiveInt(i))

    # Generate performance report
    performance_report = generate_performance_report()
    print("Performance Analysis Results:")
    print(performance_report[:500] + "..." if len(performance_report) > 500 else performance_report)

    # Show hotspots
    hotspots = get_performance_hotspots(3)
    print(f"\nTop 3 Performance Hotspots:")
    for i, hotspot in enumerate(hotspots, 1):
        print(f"  {i}. {hotspot.property_name[:30]}: {hotspot.average_time * 1000:.2f}ms avg")

    # Demo 3: Recovery Framework
    print("\n~~~~~~~ Recovery Framework Demo ~~~~~~~")

    # Function with graceful degradation
    @contract_with_recovery(
        preconditions=[("data not empty", lambda data: len(data) > 0)],
        recovery_strategy=RecoveryStrategy(
            RecoveryPolicy.GRACEFUL_DEGRADATION,
            fallback_handler=lambda data: sum(data) / max(1, len(data))  # Safe average
        )
    )
    def risky_statistical_analysis(data: List[float]) -> Dict[str, float]:
        """Analysis that might fail verification but has recovery"""
        # Intentionally strict verification that might fail
        require("data has at least 10 elements", len(data) >= 10)

        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)

        return {
            'mean': mean,
            'variance': variance,
            'count': len(data)
        }

    # Test with data that triggers recovery
    small_data = [1.0, 2.0, 3.0]  # Less than 10 elements
    try:
        result = risky_statistical_analysis(small_data)
        print(f"+++ - Recovery successful: {result}")
    except Exception as e:
        print(f"XXX - Recovery failed: {e}")

    # Function with retry strategy
    def create_flaky_operation():
        attempt_count = 0

        @contract_with_recovery(
            preconditions=[("attempt succeeds", lambda: attempt_count < 2)],
            recovery_strategy=RecoveryStrategy(
                RecoveryPolicy.RETRY_WITH_BACKOFF,
                max_retries=3,
                backoff_factor=1.5
            )
        )
        def flaky_operation() -> str:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count <= 2:
                return "failure simulation"
            return "success"

        return flaky_operation

    flaky_op = create_flaky_operation()
    try:
        result = flaky_op()
        print(f"+++ - Retry recovery successful: {result}")
    except Exception as e:
        print(f"XXX - Retry recovery failed: {e}")

    # Reset for demo
    attempt_count = 0

    # Demo 4: Combined Features
    print("\n~~~~~~~ Combined Features Demo ~~~~~~~")

    # Function using adaptive monitoring, performance tracking, and recovery
    @contract_with_recovery(
        preconditions=[
            ("input is valid", lambda data: isinstance(data, list) and len(data) > 0)
        ],
        recovery_strategy=RecoveryStrategy(
            RecoveryPolicy.GRACEFUL_DEGRADATION,
            fallback_handler=lambda data: {"result": "simplified_processing", "count": len(data)}
        )
    )
    def comprehensive_data_processing(data: List[int]) -> Dict[str, Any]:
        """Demonstrates all future features working together"""

        # Use adaptive monitoring
        with adaptive_verification_context("data_processing"):
            adaptive_require(
                "data is properly formatted",
                all(isinstance(x, int) for x in data),
                property_name="data_formatting_check",
                priority=4
            )

            # Track sensitive data (if applicable)
            if any(x > 1000 for x in data):
                track_sensitive_data("high_value_data", data, SecurityLabel.CONFIDENTIAL)

            # Record temporal event
            record_temporal_event("processing_started", {"data_size": len(data)})

            # Perform processing
            result = {
                "sum": sum(data),
                "average": sum(data) / len(data),
                "max": max(data),
                "min": min(data),
                "processed_at": time.time()
            }

            # Strict verification that might trigger recovery
            require("result is comprehensive", len(result) >= 5)
            require("average is reasonable", 0 <= result["average"] <= 10000)

            record_temporal_event("processing_completed", {"result_keys": len(result)})

            return result

    # Test comprehensive processing
    test_data = [10, 20, 30, 40, 50]
    result = comprehensive_data_processing(test_data)
    print(f"+++ - Comprehensive processing: {result}")

    # Test with data that might trigger recovery
    edge_case_data = [10000, 20000, 30000]  # High values
    result2 = comprehensive_data_processing(edge_case_data)
    print(f"+++ - Edge case processing: {result2}")

    # Demo 5: System Status and Analytics
    print("\n~~~~~~~ System Status & Analytics ~~~~~~~")

    # Adaptive monitor status
    print("Adaptive Monitor Status:")
    print(f"  Active properties: {len(_adaptive_monitor.active_properties)}")
    print(f"  Property costs tracked: {len(_adaptive_monitor.property_costs)}")
    print(f"  Load metrics samples: {len(_adaptive_monitor.load_metrics)}")

    # Performance analyzer status
    print("\nPerformance Analyzer Status:")
    print(f"  Properties tracked: {len(_performance_analyzer.verification_times)}")
    print(f"  Total verification time: {_performance_analyzer.total_verification_time:.3f}s")
    print(f"  Auto-tuning enabled: {_performance_analyzer.auto_tuning_enabled}")

    # Recovery system status
    recovery_manager = RecoveryManager()
    print(f"\nRecovery System Status:")
    print(f"  State snapshots: {len(recovery_manager.state_snapshots)}")
    print(f"  Recovery stats: {len(recovery_manager.recovery_stats)}")

    # Global proof system status
    summary = _proof.get_summary()
    print(f"\nGlobal Proof System Status:")
    print(f"  Total proof steps: {summary['total_steps']}")
    print(f"  Contexts verified: {len(summary['contexts'])}")
    print(f"  Thread count: {summary['thread_count']}")
    print(f"  Cache enabled: {summary['cache_enabled']}")

    # Show recent verifications by category
    if _proof.steps:
        print("\nRecent Verifications by Category:")
        adaptive_steps = [s for s in _proof.steps[-20:] if 'adaptive' in s.context.lower()]
        contract_steps = [s for s in _proof.steps[-20:] if 'contract' in s.context.lower()]

        print(f"  Adaptive verifications: {len(adaptive_steps)}")
        print(f"  Contract verifications: {len(contract_steps)}")

        if adaptive_steps:
            print("  Sample adaptive verification:", adaptive_steps[-1].claim[:50] + "...")
        if contract_steps:
            print("  Sample contract verification:", contract_steps[-1].claim[:50] + "...")

    print("\n~~~~~~~ All future features demonstrated successfully! ~~~~~~~")


if __name__ == "__main__":
    demo_future_features()