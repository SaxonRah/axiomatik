import threading
import time
from collections import deque, defaultdict
from contextlib import contextmanager
from typing import Set, Dict, Callable, Any

from pyproof.pyproof import ProofFailure, require


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


# Enhanced require function with adaptive monitoring
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


# Dynamic property loading
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


# Usage with context-aware loading
@contextmanager
def adaptive_verification_context(context: str):
    """Context manager that loads appropriate properties"""
    property_manager = PropertyManager()
    property_manager.load_properties_for_context(context)
    try:
        yield
    finally:
        property_manager.unload_properties_for_context(context)


# Example usage:
with adaptive_verification_context("database"):
    # Database-specific properties are automatically loaded
    adaptive_require("connection is valid", db.is_connected(),
                     property_name="db_connection_valid", priority=5)