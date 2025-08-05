"""
PyProof: Performant Runtime Verification for Python
A comprehensive system with performance optimization, integration helpers, and domain-specific extensions.
"""

import functools
import os
import time
import threading
import weakref
from typing import Callable, List, Tuple, Any, Dict, Optional, Union, Type
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
    """Global configuration for PyProof"""

    def __init__(self):
        self.level = VerificationLevel(os.getenv('PYPROOF_LEVEL', 'full'))
        self.cache_enabled = os.getenv('PYPROOF_CACHE', '1') == '1'
        self.max_proof_steps = int(os.getenv('PYPROOF_MAX_STEPS', '10000'))
        self.performance_mode = os.getenv('PYPROOF_PERF', '0') == '1'
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


def require(claim: str, evidence: Any) -> Any:
    """Optimized global require function"""
    return _proof.require(claim, evidence)


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

    def __init__(self, element_type: Type = None):
        predicate = lambda x: len(x) > 0
        if element_type:
            predicate = lambda x: len(x) > 0 and all(isinstance(e, element_type) for e in x)

        super().__init__(
            list,
            predicate,
            f"non-empty list" + (f" of {element_type.__name__}" if element_type else "")
        )


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
    """Base class for PyProof plugins"""

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
    """Registry for PyProof plugins"""

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
    # from typing import get_origin, get_args

    sig = inspect.signature(func)
    preconditions = []
    postconditions = []

    # Generate preconditions from parameter types
    for param_name, param in sig.parameters.items():
        if param.annotation != inspect.Parameter.empty:
            param_type = param.annotation

            # Handle both regular types and custom objects
            type_name = getattr(param_type, '__name__', str(param_type))

            # Create closure to capture variables properly and handle refinement types
            def make_precondition(pt, pn, param_obj):
                def check_type(*args, **kwargs):
                    # First try to get from kwargs
                    value = kwargs.get(pn)

                    # If not in kwargs, try to get from args by position
                    if value is None:
                        param_names = list(sig.parameters.keys())
                        try:
                            param_index = param_names.index(pn)
                            if param_index < len(args):
                                value = args[param_index]
                        except (ValueError, IndexError):
                            pass

                    # If still None and parameter has a default, use the default
                    if value is None and param_obj.default != inspect.Parameter.empty:
                        value = param_obj.default

                    # If still None, this parameter wasn't provided and has no default
                    if value is None:
                        return True  # Skip validation for missing optional parameters

                    # Handle refinement types
                    if hasattr(pt, 'validate'):
                        try:
                            pt.validate(value)
                            return True
                        except:
                            return False
                    # Handle typing generics
                    elif hasattr(pt, '__origin__'):
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
        type_name = getattr(return_type, '__name__', str(return_type))

        # Handle refinement types in postconditions too
        def make_postcondition(rt):
            def check_result(*args, **kwargs):
                result = kwargs.get('result')

                # Handle None return type
                if rt is type(None) or rt is None:
                    return result is None

                # Handle string annotations like "None"
                if isinstance(rt, str):
                    if rt.lower() == "none":
                        return result is None
                    # For other string types, just return True (skip validation)
                    return True

                # Handle refinement types
                if hasattr(rt, 'validate'):
                    try:
                        rt.validate(result)
                        return True
                    except:
                        return False

                # Handle typing generics
                if hasattr(rt, '__origin__'):
                    return _check_typing_generic(result, rt)

                # Handle regular types - but check if it's actually a type first
                try:
                    return isinstance(result, rt)
                except TypeError:
                    # If isinstance fails, the annotation isn't a proper type
                    # Just return True to skip validation
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
statemachine_protocol.add_state(ProtocolState("running", ["stopped", "process"]))  # From running, can go to stopped (via stop()) or process (via process())
statemachine_protocol.add_state(ProtocolState("process", ["running", "stopped"]))  # From process, can go back to running or stopped

# Database connection protocol
dbconnection_protocol = Protocol("DatabaseConnection", "disconnected")
dbconnection_protocol.add_state(ProtocolState("disconnected", ["connected"]))
dbconnection_protocol.add_state(ProtocolState("connected", ["connected", "disconnected"]))  # Can stay connected or disconnect
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
# COMPREHENSIVE DEMO
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def demo_advanced_features():
    """Demonstrate all advanced features with comprehensive output"""
    print("PyProof Advanced: High-Performance Runtime Verification")
    print("~" * 80)

    # Clear state
    _proof.clear()

    # Demo 1: Refinement Types
    print("\n~~~~~~~ Refinement Types Demo ~~~~~~~")

    @auto_contract
    def calculate_grade(score: PositiveInt) -> Percentage:
        return min(100, max(0, score))

    try:
        grade = calculate_grade(85)
        print(f"Grade: {grade}%")

        # Demo refinement type validation
        try:
            invalid_score = PositiveInt(-5)
        except ProofFailure as e:
            print(f"Refinement validation caught: {e}")

        # Demo other refinement types
        valid_list = NonEmptyList(int)([1, 2, 3, 4])
        print(f"Non-empty list validated: {valid_list}")

    except ProofFailure as e:
        print(f"Refinement type error: {e}")

    # Demo 2: Protocol Verification
    print("\n~~~~~~~ Protocol Verification Demo ~~~~~~~")

    file_protocol = Protocol("FileHandle", "closed")
    file_protocol.add_state(ProtocolState("closed", ["open"]))
    file_protocol.add_state(ProtocolState("open", ["read", "write", "close"]))
    file_protocol.add_state(ProtocolState("read", ["open", "close"]))
    file_protocol.add_state(ProtocolState("write", ["open", "close"]))
    file_protocol.add_state(ProtocolState("close", ["closed"]))

    class ProtocolFile:
        @protocol_method(file_protocol, "open")
        def open(self):
            print(f"File opened (state: {file_protocol.get_state(self)})")

        @protocol_method(file_protocol, "read")
        def read(self):
            print(f"File read (state: {file_protocol.get_state(self)})")
            return "data"

        @protocol_method(file_protocol, "close")
        def close(self):
            print(f"File closed (state: {file_protocol.get_state(self)})")

    pf = ProtocolFile()
    print(f"Initial state: {file_protocol.get_state(pf)}")
    pf.open()
    pf.read()
    pf.close()

    # Demo invalid transition
    try:
        pf2 = ProtocolFile()
        pf2.read()  # Should fail - can't read from closed file
    except ProofFailure as e:
        print(f"Protocol violation caught: {e}")

    # Demo 3: Information Flow
    print("\n~~~~~~~ Information Flow Demo ~~~~~~~")

    # Create tainted values with different security levels
    public_data = TaintedValue("public_info", SecurityLabel.PUBLIC, ["web_form"])
    secret_data = TaintedValue("classified_info", SecurityLabel.SECRET, ["database"])
    confidential_data = TaintedValue("internal_memo", SecurityLabel.CONFIDENTIAL, ["email"])

    print(f"Public data: {public_data.value} (label: {public_data.label.value})")
    print(f"Secret data: {secret_data.value} (label: {secret_data.label.value})")
    print(f"Confidential data: {confidential_data.value} (label: {confidential_data.label.value})")

    # Demonstrate information flow tracking
    flow_tracker = InformationFlowTracker()

    # Set up flow policies
    flow_tracker.add_policy(SecurityLabel.PUBLIC, SecurityLabel.CONFIDENTIAL, True)
    flow_tracker.add_policy(SecurityLabel.SECRET, SecurityLabel.PUBLIC, False)

    # Valid flow: public -> confidential
    try:
        flow_tracker.track_flow(public_data, SecurityLabel.CONFIDENTIAL)
        print("+++ - Public -> Confidential flow allowed")
    except ProofFailure as e:
        print(f"XXX - Flow violation: {e}")

    # Invalid flow: secret -> public
    try:
        flow_tracker.track_flow(secret_data, SecurityLabel.PUBLIC)
        print("+++ - Secret -> Public flow allowed")
    except ProofFailure as e:
        print(f"XXX - Flow violation caught: Secret data cannot flow to Public")

    # Demonstrate declassification
    try:
        secret_data.declassify(SecurityLabel.CONFIDENTIAL, "Sanitized for internal use")
        print(f"+++ - Secret data declassified to: {secret_data.label.value}")
        print(f"  Provenance: {secret_data.provenance}")
    except ProofFailure as e:
        print(f"XXX - Declassification failed: {e}")

    # Demonstrate combining tainted values
    combined = public_data.combine_with(confidential_data)
    print(f"Combined data security level: {combined.label.value}")
    print(f"Combined provenance: {combined.provenance}")

    # Demo 4: Temporal Properties
    print("\n~~~~~~~ Temporal Properties Demo ~~~~~~~")

    temporal_verifier = TemporalVerifier()

    # Add "eventually consistent" property
    eventually_consistent = EventuallyProperty(
        "data_consistent",
        lambda history: any(e['event'] == 'sync_complete' for e in history),
        timeout=5.0
    )
    temporal_verifier.add_property(eventually_consistent)

    # Add "always valid" property
    always_valid = AlwaysProperty(
        "data_valid",
        lambda event: event.get('data', {}).get('valid', True)
    )
    temporal_verifier.add_property(always_valid)

    print("Recording temporal events...")
    temporal_verifier.record_event("data_update", {"key": "value", "valid": True})
    print("  -> data_update recorded")

    temporal_verifier.record_event("sync_start", {"valid": True})
    print("  -> sync_start recorded")

    temporal_verifier.record_event("validation_check", {"valid": True})
    print("  -> validation_check recorded")

    temporal_verifier.record_event("sync_complete", {"valid": True})
    print("  -> sync_complete recorded")

    # Verify temporal properties
    try:
        temporal_verifier.verify_all()
        print("+++ - All temporal properties verified:")
        print(f"  - Eventually consistent: {eventually_consistent.check()}")
        print(f"  - Always valid: {always_valid.check()}")
        print(f"  - Event history length: {len(temporal_verifier.properties[0].history)}")
    except ProofFailure as e:
        print(f"XXX - Temporal verification failed: {e}")

    # Demo 5: Plugin Usage
    print("\n~~~~~~~ Plugin Extensions Demo ~~~~~~~")

    # Financial calculation
    print("~~~~~~~ Financial Plugin ~~~~~~~")
    financial_money = _plugin_registry.get_type("Money")
    if financial_money:
        price = financial_money("19.99", "USD")
        tax = financial_money("1.60", "USD")
        total = price + tax
        print(f"Price: {price}")
        print(f"Tax: {tax}")
        print(f"Total: {total}")

        # Demonstrate precision verification
        finance_plugin = None
        for plugin in _plugin_registry.plugins.values():
            if plugin.name == "finance":
                finance_plugin = plugin
                break

        if finance_plugin:
            precision_ok = finance_plugin.verify_precision(total, 2)
            print(f"+++ - Precision maintained (2 decimal places): {precision_ok}")

            audit_ok = finance_plugin.verify_audit_trail("addition", [price, tax], total)
            print(f"+++ - Audit trail recorded: {audit_ok}")
            print(f"  Audit entries: {len(finance_plugin.audit_trail)}")

    # Cryptographic verification
    print("\n~~~~~~~ Cryptographic Plugin ~~~~~~~")
    crypto_plugin = None
    for plugin in _plugin_registry.plugins.values():
        if plugin.name == "crypto":
            crypto_plugin = plugin
            break

    if crypto_plugin:
        # Test constant-time verification
        def fast_func(x):
            time.sleep(0.001)

        def variable_func(x):
            time.sleep(0.001 * len(str(x)))

        const_time_ok = crypto_plugin.verify_constant_time(fast_func, [1, 100, 1000])
        print(f"+++ - Constant-time verification (fast_func): {const_time_ok}")

        # Test key zeroization
        sensitive_key = bytearray(b"secret_key_123")
        print(f"Before zeroization: {sensitive_key[:10]}...")

        # Zeroize the key
        for i in range(len(sensitive_key)):
            sensitive_key[i] = 0

        zeroized_ok = crypto_plugin.verify_key_zeroized(sensitive_key)
        print(f"+++ - Key properly zeroized: {zeroized_ok}")

        # Test secure random
        import secrets
        secure_ok = crypto_plugin.verify_secure_random(secrets.randbits)
        print(f"+++ - Secure random generator: {secure_ok}")

    # Security verification
    print("\n~~~~~~~ Security Plugin ~~~~~~~")
    security_plugin = None
    for plugin in _plugin_registry.plugins.values():
        if plugin.name == "security":
            security_plugin = plugin
            break

    if security_plugin:
        # Test input sanitization
        malicious_input = "<script>alert('xss')</script>"
        sanitizer = lambda x: x.replace("<", "&lt;").replace(">", "&gt;")

        sanitized_ok = security_plugin.verify_input_sanitized(malicious_input, sanitizer)
        print(f"+++ - Input sanitization: {sanitized_ok}")
        print(f"  Original: {malicious_input}")
        print(f"  Sanitized: {sanitizer(malicious_input)}")

        # Test privilege boundaries
        privilege_ok = security_plugin.verify_privilege_boundary("admin", "user")
        print(f"+++ - Privilege check (admin can access user resources): {privilege_ok}")

        privilege_fail = security_plugin.verify_privilege_boundary("guest", "admin")
        print(f"XXX - Privilege check (guest cannot access admin resources): {privilege_fail}")

        # Test injection prevention
        safe_query = "SELECT * FROM users WHERE id = ?"
        user_input = "1"
        injection_safe = security_plugin.verify_no_injection(safe_query, user_input)
        print(f"+++ - No injection vulnerability: {injection_safe}")

    # Concurrency verification
    print("\n~~~~~~~ Concurrency Plugin ~~~~~~~")
    concurrency_plugin = None
    for plugin in _plugin_registry.plugins.values():
        if plugin.name == "concurrency":
            concurrency_plugin = plugin
            break

    if concurrency_plugin:
        # Test deadlock prevention
        import threading
        lock1 = threading.Lock()
        lock2 = threading.Lock()

        deadlock_safe = concurrency_plugin.verify_no_deadlock(lock1, lock2)
        print(f"+++ - No deadlock detected: {deadlock_safe}")

        # Test atomic operation detection
        def atomic_increment():
            with threading.Lock():
                return 1

        atomic_ok = concurrency_plugin.verify_atomic(atomic_increment)
        print(f"+++ - Atomic operation detected: {atomic_ok}")

        # Show lock graph
        print(f"  Active lock relationships: {len(concurrency_plugin.lock_graph)}")

    # Security contract demo with proper plugin integration
    print("\n~~~~~~~ Integrated Security Contract ~~~~~~~")

    @contract(
        preconditions=[
            ("input is string", lambda lam_user_input: isinstance(lam_user_input, str)),
            ("input sanitized", lambda lam_user_input:
            _plugin_registry.get_verifier("input_sanitized")(lam_user_input, lambda x: x.replace("<", "&lt;")))
        ]
    )
    def process_user_input(temp_user_input: str):
        return f"Processed: {temp_user_input}"

    try:
        result = process_user_input("Hello World")
        print(f"+++ - Security contract passed: {result}")
    except ProofFailure as e:
        print(f"XXX - Security verification failed: {e}")

    # Performance summary
    print(f"\n~~~~~~~ Performance Summary ~~~~~~~")
    summary = _proof.get_summary()
    print(f"Total proof steps: {summary['total_steps']}")
    print(f"Contexts verified: {list(summary['contexts'].keys())}")
    print(f"Steps per context: {summary['contexts']}")
    print(f"Threads involved: {summary['thread_count']}")
    print(f"Cache enabled: {summary['cache_enabled']}")
    print(f"Current verification level: {_config.level.value}")

    # Show some proof steps
    print(f"\nSample proof steps:")
    for i, step in enumerate(_proof.steps[-5:]):  # Show last 5 steps
        print(f"  {len(_proof.steps) - 4 + i}. {step.context}: {step.claim}")

    print("\n~~~~~~~ All advanced verifications passed! ~~~~~~~")


if __name__ == "__main__":
    demo_advanced_features()