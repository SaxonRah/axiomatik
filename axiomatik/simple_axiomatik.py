"""
Simple Axiomatik: Verification that feels like Python
"""

import functools
import time
import inspect
from typing import TypeVar, Generic, Dict, Any, Callable, Union, get_origin, get_args
from contextlib import contextmanager

# Import the underlying Axiomatik system
try:
    import axiomatik.axiomatik as _ax
    from axiomatik.axiomatik import (
        ProofFailure, require as _require, proof_context as _proof_context,
        Protocol as _Protocol, ProtocolState as _ProtocolState,
        protocol_method as _protocol_method, VerificationLevel, Config
    )
except ImportError:
    # Fallback if axiomatik module structure is different
    try:
        import axiomatik as _ax
        from axiomatik import (
            ProofFailure, require as _require, proof_context as _proof_context,
            Protocol as _Protocol, ProtocolState as _ProtocolState,
            protocol_method as _protocol_method, VerificationLevel, Config
        )
    except ImportError:
        raise ImportError("Cannot import Axiomatik. Please ensure it's installed.")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SIMPLIFIED ERROR HANDLING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class VerificationError(Exception):
    """User-friendly verification error with helpful messages"""

    def __init__(self, function_name: str, condition: str, message: str,
                 values: Dict[str, Any] = None, suggestion: str = None):
        self.function_name = function_name
        self.condition = condition
        self.message = message
        self.values = values or {}
        self.suggestion = suggestion

        # Build helpful error message
        parts = [f"{function_name}() verification failed"]
        parts.append(f"Condition: {condition}")
        if message:
            parts.append(f"Message: {message}")

        if values:
            value_strs = [f"{k}={v}" for k, v in values.items()]
            parts.append(f"Values: {', '.join(value_strs)}")

        if suggestion:
            parts.append(f"Suggestion: {suggestion}")

        super().__init__("\n".join(parts))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SIMPLE MODE CONFIGURATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class _SimpleConfig:
    """Simple configuration that maps to underlying Axiomatik config"""

    def __init__(self):
        self._mode = "dev"
        self._performance_tracking = False
        self._update_axiomatik_config()

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str):
        if value not in ["dev", "prod", "test", "off"]:
            raise ValueError(f"Invalid mode: {value}. Use 'dev', 'prod', 'test', or 'off'")

        self._mode = value
        self._update_axiomatik_config()

    def _update_axiomatik_config(self):
        """Update underlying Axiomatik configuration"""
        mode_mapping = {
            "dev": VerificationLevel.FULL,
            "prod": VerificationLevel.CONTRACTS,
            "test": VerificationLevel.DEBUG,
            "off": VerificationLevel.OFF
        }

        _ax._config.level = mode_mapping[self._mode]
        _ax._config.performance_mode = (self._mode == "prod")
        _ax._config.debug_mode = (self._mode == "test")


# Global config instance
_config = _SimpleConfig()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SIMPLE TYPE ALIASES FOR REFINEMENT TYPES TO HANDLE SUBSCRIPTED GENERICS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

T = TypeVar('T')


def _safe_isinstance(instance, type_hint):
    """Safe isinstance check that handles subscripted generics"""
    # Handle subscripted generics by checking the origin type
    origin = get_origin(type_hint)
    if origin is not None:
        # This is a subscripted generic like List[int], Dict[str, int], etc.
        return isinstance(instance, origin)
    else:
        # This is a regular type
        return isinstance(instance, type_hint)


class Positive(Generic[T]):
    """Positive number type alias"""

    def __class_getitem__(cls, item):
        if item in (int, float):
            return _PositiveNumber(item)
        return super().__class_getitem__(item)


class _PositiveNumber:
    def __init__(self, base_type):
        self.base_type = base_type

    def __instancecheck__(self, instance):
        return _safe_isinstance(instance, self.base_type) and instance > 0


class NonEmpty(Generic[T]):
    """Non-empty container type alias"""

    def __class_getitem__(cls, item):
        return _NonEmptyContainer(item)


class _NonEmptyContainer:
    def __init__(self, base_type):
        self.base_type = base_type

    def __instancecheck__(self, instance):
        # Use _safe_isinstance to handle subscripted generics
        return _safe_isinstance(instance, self.base_type) and len(instance) > 0


class Range(Generic[T]):
    """Range-constrained type alias"""

    def __class_getitem__(cls, params):
        if isinstance(params, tuple) and len(params) == 3:
            base_type, min_val, max_val = params
            return _RangeType(base_type, min_val, max_val)
        return super().__class_getitem__(params)


class _RangeType:
    def __init__(self, base_type, min_val, max_val):
        self.base_type = base_type
        self.min_val = min_val
        self.max_val = max_val

    def __instancecheck__(self, instance):
        return (_safe_isinstance(instance, self.base_type) and
                self.min_val <= instance <= self.max_val)


# Convenient aliases
PositiveInt = Positive[int]
PositiveFloat = Positive[float]
NonEmptyList = NonEmpty[list]
NonEmptyStr = NonEmpty[str]
Percentage = Range[int, 0, 100]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SIMPLIFIED REQUIRE FUNCTION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def require(condition: Any, message: str = "", **context) -> Any:
    """Simplified require function with better error messages"""

    # Get calling function info for better error messages
    frame = inspect.currentframe().f_back
    function_name = frame.f_code.co_name

    # Extract variable names from condition if it's a boolean expression
    condition_str = message or str(condition)

    try:
        result = _require(condition_str, condition)
        return result
    except ProofFailure as e:
        # Convert to user-friendly error
        values = {}

        # Try to extract variable values from the calling frame
        if hasattr(condition, '__code__'):
            # This is a callable - try to get variable values
            try:
                local_vars = frame.f_locals
                global_vars = frame.f_globals

                # Simple variable extraction (could be enhanced)
                for var_name, var_value in local_vars.items():
                    if not var_name.startswith('_'):
                        values[var_name] = var_value
            except:
                pass

        # Generate suggestion based on common patterns
        suggestion = _generate_suggestion(condition_str, values)

        raise VerificationError(
            function_name=function_name,
            condition=condition_str,
            message=message,
            values=values,
            suggestion=suggestion
        )


def _generate_suggestion(condition: str, values: Dict[str, Any]) -> str:
    """Generate helpful suggestions based on the condition"""
    condition_lower = condition.lower()

    if "!= 0" in condition or "division" in condition_lower:
        return "Check that denominator is not zero before division"
    elif "len(" in condition and "> 0" in condition:
        return "Ensure the collection is not empty before processing"
    elif "none" in condition_lower:
        return "Check for null/None values before use"
    elif "range" in condition_lower or "<" in condition or ">" in condition:
        return "Verify input values are within expected range"
    else:
        return "Review the input values and function requirements"


# Shorthand
req = require  # Common abbreviation


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ENSURE FUNCTION (POSTCONDITIONS)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def ensure(condition: Any, message: str = "") -> Any:
    """Ensure function for postconditions (same as require but semantically different)"""
    return require(condition, message)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CORE DECORATORS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def verify(func: Callable = None, *, track_performance: bool = False) -> Callable:
    """
    Simple verification decorator that enables require/ensure statements

    Usage:
        @verify
        def divide(a: float, b: float) -> float:
            require(b != 0, "Cannot divide by zero")
            result = a / b
            ensure(abs(result * b - a) < 1e-10, "Division check failed")
            return result
    """

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if _config.mode == "off":
                return f(*args, **kwargs)

            start_time = time.perf_counter() if track_performance else None

            try:
                with _proof_context(f"verify[{f.__name__}]"):
                    result = f(*args, **kwargs)

                if track_performance:
                    end_time = time.perf_counter()
                    _record_performance(f.__name__, end_time - start_time)

                return result

            except ProofFailure as e:
                # Convert to user-friendly error
                raise VerificationError(
                    function_name=f.__name__,
                    condition="internal verification",
                    message=str(e),
                    suggestion="Check function preconditions and postconditions"
                )

        return wrapper

    # Handle both @verify and @verify() usage
    if func is None:
        return decorator
    else:
        return decorator(func)


def checked(func: Callable) -> Callable:
    """
    Automatic verification from type hints

    Usage:
        @checked
        def process_items(items: NonEmpty[list[str]]) -> Positive[int]:
            return len([item.upper() for item in items])
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if _config.mode == "off":
            return func(*args, **kwargs)

        with _proof_context(f"checked[{func.__name__}]"):
            # Check preconditions from type hints
            _check_type_preconditions(func, args, kwargs)

            # Execute function
            result = func(*args, **kwargs)

            # Check postconditions from return type hint
            _check_type_postconditions(func, result)

            return result

    return wrapper


def _check_type_preconditions(func: Callable, args: tuple, kwargs: dict):
    """Check preconditions based on type hints"""
    sig = inspect.signature(func)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()

    for param_name, param in sig.parameters.items():
        if param.annotation == inspect.Parameter.empty:
            continue

        value = bound_args.arguments.get(param_name)
        if value is None:
            continue

        _check_type_constraint(param.annotation, value, param_name, func.__name__)


def _check_type_postconditions(func: Callable, result: Any):
    """Check postconditions based on return type hint"""
    sig = inspect.signature(func)
    if sig.return_annotation == inspect.Parameter.empty:
        return

    _check_type_constraint(sig.return_annotation, result, "return value", func.__name__)


def _check_type_constraint(annotation: Any, value: Any, param_name: str, func_name: str):
    """Check if value satisfies type constraint with subscripted generics"""

    # Handle our custom type aliases
    if hasattr(annotation, '__instancecheck__'):
        try:
            if not annotation.__instancecheck__(value):
                raise VerificationError(
                    function_name=func_name,
                    condition=f"{param_name} satisfies {annotation}",
                    message=f"Type constraint violation for {param_name}",
                    values={param_name: value},
                    suggestion=f"Ensure {param_name} meets the required constraints"
                )
        except TypeError:
            # Subscripted generics can't be used with isinstance
            # Fall back to checking the origin type
            origin = get_origin(annotation)
            if origin is not None:
                # Handle Union types (including Optional)
                if origin is Union:
                    args = get_args(annotation)
                    # Check if value matches any of the union members
                    for arg in args:
                        try:
                            _check_type_constraint(arg, value, param_name, func_name)
                            return  # If any check passes, we're good
                        except VerificationError:
                            continue  # Try next union member
                    # If we get here, none of the union members matched
                    union_types = [getattr(arg, '__name__', str(arg)) for arg in args]
                    raise VerificationError(
                        function_name=func_name,
                        condition=f"{param_name} matches one of {union_types}",
                        message=f"Value doesn't match any of the union types: {union_types}",
                        values={param_name: value},
                        suggestion=f"Ensure {param_name} matches one of the allowed types"
                    )
                else:
                    # Regular generic type - skip isinstance check for complex generics
                    pass
            # If we can't check it, just skip the validation
            return

    # Handle subscripted generics using get_origin
    elif hasattr(annotation, '__origin__'):
        origin = get_origin(annotation)
        if origin is not None:
            # Handle Union types (including Optional)
            if origin is Union:
                args = get_args(annotation)
                # Check if value matches any of the union members
                for arg in args:
                    try:
                        _check_type_constraint(arg, value, param_name, func_name)
                        return  # If any check passes, we're good
                    except VerificationError:
                        continue  # Try next union member
                # If we get here, none of the union members matched
                union_types = [getattr(arg, '__name__', str(arg)) for arg in args]
                raise VerificationError(
                    function_name=func_name,
                    condition=f"{param_name} matches one of {union_types}",
                    message=f"Value doesn't match any of the union types: {union_types}",
                    values={param_name: value},
                    suggestion=f"Ensure {param_name} matches one of the allowed types"
                )
            else:
                # Regular generic type - check the origin
                try:
                    if not isinstance(value, origin):
                        raise VerificationError(
                            function_name=func_name,
                            condition=f"{param_name} is {origin.__name__}",
                            message=f"Expected {origin.__name__}, got {type(value).__name__}",
                            values={param_name: value},
                            suggestion=f"Provide a {origin.__name__} for {param_name}"
                        )
                except TypeError:
                    # Some origins can't be used with isinstance, skip
                    pass
        else:
            # Fallback for other generic-like types
            try:
                if not isinstance(value, annotation):
                    raise VerificationError(
                        function_name=func_name,
                        condition=f"{param_name} is {annotation}",
                        message=f"Expected {annotation}, got {type(value).__name__}",
                        values={param_name: value},
                        suggestion=f"Provide correct type for {param_name}"
                    )
            except TypeError:
                # If isinstance fails, skip the check
                pass

    # Handle regular types
    elif isinstance(annotation, type):
        if not isinstance(value, annotation):
            raise VerificationError(
                function_name=func_name,
                condition=f"{param_name} is {annotation.__name__}",
                message=f"Expected {annotation.__name__}, got {type(value).__name__}",
                values={param_name: value},
                suggestion=f"Provide a {annotation.__name__} for {param_name}"
            )

    # Handle None type explicitly
    elif annotation is type(None):
        if value is not None:
            raise VerificationError(
                function_name=func_name,
                condition=f"{param_name} is None",
                message=f"Expected None, got {type(value).__name__}",
                values={param_name: value},
                suggestion=f"Provide None for {param_name}"
            )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PROTOCOL VERIFICATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class StatefulMeta(type):
    """Metaclass for stateful classes"""

    def __new__(mcs, name, bases, namespace, **kwargs):
        initial_state = kwargs.get('initial', 'initial')

        # Create protocol
        protocol = _Protocol(f"{name}Protocol", initial_state)

        # Collect state transitions from methods
        states_seen = {initial_state}
        transitions = []

        for attr_name, attr_value in namespace.items():
            if hasattr(attr_value, '_state_transition'):
                from_states, to_state = attr_value._state_transition
                transitions.append((from_states, to_state))
                states_seen.update(from_states)
                states_seen.add(to_state)

        # Add states to protocol
        for state_name in states_seen:
            # Collect all possible transitions for this state
            allowed_transitions = []
            for from_states, to_state in transitions:
                if state_name in from_states:
                    allowed_transitions.append(to_state)

            protocol.add_state(_ProtocolState(state_name, allowed_transitions))

        # Store protocol in namespace
        namespace['_protocol'] = protocol

        # Wrap state transition methods
        for attr_name, attr_value in namespace.items():
            if hasattr(attr_value, '_state_transition'):
                _, to_state = attr_value._state_transition
                namespace[attr_name] = _protocol_method(protocol, to_state)(attr_value)

        return super().__new__(mcs, name, bases, namespace)


def stateful(*, initial: str = 'initial'):
    """
    Class decorator for stateful protocol verification

    Usage:
        @stateful(initial="closed")
        class File:
            @state("closed", "open")
            def open(self): pass

            @state("open", "reading")
            def read(self): pass
    """

    def decorator(cls):
        return StatefulMeta(cls.__name__, (cls,), dict(cls.__dict__), initial=initial)

    return decorator


def state(from_state, to_state):
    """
    Method decorator for state transitions using valid Python syntax

    Usage:
        @state("closed", "open")
        def open(self): pass

        @state(["reading", "open"], "closed")
        def close(self): pass
    """

    def decorator(func):
        # Handle both single states and lists of states
        if isinstance(from_state, str):
            from_states = [from_state]
        elif isinstance(from_state, (list, tuple)):
            from_states = list(from_state)
        else:
            raise ValueError(f"from_state must be string or list, got {type(from_state)}")

        if not isinstance(to_state, str):
            raise ValueError(f"to_state must be string, got {type(to_state)}")

        func._state_transition = (from_states, to_state)
        return func

    return decorator


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PERFORMANCE TRACKING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

_performance_data = {}


def _record_performance(function_name: str, execution_time: float):
    """Record performance data for a function"""
    if function_name not in _performance_data:
        _performance_data[function_name] = []

    _performance_data[function_name].append(execution_time)


def performance_report() -> str:
    """Generate simple performance report"""
    if not _performance_data:
        return "No performance data available. Use @verify(track_performance=True)"

    lines = ["Performance Report", "~" * 62]

    for func_name, times in _performance_data.items():
        avg_time = sum(times) / len(times)
        total_time = sum(times)
        call_count = len(times)

        lines.append(f"{func_name:30} {avg_time * 1000:6.1f}ms avg, "
                     f"called {call_count:3d}x, {total_time * 1000:6.1f}ms total")

    return "\n".join(lines)


def clear_performance_data():
    """Clear all performance data"""
    global _performance_data
    _performance_data = {}


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# INTEGRATION HELPERS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def enable_for_dataclass(cls):
    """
    Enable verification for dataclass fields

    Usage:
        @enable_for_dataclass
        @dataclass
        class User:
            name: NonEmpty[str]
            age: Range[int, 0, 150]
    """

    original_init = cls.__init__

    @functools.wraps(original_init)
    def checked_init(self, *args, **kwargs):
        # Let dataclass do its thing first
        original_init(self, *args, **kwargs)

        # Then check field constraints
        if _config.mode != "off":
            _check_dataclass_fields(self)

    cls.__init__ = checked_init
    return cls


def _check_dataclass_fields(instance):
    """Check dataclass field constraints"""
    import dataclasses

    if not dataclasses.is_dataclass(instance):
        return

    for field in dataclasses.fields(instance):
        if field.type == dataclasses.MISSING:
            continue

        value = getattr(instance, field.name)
        _check_type_constraint(field.type, value, field.name, instance.__class__.__name__)


def expect_approximately(actual: float, expected: float, tolerance: float = 1e-10) -> bool:
    """Helper for floating point comparisons"""
    return abs(actual - expected) <= tolerance


# Make it available with different syntax
def approx_equal(a: float, b: float, tolerance: float = 1e-10) -> bool:
    """Alias for expect_approximately"""
    return expect_approximately(a, b, tolerance)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SIMPLIFIED CONTEXT MANAGERS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@contextmanager
def verification_context(name: str):
    """Simple context manager for verification"""
    with _proof_context(name):
        yield


@contextmanager
def production_mode():
    """Temporarily switch to production mode"""
    old_mode = _config.mode
    _config.mode = "prod"
    try:
        yield
    finally:
        _config.mode = old_mode


@contextmanager
def no_verification():
    """Temporarily disable verification"""
    old_mode = _config.mode
    _config.mode = "off"
    try:
        yield
    finally:
        _config.mode = old_mode


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MODULE CONVENIENCE FUNCTIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def set_mode(mode: str):
    """Set verification mode"""
    _config.mode = mode


def get_mode() -> str:
    """Get current verification mode"""
    return _config.mode


def report():
    """Generate comprehensive status report"""
    lines = [
        "Simple Axiomatik Status",
        "~" * 80,
        f"Mode: {_config.mode}",
        f"Performance tracking: {bool(_performance_data)}",
        "",
        "Underlying Axiomatik Status:",
        f"  Verification level: {_ax._config.level.value}",
        f"  Performance mode: {_ax._config.performance_mode}",
        f"  Debug mode: {_ax._config.debug_mode}",
    ]

    # Add performance data if available
    if _performance_data:
        lines.extend(["", performance_report()])

    # Add underlying proof data
    summary = _ax._proof.get_summary()
    lines.extend([
        "",
        f"Total verifications: {summary['total_steps']}",
        f"Contexts used: {len(summary['contexts'])}",
        f"Threads involved: {summary['thread_count']}"
    ])

    return "\n".join(lines)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DEMO FUNCTION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def demo():
    """Demonstrate Simple Axiomatik features with correct syntax"""
    print("Simple Axiomatik Demo: Verification that feels like Python")
    print("=" * 60)

    # Set development mode
    set_mode("dev")
    print(f"Mode: {get_mode()}")

    # Demo 1: Simple verification
    print("\n1. Simple Verification with @verify:")

    @verify
    def safe_divide(a: float, b: float) -> float:
        """Divide two numbers safely"""
        require(b != 0, "Cannot divide by zero")
        result = a / b
        ensure(approx_equal(result * b, a), "Division check failed")
        return result

    try:
        result = safe_divide(10.0, 2.0)
        print(f"   safe_divide(10.0, 2.0) = {result}")
    except VerificationError as e:
        print(f"   Error: {e}")

    try:
        result = safe_divide(10.0, 0.0)
        print(f"   This shouldn't print: {result}")
    except VerificationError as e:
        print(f"   Caught error: Cannot divide by zero")

    # Demo 2: Automatic verification from types
    print("\n2. Automatic Verification with @checked:")

    @checked
    def process_items(items: NonEmpty[list], multiplier: PositiveInt) -> PositiveInt:
        """Process non-empty list with positive multiplier"""
        return len(items) * multiplier

    try:
        result = process_items(["a", "b", "c"], 3)
        print(f"   process_items(['a', 'b', 'c'], 3) = {result}")
    except VerificationError as e:
        print(f"   Error: {e}")

    try:
        result = process_items([], 3)  # Empty list should fail
        print(f"   This shouldn't print: {result}")
    except VerificationError as e:
        print(f"   Caught error: Empty list rejected")

    # Demo 3: Type aliases
    print("\n3. Type Aliases:")

    @checked
    def calculate_grade(score: PositiveInt, total: PositiveInt) -> Percentage:
        """Calculate percentage grade"""
        percentage = (score * 100) // total
        return min(100, percentage)

    try:
        grade = calculate_grade(85, 100)
        print(f"   calculate_grade(85, 100) = {grade}%")
    except VerificationError as e:
        print(f"   Error: {e}")

    # Demo 4: Stateful protocols
    print("\n4. Stateful Protocol Verification:")

    @stateful(initial="closed")
    class SimpleFile:
        def __init__(self, name: str):
            self.name = name
            self.content = ""

        @state("closed", "open")
        def open(self):
            print(f"     Opening {self.name}")

        @state("open", "reading")
        def read(self):
            print(f"     Reading from {self.name}")
            return self.content

        @state(["reading", "open"], "closed")
        def close(self):
            print(f"     Closing {self.name}")

    try:
        f = SimpleFile("test.txt")
        f.open()
        content = f.read()
        f.close()
        print("   File protocol worked correctly")
    except VerificationError as e:
        print(f"   Protocol error: {e}")

    # Demo 5: Performance tracking
    print("\n5. Performance Tracking:")

    @verify(track_performance=True)
    def slow_function(n: int) -> int:
        """Simulate slow function"""
        time.sleep(0.001)  # 1ms delay
        return n * 2

    # Call it a few times
    for i in range(5):
        slow_function(i)

    print("   Performance data collected")

    # Demo 6: Integration with dataclasses
    print("\n6. Dataclass Integration:")

    from dataclasses import dataclass

    @enable_for_dataclass
    @dataclass
    class User:
        name: NonEmptyStr
        age: Range[int, 0, 150]
        email: str

    try:
        user = User("Alice", 30, "alice@example.com")
        print(f"   Created user: {user.name}, age {user.age}")
    except VerificationError as e:
        print(f"   Validation error: {e}")

    # Demo 7: Context managers
    print("\n7. Context Managers:")

    with verification_context("demo_context"):
        require(True, "This should pass")
        print("   Verification context works")

    with no_verification():
        # This would normally fail, but verification is disabled
        # require(False, "This would fail normally")
        print("   No verification mode works")

    # Final report
    print("\n8. Status Report:")
    print(report())

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    demo()