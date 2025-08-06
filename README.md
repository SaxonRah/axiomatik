# Axiomatik: Performant Runtime Verification for Python

**Axiomatik is a comprehensive runtime verification system that brings formal verification concepts to practical Python programming. It provides proof-based assertions, contracts, invariants, and advanced verification features with performance optimizations for production use.**

**Axiomatik brings the rigor of formal verification to everyday Python programming through three complementary layers: Simple for ease of use, Core for comprehensive verification, and Future for enterprise-grade adaptive capabilities.**
## Architecture Overview

Axiomatik is organized in three complementary layers:

1. **Simple Axiomatik** (`simple_axiomatik.py`) - Pythonic wrapper that feels natural and integrates seamlessly
2. **Core Axiomatik** (`axiomatik.py`) - Complete verification engine with protocols, information flow, and temporal properties  
3. **Future Axiomatik** (`future_axiomatik.py`) - Enterprise-grade features including adaptive monitoring, performance introspection, and recovery framework

## Quick Start with Simple Axiomatik

The easiest way to get started is with Simple Axiomatik:

```python
import axiomatik.simple_axiomatik as ax

# Set verification mode
ax.set_mode("dev")  # Full verification during development

# Basic verification with helpful error messages
@ax.verify
def safe_divide(a: float, b: float) -> float:
    ax.require(b != 0, "Cannot divide by zero")
    result = a / b
    ax.ensure(ax.approx_equal(result * b, a), "Division check failed")
    return result

# Automatic verification from type hints
@ax.checked
def process_items(items: ax.NonEmpty[list], multiplier: ax.PositiveInt) -> ax.PositiveInt:
    """Types do the validation automatically"""
    return len(items) * multiplier

# Protocol verification for stateful objects
@ax.stateful(initial="closed")
class File:
    @ax.state("closed", "open")
    def open(self): pass
    
    @ax.state("open", "reading") 
    def read(self): pass
    
    @ax.state(["reading", "open"], "closed")
    def close(self): pass
```

## Core Features

### Runtime Verification
- **Proof-based assertions** with `require()` instead of `assert`
- **Function contracts** with preconditions and postconditions
- **Loop invariants** and termination proofs
- **Protocol verification** for API usage patterns
- **Data structure invariants** for custom classes

### Advanced Verification
- **Refinement types** for precise constraints (`PositiveInt`, `NonEmptyList`, etc.)
- **Information flow tracking** for security-sensitive data
- **Temporal properties** for event sequences and timing
- **Ghost state** for proof-only auxiliary data
- **Plugin system** for domain-specific verification

### Performance & Production
- **Configurable verification levels** (off/contracts/invariants/full/debug)
- **Proof caching** for expensive computations
- **Thread-safe** operation with concurrent proof traces
- **Performance mode** for production deployments
- **Automatic instrumentation** with `axiomatikify` tool


## Simple Wrapper
See `SimpleAxiomatik.md` for a simple wrapper around Axiomatik.

Reduced barrier for entry.


## Future Features
See `FutureFeatures.md` for Adaptive Monitoring, Performance Introspection, Recovery Framework.

While the demo runs, nothing is assumed to be production ready. Here be dragons. Beware.


## Installation

```bash
pip install axiomatik
```
### NOTE: axiomatik is now on pip! **MIGHT BE BROKEN - USE REPO FOR NOW.**
**Maybe you'd like to try the pypi release and show issues.**
This is my first pypi project, I'm used to closed source managment. There will be teething issues while I figure pypi out.

Or for development:
```bash
git clone https://github.com/your-org/axiomatik
cd axiomatik
pip install -e .
```

## Documentation

### Basic Usage

#### 1. Proof-Based Assertions

Replace `assert` with `require()` for better error messages and proof traces:

```python
# Built-in type aliases from simple_axiomatik.py
@ax.checked
def calculate_interest(
    principal: ax.Positive[float],        # > 0.0
    rate: ax.Range[float, 0.0, 0.5],     # 0-50% annual rate  
    years: ax.Range[int, 1, 50],         # 1-50 years
    compounding: ax.PositiveInt = 12      # Monthly by default
) -> ax.Positive[float]:
    return principal * (1 + rate/compounding) ** (compounding * years)

# Dataclass integration with automatic validation
@ax.enable_for_dataclass
@dataclass
class User:
    name: ax.NonEmpty[str]
    age: ax.Range[int, 0, 150] 
    email: ax.NonEmpty[str]
    
    def __post_init__(self):
        ax.require("@" in self.email, "Email must contain @ symbol")
```

### Simple Axiomatik Error Handling

Simple Axiomatik provides clear, helpful error messages:

```python
try:
    result = process_items([], 5)
except ax.VerificationError as e:
    print(e)
    # Output:
    # process_items() verification failed
    # Condition: items satisfies non-empty list
    # Message: Type constraint violation for items
    # Values: items=[]
    # Suggestion: Ensure items meets the required constraints
```

## Core Axiomatik Features

The complete system (`axiomatik.py`) provides enterprise-grade verification capabilities:

### 1. Proof-Based Runtime Verification

```python
from axiomatik import require, proof_context, ProofFailure

def factorial(n: int) -> int:
    require("n is non-negative", n >= 0)
    require("n is reasonable size", n <= 100)
    
    if n <= 1:
        return 1
    
    result = 1
    with proof_context("factorial_loop"):
        for i in range(2, n + 1):
            old_result = result
            result *= i
            require("result increased", result > old_result)
            require("result is positive", result > 0)
    
    return result
```

### 2. Function Contracts

```python
from axiomatik import contract, auto_contract

@contract(
    preconditions=[
        ("list is not empty", lambda items: len(items) > 0),
        ("all items are numbers", lambda items: all(isinstance(x, (int, float)) for x in items))
    ],
    postconditions=[
        ("result is in list", lambda items, result: result in items),
        ("result is maximum", lambda items, result: result >= max(items))
    ]
)
def find_maximum(items):
    return max(items)

# Automatic contract generation from type hints
@auto_contract  
def calculate_grade(score: PositiveInt) -> Percentage:
    return min(100, max(0, score))
```

### 3. Refinement Types

```python
from axiomatik import RefinementType, PositiveInt, NonEmptyList, Percentage

# Built-in refinement types
age = PositiveInt(25)
items = NonEmptyList([1, 2, 3, 4])
score = Percentage(85)

# Custom refinement types
EvenInt = RefinementType(int, lambda x: x % 2 == 0, "even integer")
ValidEmail = RefinementType(str, lambda x: "@" in x and "." in x, "valid email")

even_number = EvenInt(42)  # OK
email = ValidEmail("user@example.com")  # OK
```

### 4. Protocol Verification

```python
from axiomatik import Protocol, ProtocolState, protocol_method

# Use predefined protocols from axiomatik.py
class FileManager:
    @protocol_method(filemanager_protocol, "open")
    def open(self):
        self.is_open = True
    
    @protocol_method(filemanager_protocol, "read")
    def read(self):
        return self.content
    
    @protocol_method(filemanager_protocol, "close")
    def close(self):
        self.is_open = False

# Or define custom protocols
database_protocol = Protocol("Database", "disconnected")
database_protocol.add_state(ProtocolState("disconnected", ["connected"]))
database_protocol.add_state(ProtocolState("connected", ["transaction", "disconnected"]))
database_protocol.add_state(ProtocolState("transaction", ["connected"]))
```

### 5. Information Flow Tracking

```python
from axiomatik import TaintedValue, SecurityLabel, InformationFlowTracker

# Track sensitive data through computations
secret_data = TaintedValue("classified_info", SecurityLabel.SECRET, ["database"])
public_data = TaintedValue("public_info", SecurityLabel.PUBLIC, ["web_form"])

# Information flow is automatically tracked
combined = secret_data.combine_with(public_data)
print(f"Security level: {combined.label}")  # SECRET (highest)

# Controlled declassification with justification
secret_data.declassify(SecurityLabel.CONFIDENTIAL, "Sanitized for internal use")

# Flow policy enforcement
flow_tracker = InformationFlowTracker()
flow_tracker.add_policy(SecurityLabel.SECRET, SecurityLabel.PUBLIC, False)
try:
    flow_tracker.track_flow(secret_data, SecurityLabel.PUBLIC)  # Raises ProofFailure
except ProofFailure:
    print("Illegal information flow prevented")

# Automatic sensitive data tracking
password = track_sensitive_data("user_password", "secret123", SecurityLabel.SECRET)
```

### 6. Temporal Properties

```python
from axiomatik import (
    TemporalVerifier, EventuallyProperty, AlwaysProperty, 
    record_temporal_event, add_temporal_property, verify_temporal_properties
)

# Define temporal properties
eventually_complete = EventuallyProperty(
    "task_completes",
    lambda history: any(e['event'] == 'task_done' for e in history),
    timeout=5.0
)

always_valid = AlwaysProperty(
    "data_valid",
    lambda event: event.get('data', {}).get('valid', True)
)

# Add to global verifier
add_temporal_property(eventually_complete)
add_temporal_property(always_valid)

# Record events during execution
record_temporal_event("task_start", {"valid": True})
record_temporal_event("processing", {"valid": True, "progress": 50})
record_temporal_event("task_done", {"valid": True, "result": "success"})

# Verification happens automatically
verify_temporal_properties()  # Passes if all properties hold
```

### 7. Plugin System

Axiomatik includes specialized plugins for different domains:

```python
# Financial calculations with precision tracking
from axiomatik import _plugin_registry

Money = _plugin_registry.get_type("Money")
price = Money("19.99", "USD")
tax = Money("1.60", "USD")
total = price + tax

# Cryptographic verification
crypto_verifier = _plugin_registry.get_verifier("constant_time")
secure_compare = lambda a, b: a == b
is_constant_time = crypto_verifier(secure_compare, ["secret1", "secret2"])

# Security verification
security_verifier = _plugin_registry.get_verifier("input_sanitized")
sanitizer = lambda x: x.replace("<", "&lt;").replace(">", "&gt;")
is_safe = security_verifier("<script>alert('xss')</script>", sanitizer)

# Concurrency verification
concurrency_verifier = _plugin_registry.get_verifier("no_deadlock")
import threading
lock1, lock2 = threading.Lock(), threading.Lock()
is_deadlock_safe = concurrency_verifier(lock1, lock2)
```

## Future Axiomatik: Advanced Features

The `future_axiomatik.py` module provides cutting-edge enterprise capabilities:

### 1. Adaptive Monitoring

Dynamically adjusts verification behavior based on runtime conditions:

```python
from axiomatik.future_axiomatik import (
    adaptive_require, adaptive_verification_context, 
    auto_tune_verification_level
)

# Enable intelligent auto-tuning for performance targets
auto_tune_verification_level(target_overhead_percent=5.0)

# Context-aware property loading with priority-based verification
with adaptive_verification_context("database"):
    adaptive_require(
        "connection is valid",
        db.is_connected(),
        property_name="db_connection_check",
        priority=5  # Critical - always verify
    )

with adaptive_verification_context("network"):
    adaptive_require(
        "network is available",
        network.ping("api.example.com"),
        property_name="network_check",
        priority=3  # Medium - may be sampled under load
    )
```

**Key Adaptive Features:**
- **Smart Sampling**: Reduces verification frequency for expensive properties under high load
- **Priority-Based Verification**: Critical properties (priority 4-5) always verified
- **Auto-Tuning**: Automatically adjusts verification levels to meet performance targets
- **Context-Aware Loading**: Loads appropriate properties based on execution context

### 2. Performance Introspection

Comprehensive performance analysis and optimization:

```python
from axiomatik.future_axiomatik import (
    get_performance_hotspots, generate_performance_report, 
    visualize_performance, PerformanceAnalyzer
)

# Run your application with verification enabled...

# Analyze performance hotspots
hotspots = get_performance_hotspots(10)
for hotspot in hotspots:
    print(f"{hotspot.property_name}: {hotspot.average_time*1000:.2f}ms avg "
          f"({hotspot.percentage_of_total:.1f}% of total)")

# Generate comprehensive performance report
report = generate_performance_report()
print(report)

# Create visualization (requires matplotlib)
visualize_performance("verification_hotspots.png")

# Access detailed analytics
analyzer = PerformanceAnalyzer()
analyzer.auto_tune_verification_level(target_overhead_percent=3.0)
```

**Sample Performance Analysis Output:**
```
Axiomatik Performance Analysis
==================================================

Total properties verified: 13
Total verification calls: 260  
Total verification time: 0.001s
Average per verification: 0.002ms

Top Performance Hotspots:
  1. postcondition: result is int          0.2ms (15.1%) [40 calls]
  2. connection is valid                   0.1ms ( 8.9%) [50 calls] 
  3. network is available                  0.1ms ( 7.9%) [50 calls]
  4. precondition: data not empty          0.1ms ( 6.2%) [30 calls]
  5. value satisfies: positive integer     0.0ms ( 4.8%) [25 calls]

Performance by Context:
  contract_verification    : 0.4ms (40.2%)
  adaptive_monitoring     : 0.2ms (25.1%) 
  protocol_verification   : 0.1ms (15.3%)
  temporal_verification   : 0.1ms (12.8%)
```

### 3. Recovery Framework

Multiple recovery strategies for handling verification failures gracefully:

```python
from axiomatik.future_axiomatik import (
    contract_with_recovery, RecoveryStrategy, RecoveryPolicy, RecoveryManager
)

# Graceful degradation - use simpler algorithm on verification failure
@contract_with_recovery(
    preconditions=[("data has sufficient size", lambda data: len(data) >= 10)],
    recovery_strategy=RecoveryStrategy(
        RecoveryPolicy.GRACEFUL_DEGRADATION,
        fallback_handler=lambda data: {
            "result": "simplified_analysis", 
            "mean": sum(data)/len(data)
        }
    )
)
def complex_statistical_analysis(data: List[float]) -> Dict[str, Any]:
    # Complex analysis requiring at least 10 data points
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    
    return {
        "mean": mean,
        "variance": variance, 
        "std_dev": variance ** 0.5,
        "analysis": "complete"
    }

# Retry with exponential backoff for transient failures
@contract_with_recovery(
    preconditions=[("network is available", lambda: check_network_connection())],
    recovery_strategy=RecoveryStrategy(
        RecoveryPolicy.RETRY_WITH_BACKOFF,
        max_retries=3,
        backoff_factor=2.0
    )
)
def fetch_external_data(url: str) -> Dict:
    return requests.get(url).json()

# Circuit breaker pattern for repeated failures
@contract_with_recovery(
    preconditions=[("service is healthy", lambda: health_check())],
    recovery_strategy=RecoveryStrategy(
        RecoveryPolicy.CIRCUIT_BREAKER,
        circuit_breaker_threshold=5,
        fallback_handler=lambda: {
            "status": "degraded", 
            "data": get_cached_fallback_data()
        }
    )
)
def call_external_service():
    return external_api.get_data()

# State rollback for critical operations
@contract_with_recovery(
    preconditions=[("state is consistent", lambda: validate_system_state())],
    recovery_strategy=RecoveryStrategy(RecoveryPolicy.ROLLBACK_STATE)
)
def critical_state_mutation():
    modify_critical_state()
    return "success"
```

**Recovery Policies:**

| Policy | Description | Use Case |
|--------|-------------|----------|
| `FAIL_FAST` | Raise exception immediately (default) | Development, testing |
| `GRACEFUL_DEGRADATION` | Use simpler fallback algorithm | Statistical analysis, ML inference |
| `RETRY_WITH_BACKOFF` | Retry with exponential backoff | Network operations, external APIs |
| `CIRCUIT_BREAKER` | Disable after repeated failures | Service dependencies |
| `ROLLBACK_STATE` | Restore previous known-good state | Database transactions, critical updates |

### 4. Combined Enterprise Features

All three future features work together seamlessly:

```python
from axiomatik.future_axiomatik import (
    contract_with_recovery, adaptive_verification_context, 
    RecoveryStrategy, RecoveryPolicy
)

@contract_with_recovery(
    preconditions=[("input is valid", lambda data: len(data) > 0)],
    recovery_strategy=RecoveryStrategy(
        RecoveryPolicy.GRACEFUL_DEGRADATION,
        fallback_handler=lambda data: {
            "result": "basic_processing", 
            "count": len(data)
        }
    )
)
def enterprise_data_processor(data: List[int]) -> Dict[str, Any]:
    """Production function with adaptive monitoring, performance tracking, and recovery"""
    
    # Adaptive monitoring adjusts verification based on load
    with adaptive_verification_context("data_processing"):
        adaptive_require(
            "data is properly formatted",
            all(isinstance(x, int) for x in data),
            property_name="data_format_check", 
            priority=4
        )
        
        # Performance is automatically tracked
        result = {
            "sum": sum(data),
            "average": sum(data) / len(data),
            "max": max(data), 
            "min": min(data)
        }
        
        # Strict verification that might trigger recovery
        require("result is comprehensive", len(result) >= 4)
        require("average is reasonable", 0 <= result["average"] <= 1000000)
        
        return result
```

## Automatic Instrumentation

Use `axiomatikify.py` to automatically add verification to existing code:

```bash
# Instrument entire project with all features
python axiomatikify.py src/ instrumented/ --all

# Selective instrumentation
python axiomatikify.py src/ instrumented/ --contracts --loops --asserts --temporal --protocols
```

**Before:**
```python
def factorial(n):
    assert n >= 0
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

class FileManager:
    def open(self): pass
    def read(self): pass  
    def close(self): pass
```

**After automatic instrumentation:**
```python
import axiomatik
from axiomatik import require, auto_contract, protocol_method, filemanager_protocol

@auto_contract
def factorial(n):
    require("n >= 0", n >= 0)
    result = 1
    for i in range(1, n + 1):
        with axiomatik.proof_context('for_loop_invariant'):
            result *= i
    return result

class FileManager:
    @protocol_method(filemanager_protocol, "open")
    def open(self): pass
    
    @protocol_method(filemanager_protocol, "read")
    def read(self): pass
    
    @protocol_method(filemanager_protocol, "close")  
    def close(self): pass
```

The instrumentation tool provides:
- **Contract injection** from type hints via `auto_contract`
- **Loop instrumentation** with proof contexts for invariants
- **Assert conversion** to `require()` calls with better error messages
- **Protocol detection** for stateful classes like FileManager and StateMachine
- **Temporal event recording** for function entry/exit
- **Information flow tracking** for sensitive data assignments

## Configuration and Deployment

### Simple Axiomatik Configuration

```python
import axiomatik.simple_axiomatik as ax

# Environment-based mode setting
ax.set_mode("dev")   # Full verification during development
ax.set_mode("prod")  # Essential checks only for production
ax.set_mode("test")  # Comprehensive verification with debug info
ax.set_mode("off")   # No verification overhead

# Context managers for temporary changes
with ax.production_mode():
    result = performance_critical_function()

with ax.no_verification():
    result = expensive_function()
```

### Core Axiomatik Configuration

```python
import os
from axiomatik import Config, VerificationLevel

# Configure based on environment
if os.getenv('ENVIRONMENT') == 'production':
    Config().level = VerificationLevel.CONTRACTS  # Minimal overhead
elif os.getenv('ENVIRONMENT') == 'staging': 
    Config().level = VerificationLevel.FULL
else:  # Development
    Config().level = VerificationLevel.DEBUG
```

### Future Axiomatik Configuration

```python
from axiomatik.future_axiomatik import auto_tune_verification_level, PerformanceAnalyzer

# Production configuration with adaptive monitoring
if os.getenv('ENVIRONMENT') == 'production':
    auto_tune_verification_level(target_overhead_percent=2.0)
elif os.getenv('ENVIRONMENT') == 'staging':
    auto_tune_verification_level(target_overhead_percent=5.0)
else:  # Development
    auto_tune_verification_level(target_overhead_percent=15.0)
```

### Environment Variables

```bash
# Core Axiomatik configuration
export AXIOMATIK_LEVEL=full        # off|contracts|invariants|full|debug
export AXIOMATIK_CACHE=1           # Enable proof caching
export AXIOMATIK_MAX_STEPS=10000   # Maximum proof steps
export AXIOMATIK_PERF=1            # Performance mode
```

### Verification Levels

| Level | Overhead | Features | Use Case |
|-------|----------|----------|----------|
| OFF | 0% | None | Production (max performance) |
| CONTRACTS | 5-10% | Function contracts only | Integration testing |
| INVARIANTS | 10-15% | Contracts + data invariants | Staging environment |
| FULL | 10-25% | All verification features | Development |
| DEBUG | 25-50% | Full + detailed logging | Debugging sessions |

## Real-World Examples

### Web API Validation (Simple Axiomatik)

From `simple_usage.py`:

```python
@ax.enable_for_dataclass
@dataclass
class CreateUserRequest:
    username: ax.NonEmpty[str]
    email: ax.NonEmpty[str]
    age: ax.Range[int, 13, 120]
    password: ax.NonEmpty[str]
    
    def __post_init__(self):
        ax.require("@" in self.email, "Email must contain @ symbol")
        ax.require(len(self.password) >= 8, "Password must be at least 8 characters")

@ax.checked
def create_user(request: CreateUserRequest) -> Dict[str, any]:
    ax.require(not _username_exists(request.username), "Username already taken")
    ax.require(_is_valid_email_domain(request.email), "Email domain not allowed")
    
    user_id = abs(hash(request.username)) % 10000
    return {
        "user_id": user_id,
        "username": request.username,
        "status": "created"
    }
```

### Financial Calculations with Recovery (Future Axiomatik)

```python
from axiomatik.future_axiomatik import contract_with_recovery, RecoveryStrategy, RecoveryPolicy

@contract_with_recovery(
    preconditions=[
        ("principal is positive", lambda principal, rate, years: principal > 0),
        ("rate is reasonable", lambda principal, rate, years: 0 <= rate <= 0.5)
    ],
    recovery_strategy=RecoveryStrategy(
        RecoveryPolicy.GRACEFUL_DEGRADATION,
        fallback_handler=lambda principal, rate, years: {
            "result": "simple_interest",
            "amount": principal * (1 + rate * years)
        }
    )
)
def calculate_compound_interest(principal: float, rate: float, years: int) -> Dict[str, float]:
    # Complex compound interest calculation
    if rate > 0.3:  # Trigger recovery for unreasonable rates
        require("rate is reasonable for compound calculation", rate <= 0.3)
    
    amount = principal * ((1 + rate) ** years)
    return {"principal": principal, "amount": amount, "interest": amount - principal}
```

### Data Processing Pipeline (Core Axiomatik)

From `simple_usage.py`, enhanced with core features:

```python
from axiomatik import protocol_method, statemachine_protocol, record_temporal_event

class DataProcessor:
    @protocol_method(statemachine_protocol, "running")
    def start_processing(self, data: List[str]):
        record_temporal_event("processing_started", {"data_count": len(data)})
        require("data is not empty", len(data) > 0)
        self.data = data
    
    @protocol_method(statemachine_protocol, "process") 
    def process_batch(self) -> Dict[str, int]:
        record_temporal_event("batch_processing", {"remaining": len(self.data)})
        require("data available", len(self.data) > 0)
        
        # Process batch
        batch_size = min(50, len(self.data))
        processed = self.data[:batch_size]
        self.data = self.data[batch_size:]
        
        record_temporal_event("batch_completed", {"processed": len(processed)})
        return {"processed": len(processed), "remaining": len(self.data)}
    
    @protocol_method(statemachine_protocol, "stopped")
    def finish_processing(self):
        record_temporal_event("processing_completed")
        require("all data processed", len(self.data) == 0)
```

### Game State Management (All Three Systems)

```python
# Using Simple Axiomatik for basic structure
@ax.stateful(initial="waiting")
class GameSession:
    def __init__(self, game_id: str):
        self.game_id = game_id
        self.players = []
    
    @ax.state("waiting", "lobby")
    def open_lobby(self):
        record_temporal_event("lobby_opened", {"game_id": self.game_id})
    
    # Using Core Axiomatik for verification
    @ax.verify
    def add_player(self, player_name: ax.NonEmpty[str]) -> bool:
        require("lobby not full", len(self.players) < 4)
        require("name not taken", player_name not in self.players)
        
        self.players.append(player_name)
        record_temporal_event("player_joined", {"player": player_name})
        return True
    
    # Using Future Axiomatik for adaptive monitoring
    @contract_with_recovery(
        preconditions=[("game can start", lambda self: len(self.players) >= 2)],
        recovery_strategy=RecoveryStrategy(
            RecoveryPolicy.GRACEFUL_DEGRADATION,
            fallback_handler=lambda self: self._start_single_player_mode()
        )
    )
    @ax.state("lobby", "playing")
    def start_game(self):
        with adaptive_verification_context("game_logic"):
            adaptive_require(
                "minimum players",
                len(self.players) >= 2,
                property_name="multiplayer_check",
                priority=4
            )
```

### Scientific Data Analysis (Core Axiomatik)

```python
from axiomatik import require, auto_contract, PositiveInt, NonEmptyList

@auto_contract
def detect_outliers(data: NonEmptyList[float], 
                   method: str = "iqr",
                   threshold: PositiveFloat = 1.5) -> Dict[str, Any]:
    require("valid method", method in ["iqr", "zscore"])
    
    values = list(data)
    outliers = []
    
    if method == "iqr":
        sorted_values = sorted(values)
        n = len(sorted_values)
        q1 = sorted_values[n // 4]
        q3 = sorted_values[3 * n // 4]
        iqr = q3 - q1
        
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        for i, value in enumerate(values):
            if value < lower_bound or value > upper_bound:
                outliers.append({"index": i, "value": value})
    
    result = {
        "method": method,
        "outliers_found": len(outliers),
        "outliers": outliers
    }
    
    require("outlier count is valid", result["outliers_found"] >= 0)
    return result
```

## Testing Integration

All three systems work seamlessly with standard testing frameworks:

```python
import pytest
from axiomatik import verification_mode, ProofFailure
import axiomatik.simple_axiomatik as ax
from axiomatik.future_axiomatik import contract_with_recovery, RecoveryPolicy

def test_simple_axiomatik():
    with pytest.raises(ax.VerificationError, match="Cannot divide by zero"):
        safe_divide(10.0, 0.0)

def test_core_axiomatik():
    with verification_mode():  # Enable debug-level verification
        result = verified_function(test_input)
        assert result == expected

def test_future_axiomatik_recovery():
    # Test that recovery strategy works
    @contract_with_recovery(
        preconditions=[("input valid", lambda x: x > 0)],
        recovery_strategy=RecoveryStrategy(
            RecoveryPolicy.GRACEFUL_DEGRADATION,
            fallback_handler=lambda x: 0
        )
    )
    def test_function(x):
        return x * 2
    
    result = test_function(-5)  # Triggers recovery
    assert result == 0  # Fallback value

def test_performance_tracking():
    ax.clear_performance_data()
    
    @ax.verify(track_performance=True)
    def tracked_function():
        return 42
    
    tracked_function()
    report = ax.performance_report()
    assert "tracked_function" in report
```

## Demonstration Files

The repository includes comprehensive demonstration files:

- **`simple_quick.py`** - Complete tutorial covering all Simple Axiomatik features with step-by-step examples
- **`simple_usage.py`** - Production-ready patterns for web APIs, finance, gaming, and data analysis using Simple Axiomatik
- **`test.py`** - Comprehensive test scenarios showing automatic instrumentation capabilities
- **`axiomatik.py`** - Full demonstration of core verification features including protocols, information flow, and temporal properties
- **`future_axiomatik.py`** - Advanced enterprise features demo with adaptive monitoring, performance analysis, and recovery

## Installation

```bash
# Install from PyPI (Note: may have issues as I figure out pypi, use repo for now)
pip install axiomatik

# Development installation
git clone https://github.com/your-org/axiomatik
cd axiomatik
pip install -e .
```

## Performance Characteristics

Each layer is optimized for different use cases:

**Simple Axiomatik Performance:**
- Dev mode: 10-25% overhead with comprehensive checking
- Prod mode: 2-5% overhead with essential checks only
- Off mode: 0% overhead, complete bypass

**Core Axiomatik Performance:**
- Proof caching reduces repeated verification costs
- Thread-safe concurrent execution
- Configurable verification levels

**Future Axiomatik Performance:**
- Adaptive monitoring: Self-tuning to performance targets
- Performance introspection: Real-time hotspot identification
- Recovery framework: Maintains availability under verification failures

## Production Deployment Strategy

```python
import os

# Layer 1: Simple Axiomatik - Always available
import axiomatik.simple_axiomatik as ax

# Layer 2: Core Axiomatik - For advanced verification
try:
    import axiomatik
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

# Layer 3: Future Axiomatik - For enterprise features
try:
    import axiomatik.future_axiomatik
    FUTURE_AVAILABLE = True
except ImportError:
    FUTURE_AVAILABLE = False

# Configure based on environment and available features
if os.getenv('ENVIRONMENT') == 'production':
    ax.set_mode("prod")
    if FUTURE_AVAILABLE:
        future_axiomatik.auto_tune_verification_level(target_overhead_percent=2.0)
elif os.getenv('ENVIRONMENT') == 'staging':
    ax.set_mode("dev") 
    if FUTURE_AVAILABLE:
        future_axiomatik.auto_tune_verification_level(target_overhead_percent=5.0)
else:  # Development
    ax.set_mode("dev")
    if FUTURE_AVAILABLE:
        future_axiomatik.auto_tune_verification_level(target_overhead_percent=15.0)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for your changes  
4. Run the test suite: `python -m pytest`
5. Test all demonstration files:
   - `python simple_quick.py`
   - `python simple_usage.py`
   - `python test.py`
   - `python axiomatik.py` (run demo_advanced_features)
   - `python future_axiomatik.py` (run demo_future_features)
6. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

---
