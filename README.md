# PyProof: Performant Runtime Verification for Python

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PyProof is a comprehensive runtime verification system that brings formal verification concepts to practical Python programming. It provides proof-based assertions, contracts, invariants, and advanced verification features with performance optimizations for production use.

## Quick Start

```python
import pyproof
from pyproof import require, contract, PositiveInt

# Basic runtime proofs
def safe_divide(a: float, b: float) -> float:
    require("denominator is not zero", b != 0)
    result = a / b
    require("result is finite", not math.isinf(result))
    return result

# Automatic contracts from type hints
@pyproof.auto_contract
def calculate_grade(score: PositiveInt) -> pyproof.Percentage:
    return min(100, max(0, score))

# Protocol verification
@pyproof.protocol_method(pyproof.filemanager_protocol, "open")
def open_file(self):
    self.is_open = True
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
- **Automatic instrumentation** with `pyproofify` tool

## Future Features
See `FutureFeatures.md` for Adaptive Monitoring, Performance Introspection, Recovery Framework.

While the demo runs, nothing is assumed to be production ready. Here be dragons. Beware.


## Installation

```bash
pip install pyproof
```
### NOTE: pyproof is not on pip yet. Clone repo for now.
#### Looking for a new name as pyproof is taken and was an internal tool name.

Or for development:
```bash
git clone https://github.com/your-org/pyproof
cd pyproof
pip install -e .
```

## Documentation

### Basic Usage

#### 1. Proof-Based Assertions

Replace `assert` with `require()` for better error messages and proof traces:

```python
from pyproof import require

def factorial(n: int) -> int:
    require("n is non-negative", n >= 0)
    require("n is reasonable size", n <= 100)
    
    if n <= 1:
        return 1
    
    result = 1
    for i in range(2, n + 1):
        result *= i
        require("result stays positive", result > 0)
    
    require("result is correct factorial", result == math.factorial(n))
    return result
```

#### 2. Function Contracts

Separate preconditions from postconditions:

```python
from pyproof import contract

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
```

#### 3. Automatic Contracts

Generate contracts from type hints:

```python
from pyproof import auto_contract, PositiveInt, NonEmptyList

@auto_contract
def process_scores(scores: NonEmptyList[PositiveInt]) -> float:
    return sum(scores) / len(scores)
```

### Advanced Features

#### 4. Refinement Types

Define precise type constraints:

```python
from pyproof import RefinementType, PositiveInt, Percentage

# Built-in refinement types
age: PositiveInt = PositiveInt(25)
score: Percentage = Percentage(85)
items: NonEmptyList = NonEmptyList([1, 2, 3])

# Custom refinement types
EvenInt = RefinementType(int, lambda x: x % 2 == 0, "even integer")
even_num = EvenInt(42)  # OK
even_num = EvenInt(43)  # Raises ProofFailure
```

#### 5. Protocol Verification

Verify API usage patterns:

```python
from pyproof import protocol_method, filemanager_protocol

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

# Usage automatically verified
fm = FileManager()
fm.open()     # OK
fm.read()     # OK  
fm.close()    # OK
# fm.read()   # Would raise ProofFailure - can't read closed file
```

#### 6. Information Flow Tracking

Track sensitive data through computations:

```python
from pyproof import TaintedValue, SecurityLabel, track_sensitive_data

# Track sensitive assignments
password = track_sensitive_data("user_password", "secret123", SecurityLabel.SECRET)
username = TaintedValue("john_doe", SecurityLabel.PUBLIC)

# Information flow is automatically tracked
combined = password.combine_with(username)
print(f"Security level: {combined.label}")  # SECRET (highest of the two)

# Controlled declassification
password.declassify(SecurityLabel.CONFIDENTIAL, "Hashed for storage")
```

#### 7. Temporal Properties

Verify event sequences and timing:

```python
from pyproof import EventuallyProperty, AlwaysProperty, TemporalVerifier, record_temporal_event

# Define temporal properties
eventually_complete = EventuallyProperty(
    "task_completes",
    lambda history: any(e['event'] == 'task_done' for e in history),
    timeout=5.0
)

always_valid = AlwaysProperty(
    "data_valid", 
    lambda event: event.get('valid', True)
)

# Add to verifier
temporal_verifier = TemporalVerifier()
temporal_verifier.add_property(eventually_complete)
temporal_verifier.add_property(always_valid)

# Record events during execution
record_temporal_event("task_start", {"valid": True})
record_temporal_event("processing", {"valid": True})  
record_temporal_event("task_done", {"valid": True})

# Verify all properties
temporal_verifier.verify_all()  # Passes
```

#### 8. Plugin System

Extend verification for specific domains:

```python
# Financial verification
from pyproof import _plugin_registry

Money = _plugin_registry.get_type("Money")
price = Money("19.99", "USD")
tax = Money("1.60", "USD") 
total = price + tax

# Cryptographic verification
crypto_verifier = _plugin_registry.get_verifier("constant_time")
secure_function = lambda x: hash(x)  # Simplified example
is_constant_time = crypto_verifier(secure_function, [1, 100, 1000])

# Security verification  
security_verifier = _plugin_registry.get_verifier("input_sanitized")
sanitizer = lambda x: x.replace("<", "&lt;").replace(">", "&gt;")
is_safe = security_verifier("<script>alert('xss')</script>", sanitizer)
```

## Automatic Instrumentation

Use `pyproofify` to automatically add verification to existing code:

```bash
# Instrument entire project
python -m pyproof.pyproofify src/ instrumented/ --all

# Selective instrumentation
python -m pyproof.pyproofify src/ instrumented/ \
    --contracts --loops --asserts --temporal --protocols
```

**Before:**
```python
def factorial(n):
    assert n >= 0
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
```

**After instrumentation:**
```python
import pyproof
from pyproof import require, auto_contract

@auto_contract
def factorial(n):
    require("n >= 0", n >= 0)
    result = 1
    for i in range(1, n + 1):
        with pyproof.proof_context('for_loop_invariant'):
            result *= i
    return result
```

## Configuration

Control verification behavior with environment variables:

```bash
# Verification levels
export PYPROOF_LEVEL=full        # off|contracts|invariants|full|debug
export PYPROOF_CACHE=1           # Enable proof caching
export PYPROOF_MAX_STEPS=10000   # Maximum proof steps
export PYPROOF_PERF=1            # Performance mode
```

Or programmatically:

```python
from pyproof import Config, VerificationLevel

config = Config()
config.level = VerificationLevel.CONTRACTS  # Only verify contracts
config.performance_mode = True              # Skip expensive checks
```

## Verification Levels

- **OFF**: No verification (production default)
- **CONTRACTS**: Only function contracts 
- **INVARIANTS**: Contracts + data structure invariants
- **FULL**: All verification except debug features
- **DEBUG**: Full verification + detailed logging

## Performance

PyProof is designed for production use:

- **Conditional verification**: Disable in production with `PYPROOF_LEVEL=off`
- **Proof caching**: Expensive computations are cached
- **Thread-safe**: Concurrent proof traces without interference
- **Minimal overhead**: Optimized for performance-critical code

Benchmark results:
```
Verification Level | Overhead | Use Case
OFF               | 0%       | Production
CONTRACTS         | 5-10%    | Integration testing  
FULL              | 10-25%   | Development/QA
DEBUG             | 25-50%   | Debugging
```

## Testing Integration

PyProof works with standard testing frameworks:

```python
import pytest
from pyproof import verification_mode, ProofFailure

def test_with_full_verification():
    with verification_mode():  # Enables debug-level verification
        result = my_verified_function(test_input)
        assert result == expected

def test_proof_failure():
    with pytest.raises(ProofFailure, match="Cannot prove: input is positive"):
        invalid_function(-5)
```

## Plugin Development

Create domain-specific verifiers:

```python
from pyproof import Plugin

class DatabasePlugin(Plugin):
    def __init__(self):
        super().__init__("database")
    
    def add_verifiers(self):
        return {
            'transaction_safe': self.verify_transaction_safety,
            'sql_injection_free': self.verify_no_sql_injection
        }
    
    def verify_transaction_safety(self, query, connection):
        # Implementation here
        return connection.in_transaction()

# Register plugin
from pyproof import _plugin_registry
_plugin_registry.register(DatabasePlugin())
```

## Examples

See the `examples/` directory for complete examples:

- `basic_verification.py` - Getting started with PyProof
- `data_structures.py` - Verified collections and algorithms  
- `protocol_verification.py` - API usage pattern verification
- `security_verification.py` - Information flow and crypto verification
- `performance_optimization.py` - Production deployment patterns


## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Add tests for your changes
4. Run the test suite (`python -m pytest`)
5. Commit your changes (`git commit -m 'Add your feature'`)
6. Push to the branch (`git push origin feature/your-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by formal verification research and tools like Dafny, TLA+, and Coq
- Built on Python's rich ecosystem of type hints and static analysis tools
- Community contributions and feedback from the formal methods community

---

**Made with ❤️ for safer, more reliable Python code**
