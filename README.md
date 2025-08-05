# PyProof: Runtime Formal Verification for Python

## Overview

PyProof is a runtime verification system that brings formal verification concepts to practical Python programming. Instead of requiring separate proof languages or complex theorem provers, PyProof embeds verification directly into your code using decorators, context managers, and runtime assertions.

The system bridges the gap between informal testing and formal verification by:
- **Proving properties at runtime** rather than just checking them
- **Building proof traces** that document what was verified
- **Failing fast** when properties can't be proven
- **Composing proofs** across function calls and data structures

## Core Concepts

### 1. The Proof System

At the heart of PyProof is the `Proof` class that tracks verification steps:

```python
class Proof:
    def __init__(self):
        self.steps = []
    
    def require(self, claim, evidence):
        """Runtime check that's also a proof step"""
        if not evidence:
            raise ProofFailure(f"Cannot prove: {claim}")
        self.steps.append(claim)
        return evidence
```

Every `require()` call either:
- **Succeeds**: Adds the claim to the proof trace
- **Fails**: Raises `ProofFailure` with details

### 2. Proof vs. Testing

Traditional testing checks examples; PyProof proves properties:

```python
# Testing approach
assert divide(10, 2) == 5
assert divide(7, 3) == 2.333...

# Proof approach  
def proven_divide(a, b):
    require("b is not zero", b != 0)
    result = a / b
    require("result * b equals a", abs(result * b - a) < 1e-10)
    return result
```

The proof approach guarantees the property holds for **all** inputs that satisfy the preconditions.

## System Components

### Function Contracts

Separate assumptions (preconditions) from guarantees (postconditions):

```python
@contract(
    preconditions=[
        ("input is a list", lambda lst: isinstance(lst, list)),
        ("list is not empty", lambda lst: len(lst) > 0)
    ],
    postconditions=[
        ("result is maximum element", lambda lst, result: result == max(lst)),
        ("result is in original list", lambda lst, result: result in lst)
    ]
)
def find_maximum(lst):
    return max(lst)
```

**Benefits:**
- Clear separation of concerns
- Automatic verification at function boundaries  
- Composable - proven functions can be trusted by callers
- Self-documenting interfaces

### Loop Invariants

Prove properties that hold throughout loop execution:

```python
def proven_sum(numbers):
    total = 0
    i = 0
    
    def sum_invariant(total, i, numbers):
        return total == sum(numbers[:i])  # "total equals sum of first i elements"
    
    with ProvenLoop(sum_invariant) as loop:
        while i < len(numbers) and loop.iterate(total=total, i=i, numbers=numbers):
            total += numbers[i]
            i += 1
            # Invariant automatically checked after each iteration
    
    return total
```

**Key Properties:**
- Invariant checked before first iteration
- Invariant maintained after each iteration  
- Termination guaranteed by max iteration bounds
- Progress tracking prevents infinite loops

### Data Structure Invariants

Maintain structural properties across operations:

```python
class ProvenSortedList:
    def __init__(self, items=None):
        self.items = items or []
        require("initially sorted", self._is_sorted())
    
    def _is_sorted(self):
        return all(self.items[i] <= self.items[i+1] 
                  for i in range(len(self.items)-1))
    
    def insert(self, value):
        # Find correct position
        pos = bisect.bisect_left(self.items, value)
        self.items.insert(pos, value)
        
        # Prove invariant maintained
        require("remains sorted after insert", self._is_sorted())
        require("value was inserted", value in self.items)
        return self
```

### State Machine Verification

Prove systems follow valid protocols:

```python
class ProvenStateMachine:
    def __init__(self, initial_state, transitions):
        self.state = initial_state
        self.transitions = transitions
        require("initial state is valid", initial_state in transitions)
    
    def transition_to(self, new_state):
        require("transition is allowed", 
                new_state in self.transitions.get(self.state, []))
        
        old_state = self.state
        self.state = new_state
        
        require("state updated correctly", self.state == new_state)
        return self

# Usage: File handle protocol
file_machine = ProvenStateMachine('closed', {
    'closed': ['opening'],
    'opening': ['open', 'error'],
    'open': ['reading', 'writing', 'closing'],
    'reading': ['open', 'error'],
    'writing': ['open', 'error'],  
    'closing': ['closed', 'error'],
    'error': ['closed']
})
```

### Resource Lifecycle Management

Prove resources are properly acquired and released:

```python
class ProvenResource:
    _active_resources = set()
    
    def __init__(self, resource_id):
        require("resource available", resource_id not in self._active_resources)
        self.resource_id = resource_id
        self.acquired = True
        self._active_resources.add(resource_id)
    
    def release(self):
        require("resource is acquired", self.acquired)
        self.acquired = False
        self._active_resources.remove(self.resource_id)
    
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.acquired:
            self.release()

# Prove no resource leaks
@contract(
    postconditions=[
        ("no resources leaked", lambda result: len(ProvenResource._active_resources) == 0)
    ]
)
def process_data():
    with ProvenResource("database") as db:
        with ProvenResource("file_handle") as f:
            return "processed"
```

### Ghost State

Add proof-only data that doesn't affect program execution:

```python
class GhostState:
    def __init__(self):
        self._data = {}
    
    def set(self, key, value):
        self._data[key] = value
    
    def get(self, key):
        return self._data.get(key)

_ghost = GhostState()

def proven_reverse(lst):
    require("input is a list", isinstance(lst, list))
    
    # Ghost state: remember original
    _ghost.set('original', lst.copy())
    _ghost.set('original_length', len(lst))
    
    # Reverse in place
    left, right = 0, len(lst) - 1
    while left < right:
        lst[left], lst[right] = lst[right], lst[left]
        left += 1
        right -= 1
    
    # Prove correctness using ghost state
    original = _ghost.get('original')
    require("length preserved", len(lst) == _ghost.get('original_length'))
    require("elements reversed", 
            all(lst[i] == original[-(i+1)] for i in range(len(lst))))
    
    return lst
```

## Architecture

### Core Classes

```
Proof                    # Tracks verification steps
├── require()           # Add proof step or fail
└── steps              # List of proven claims

ProofFailure            # Exception for unprovable claims

Contract               # Function pre/postcondition decorator
├── preconditions      # List of (claim, predicate) pairs  
└── postconditions     # List of (claim, predicate) pairs

ProvenLoop             # Loop invariant verification
├── invariant_fn       # Invariant predicate function
├── max_iterations     # Termination bound
└── iterate()          # Check invariant, advance iteration

GhostState             # Proof-only auxiliary data
├── set()              # Store ghost variables
└── get()              # Retrieve ghost variables
```

### Global State

```python
_proof = Proof()        # Global proof trace
_ghost = GhostState()   # Global ghost state

def require(claim, evidence):
    """Global require function"""
    return _proof.require(claim, evidence)
```

## Usage Patterns

### Progressive Verification

Start simple, add more verification as needed:

```python
# Level 1: Basic assertions
def divide(a, b):
    assert b != 0
    return a / b

# Level 2: Proof-based assertions  
def divide(a, b):
    require("denominator not zero", b != 0)
    return a / b

# Level 3: Full contracts
@contract(
    preconditions=[("b != 0", lambda a, b: b != 0)],
    postconditions=[("result * b ≈ a", lambda a, b, result: abs(result * b - a) < 1e-10)]
)
def divide(a, b):
    return a / b
```

### Proof Composition

Proven functions can be used confidently:

```python
@contract(...)
def proven_sqrt(x):
    # ... implementation with proofs
    return result

@contract(...)  
def proven_distance(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    # Can trust proven_sqrt because of its contract
    return proven_sqrt(dx*dx + dy*dy)
```

### Debug vs. Production

```python
import os

# Only verify in debug mode
VERIFY = os.getenv('PYPROOF_VERIFY', '1') == '1'

def require(claim, evidence):
    if VERIFY:
        return _proof.require(claim, evidence)
    return evidence
```

## Benefits

### For Development
- **Catch bugs early** - Properties are checked at runtime
- **Better documentation** - Contracts explain function behavior
- **Incremental adoption** - Add verification gradually
- **No new tools** - Pure Python, works with existing workflow

### For Maintenance  
- **Executable specifications** - Contracts stay in sync with code
- **Regression prevention** - Proofs prevent breaking changes
- **Refactoring confidence** - Invariants ensure correctness
- **Clear interfaces** - Pre/postconditions document assumptions

### For Reliability
- **Mathematical rigor** - Properties are proven, not just tested
- **Comprehensive coverage** - Invariants hold for all executions
- **Composable guarantees** - Verified components can be trusted
- **Audit trails** - Proof traces show what was verified

## When to Use PyProof

**Ideal for:**
- Critical algorithms with mathematical properties
- Data structures with invariants  
- Resource management code
- Protocol implementations
- Security-sensitive functions
- Code with complex preconditions

**Consider alternatives for:**
- Simple CRUD operations
- UI event handling
- Performance-critical inner loops
- Prototype/exploratory code

## Getting Started

1. **Start with basic `require()` statements** for critical properties
2. **Add function contracts** for key interfaces  
3. **Use loop invariants** for complex algorithms
4. **Add data structure invariants** for custom classes
5. **Consider ghost state** for complex mathematical proofs

The system grows with your needs - start simple and add verification as your confidence and requirements increase.