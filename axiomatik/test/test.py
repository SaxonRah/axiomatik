# test_example.py - Comprehensive test file for axiomatikify features
"""
Test file demonstrating all Axiomatik instrumentation features:
- Loops (for/while with invariants)
- Ghost state tracking
- Assert to require conversion
- Temporal event recording
- Protocol detection and verification
- Auto-contract generation from type hints
- Information flow tracking
- Exception handling
"""

from typing import List, Optional, Dict, Union
import time
import threading


# =============================================================================
# 1. PROTOCOL CLASSES - Tests protocol detection and instrumentation
# =============================================================================

class FileManager:
    """File-like protocol that should be detected and instrumented"""

    def __init__(self, filename: str):
        self.filename = filename
        self.is_open = False
        self.content = ""
        self.position = 0

    def open(self) -> bool:
        """Open the file - should get protocol instrumentation"""
        assert not self.is_open, "File is already open"
        self.is_open = True
        self.position = 0
        # Simulate loading some content
        self.content = f"Sample content from {self.filename}"
        return True

    def read(self, size: int = -1) -> str:
        """Read from file - should get protocol instrumentation"""
        assert self.is_open, "Cannot read from closed file"
        assert size >= -1, "Size must be -1 or positive"

        if size == -1:
            result = self.content[self.position:]
            self.position = len(self.content)
        else:
            end_pos = min(self.position + size, len(self.content))
            result = self.content[self.position:end_pos]
            self.position = end_pos

        return result

    def write(self, data: str) -> int:
        """Write to file - should get protocol instrumentation"""
        assert self.is_open, "Cannot write to closed file"
        assert isinstance(data, str), "Data must be string"

        # Append data to content
        self.content += data
        written_bytes = len(data)

        return written_bytes

    def close(self) -> None:
        """Close the file - should get protocol instrumentation"""
        assert self.is_open, "File is already closed"
        self.is_open = False
        self.position = 0


class StateMachine:
    """State machine protocol that should be detected and instrumented"""

    def __init__(self):
        self.state = "stopped"
        self.data = {}

    def start(self) -> bool:
        """Start the machine - should get protocol instrumentation"""
        assert self.state == "stopped", "Machine must be stopped to start"
        self.state = "running"
        return True

    def stop(self) -> bool:
        """Stop the machine - should get protocol instrumentation"""
        assert self.state == "running", "Machine must be running to stop"
        self.state = "stopped"
        return True

    def reset(self) -> None:
        """Reset the machine - should get protocol instrumentation"""
        self.state = "stopped"
        self.data.clear()

    def process(self, item: str) -> str:
        """Process an item"""
        assert self.state == "running", "Machine must be running to process"
        assert len(item) > 0, "Item cannot be empty"

        result = f"processed_{item}"
        self.data[item] = result
        return result


# =============================================================================
# 2. MATHEMATICAL FUNCTIONS - Tests loops, asserts, and contracts
# =============================================================================

def factorial(n: int) -> int:
    """Calculate factorial with loops and assertions - tests multiple features"""
    assert n >= 0, "Factorial is only defined for non-negative integers"
    assert n <= 20, "Input too large for factorial calculation"

    if n <= 1:
        return 1

    result = 1
    i = 2

    # This while loop should get instrumented with proof context
    while i <= n:
        # Loop invariant: result == (i-1)!
        assert result > 0, "Result should always be positive"
        result *= i
        i += 1

        # Simulate some ghost state that might be tracked
        if i % 5 == 0:
            # Ghost state: mark every 5th iteration
            pass

    assert result > 0, "Final result should be positive"
    return result


def fibonacci_sequence(count: int) -> List[int]:
    """Generate Fibonacci sequence with for loop and type hints"""
    assert count >= 0, "Count must be non-negative"
    assert count <= 100, "Count too large"

    if count == 0:
        return []
    elif count == 1:
        return [0]
    elif count == 2:
        return [0, 1]

    sequence = [0, 1]

    # This for loop should get instrumented
    for i in range(2, count):
        # Loop invariant: sequence has exactly i elements
        assert len(sequence) == i, "Sequence length should match iteration"
        next_val = sequence[i - 1] + sequence[i - 2]
        # Fixed: Fibonacci is non-decreasing, not strictly increasing (0,1,1,2,3,5...)
        assert next_val >= sequence[i - 1], "Fibonacci should be non-decreasing"
        sequence.append(next_val)

    assert len(sequence) == count, "Final sequence length should match count"
    return sequence


def binary_search(arr: List[int], target: int) -> Optional[int]:
    """Binary search with nested assertions and ghost state tracking"""
    assert len(arr) > 0, "Array cannot be empty"

    # Check if array is sorted (expensive check for ghost state)
    for i in range(len(arr) - 1):
        assert arr[i] <= arr[i + 1], "Array must be sorted for binary search"

    left = 0
    right = len(arr) - 1

    # This while loop should get instrumented
    while left <= right:
        # Loop invariant: if target exists, it's in arr[left:right+1]
        assert 0 <= left <= len(arr), "Left bound should be valid"
        assert 0 <= right < len(arr), "Right bound should be valid"

        mid = (left + right) // 2
        assert left <= mid <= right, "Mid should be between bounds"

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return None


# =============================================================================
# 3. SENSITIVE DATA HANDLING - Tests information flow tracking
# =============================================================================

def handle_credentials(username: str, password: str, api_key: str) -> Dict[str, str]:
    """Handle sensitive data - should trigger information flow tracking"""
    assert len(username) > 0, "Username cannot be empty"
    assert len(password) >= 8, "Password must be at least 8 characters"
    assert len(api_key) >= 16, "API key must be at least 16 characters"

    # These assignments should be detected as sensitive
    secret_token = f"secret_{api_key}"
    credential_hash = abs(hash(password))  # Use abs to avoid negative numbers
    auth_key = f"Bearer {api_key}"

    # Simulate some processing
    processed_creds = {
        "user": username,
        "token": secret_token,
        "hash": str(credential_hash),
        "auth": auth_key
    }

    return processed_creds


def encrypt_data(data: str, encryption_key: str) -> str:
    """Encrypt data with key validation"""
    assert len(data) > 0, "Data cannot be empty"
    assert len(encryption_key) >= 32, "Encryption key must be at least 32 characters"

    # Simulate encryption (sensitive operation)
    encrypted = ""
    for i, char in enumerate(data):
        key_char = encryption_key[i % len(encryption_key)]
        # Simple Caesar cipher variant
        encrypted_char = chr((ord(char) + ord(key_char)) % 128 + 32)  # Keep in printable range
        encrypted += encrypted_char

    assert len(encrypted) == len(data), "Encrypted data should have same length"
    return encrypted


# =============================================================================
# 4. TEMPORAL PROCESSING - Tests temporal event recording
# =============================================================================

def multi_step_process(items: List[str]) -> Dict[str, str]:
    """Multi-step process that should generate temporal events"""
    assert len(items) > 0, "Items list cannot be empty"

    results = {}

    # Step 1: Validation
    for item in items:
        assert isinstance(item, str), "All items must be strings"
        assert len(item) > 0, "Items cannot be empty"

    # Step 2: Processing with timing
    start_time = time.time()

    for item in items:
        # Simulate some processing time
        time.sleep(0.001)  # 1ms delay
        processed = item.upper().strip()
        results[item] = processed

    end_time = time.time()
    processing_time = end_time - start_time

    # Step 3: Validation of results
    assert len(results) == len(items), "Should have result for each item"
    assert processing_time >= 0, "Processing time should be non-negative"

    return results


def async_worker_simulation(task_id: int, work_items: List[str]) -> bool:
    """Simulate async work - should get temporal instrumentation"""
    assert task_id > 0, "Task ID must be positive"
    assert len(work_items) > 0, "Must have work items"

    thread_id = threading.get_ident()

    # Simulate work phases
    for phase in ["initialize", "process", "finalize"]:
        for item in work_items:
            # Simulate work
            result = f"{phase}_{item}_{task_id}_{thread_id}"
            assert len(result) > 0, "Result should not be empty"
            time.sleep(0.001)  # Small delay

    return True


# =============================================================================
# 5. EXCEPTION HANDLING - Tests exception instrumentation
# =============================================================================

def risky_operation(value: int, denominator: int) -> float:
    """Operation with exception handling - should get temporal instrumentation"""
    assert isinstance(value, int), "Value must be integer"
    assert isinstance(denominator, int), "Denominator must be integer"

    try:
        # This try block should get temporal instrumentation
        if denominator == 0:
            raise ValueError("Cannot divide by zero")

        result = value / denominator
        assert isinstance(result, float), "Result should be float"
        return result

    except ValueError as e:
        # This except block should get temporal instrumentation
        assert "divide by zero" in str(e), "Should be division error"
        return float('inf')

    except Exception as e:
        # This general except should also get instrumentation
        assert isinstance(e, Exception), "Should be an exception"
        return float('nan')

    finally:
        # Finally block should also get instrumentation
        pass


def file_processing_with_errors(filename: str) -> Optional[str]:
    """File processing with comprehensive error handling"""
    assert len(filename) > 0, "Filename cannot be empty"

    try:
        # Simulate file operations that might fail
        if "invalid" in filename:
            raise FileNotFoundError(f"File {filename} not found")

        if "corrupt" in filename:
            raise ValueError(f"File {filename} is corrupted")

        # Simulate successful processing
        content = f"Content of {filename}"
        assert len(content) > 0, "Content should not be empty"
        return content

    except FileNotFoundError as e:
        assert "not found" in str(e), "Should be file not found error"
        return None

    except ValueError as e:
        assert "corrupted" in str(e), "Should be corruption error"
        return None

    except Exception as e:
        # Catch-all for unexpected errors
        return None

    finally:
        # Cleanup operations
        pass


# =============================================================================
# 6. COMPLEX WORKFLOWS - Tests multiple features together
# =============================================================================

def comprehensive_workflow(data: List[Dict[str, Union[str, int]]]) -> Dict[str, List[str]]:
    """Complex workflow combining multiple features"""
    assert len(data) > 0, "Data cannot be empty"
    assert all(isinstance(item, dict) for item in data), "All items must be dictionaries"

    results = {
        "processed": [],
        "errors": [],
        "stats": []
    }

    # Phase 1: Validation with loop
    for i, item in enumerate(data):
        assert "id" in item, f"Item {i} missing required 'id' field"
        assert "value" in item, f"Item {i} missing required 'value' field"

        # Nested validation loop
        for key, value in item.items():
            assert key is not None, "Key cannot be None"
            assert value is not None, "Value cannot be None"

    # Phase 2: Processing with error handling
    for item in data:
        try:
            item_id = str(item["id"])
            item_value = str(item["value"])

            # Simulate processing that might fail
            if "error" in item_value:
                raise ValueError(f"Invalid value: {item_value}")

            processed_value = f"processed_{item_id}_{item_value}"
            results["processed"].append(processed_value)

        except ValueError as e:
            error_msg = f"Error processing item {item}: {e}"
            results["errors"].append(error_msg)

        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            results["errors"].append(error_msg)

    # Phase 3: Statistics calculation
    total_items = len(data)
    successful_items = len(results["processed"])
    error_items = len(results["errors"])

    assert total_items == successful_items + error_items, "Item counts should match"

    stats = [
        f"Total: {total_items}",
        f"Successful: {successful_items}",
        f"Errors: {error_items}",
        f"Success rate: {successful_items / total_items:.2%}" if total_items > 0 else "Success rate: 0%"
    ]

    results["stats"] = stats

    return results


# =============================================================================
# 7. SIMPLE ARITHMETIC FUNCTIONS - Basic test cases
# =============================================================================

def simple_addition(a: int, b: int) -> int:
    """Simple addition with basic assertions"""
    assert isinstance(a, int), "First argument must be integer"
    assert isinstance(b, int), "Second argument must be integer"

    result = a + b
    assert isinstance(result, int), "Result must be integer"
    return result


def count_items(items: List[str]) -> int:
    """Count items with simple loop"""
    assert isinstance(items, list), "Input must be a list"

    count = 0
    for item in items:
        assert isinstance(item, str), "All items must be strings"
        count += 1

    assert count == len(items), "Count should match list length"
    return count


def find_maximum(numbers: List[int]) -> int:
    """Find maximum with loop and assertions"""
    assert len(numbers) > 0, "List cannot be empty"
    assert all(isinstance(n, int) for n in numbers), "All items must be integers"

    max_val = numbers[0]

    for num in numbers[1:]:
        if num > max_val:
            max_val = num
        assert max_val >= num or max_val == num, "Max should be greater or equal"

    assert max_val in numbers, "Maximum should be in the original list"
    return max_val


# =============================================================================
# 8. MAIN TESTING FUNCTION
# =============================================================================

def main() -> None:
    """Main function to test all features"""
    print("Starting comprehensive Axiomatik feature test...")

    # Test 1: Protocol classes
    print("\n1. Testing Protocol Classes:")
    fm = FileManager("test.txt")
    fm.open()
    fm.write("Hello, World!")
    content = fm.read()
    fm.close()
    print(f"FileManager test: '{content}'")

    sm = StateMachine()
    sm.start()
    result = sm.process("test_item")
    sm.stop()
    print(f"StateMachine test: {result}")

    # Test 2: Mathematical functions
    print("\n2. Testing Mathematical Functions:")
    fact_result = factorial(5)
    print(f"Factorial of 5: {fact_result}")

    fib_result = fibonacci_sequence(10)
    print(f"Fibonacci sequence (10): {fib_result}")

    search_result = binary_search([1, 3, 5, 7, 9, 11], 7)
    print(f"Binary search result: {search_result}")

    # Test 3: Simple arithmetic
    print("\n3. Testing Simple Functions:")
    add_result = simple_addition(5, 3)
    print(f"Addition: {add_result}")

    count_result = count_items(["a", "b", "c", "d"])
    print(f"Count: {count_result}")

    max_result = find_maximum([3, 1, 4, 1, 5, 9, 2, 6])
    print(f"Maximum: {max_result}")

    # Test 4: Sensitive data handling
    print("\n4. Testing Sensitive Data Handling:")
    creds = handle_credentials("user123", "password123", "api_key_1234567890abcdef")
    print(f"Credentials processed: {len(creds)} fields")

    encrypted = encrypt_data("secret message", "encryption_key_32_characters_long!")
    print(f"Encryption test: {len(encrypted)} characters")

    # Test 5: Temporal processing
    print("\n5. Testing Temporal Processing:")
    items = ["item1", "item2", "item3"]
    processed = multi_step_process(items)
    print(f"Multi-step process: {processed}")

    worker_result = async_worker_simulation(1, ["task1", "task2"])
    print(f"Async worker: {worker_result}")

    # Test 6: Exception handling
    print("\n6. Testing Exception Handling:")
    safe_result = risky_operation(10, 2)
    print(f"Safe operation: {safe_result}")

    error_result = risky_operation(10, 0)
    print(f"Error operation: {error_result}")

    file_result = file_processing_with_errors("valid.txt")
    print(f"File processing: {file_result}")

    # Test 7: Complex workflow
    print("\n7. Testing Complex Workflow:")
    test_data = [
        {"id": 1, "value": "data1"},
        {"id": 2, "value": "data2"},
        {"id": 3, "value": "error_data"},
        {"id": 4, "value": "data4"}
    ]
    workflow_result = comprehensive_workflow(test_data)
    print(f"Workflow stats: {workflow_result['stats']}")

    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    main()