#!/usr/bin/env python3
"""
protocol_verification.py - API usage pattern verification

This example demonstrates:
- File-like protocol verification
- Database connection protocols
- State machine protocols
- HTTP client protocols
- Custom protocol definition
- Protocol violation detection

Run with: python protocol_verification.py
"""

import time
import threading
from typing import Optional, Dict, Any
import pyproof.pyproof
from pyproof.pyproof import require, protocol_method, Protocol, ProtocolState, ProofFailure
from pyproof.pyproof import filemanager_protocol, statemachine_protocol, dbconnection_protocol


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. FILE-LIKE PROTOCOL IMPLEMENTATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class VerifiedFileManager:
    """File manager with verified protocol compliance"""

    def __init__(self, filename: str):
        self.filename = filename
        self.is_open = False
        self.content = ""
        self.position = 0
        self.mode = None

    @protocol_method(filemanager_protocol, "open")
    def open(self, mode: str = "r") -> bool:
        """Open file - verified to follow file protocol"""
        require("file is not already open", not self.is_open)
        require("mode is valid", mode in ["r", "w", "a", "r+", "w+", "a+"])

        self.is_open = True
        self.mode = mode
        self.position = 0

        # Simulate loading content for read modes
        if "r" in mode:
            self.content = f"Sample content from {self.filename}\nLine 2\nLine 3"
        elif mode in ["w", "w+"]:
            self.content = ""  # Clear content for write mode

        require("file is now open", self.is_open)
        require("mode is set", self.mode == mode)
        return True

    @protocol_method(filemanager_protocol, "read")
    def read(self, size: int = -1) -> str:
        """Read from file - protocol verified"""
        require("file is open", self.is_open)
        require("mode allows reading", "r" in self.mode or "+" in self.mode)
        require("size is valid", size >= -1)

        if size == -1:
            # Read all remaining content
            result = self.content[self.position:]
            self.position = len(self.content)
        else:
            # Read specified amount
            end_pos = min(self.position + size, len(self.content))
            result = self.content[self.position:end_pos]
            self.position = end_pos

        require("position is valid", 0 <= self.position <= len(self.content))
        return result

    @protocol_method(filemanager_protocol, "write")
    def write(self, data: str) -> int:
        """Write to file - protocol verified"""
        require("file is open", self.is_open)
        require("mode allows writing", "w" in self.mode or "a" in self.mode or "+" in self.mode)
        require("data is string", isinstance(data, str))

        if "a" in self.mode:
            # Append mode
            self.content += data
            self.position = len(self.content)
        else:
            # Write/overwrite mode
            if self.mode in ["w", "w+"]:
                # Overwrite from current position
                before = self.content[:self.position]
                after = self.content[self.position + len(data):]
                self.content = before + data + after
            else:
                # Insert at current position
                before = self.content[:self.position]
                after = self.content[self.position:]
                self.content = before + data + after

            self.position += len(data)

        bytes_written = len(data)
        require("wrote correct amount", bytes_written == len(data))
        require("position updated correctly", self.position <= len(self.content))
        return bytes_written

    @protocol_method(filemanager_protocol, "closed")  # Target "closed" directly
    def close(self) -> None:
        """Close file - protocol verified"""
        require("file is open", self.is_open)

        self.is_open = False
        self.mode = None
        self.position = 0

        require("file is now closed", not self.is_open)
        require("mode is cleared", self.mode is None)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2. DATABASE CONNECTION PROTOCOL
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class VerifiedDatabaseConnection:
    """Database connection with protocol verification"""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.is_connected = False
        self.in_transaction = False
        self.cursor = None

    @protocol_method(dbconnection_protocol, "connected")
    def connect(self) -> bool:
        """Connect to database"""
        require("not already connected", not self.is_connected)
        require("connection string is valid", len(self.connection_string) > 0)

        # Simulate connection
        time.sleep(0.001)  # Simulate connection time
        self.is_connected = True

        require("now connected", self.is_connected)
        return True

    @protocol_method(dbconnection_protocol, "connected")
    def execute_query(self, query: str) -> Dict[str, Any]:
        """Execute query - must be connected"""
        require("is connected", self.is_connected)
        require("not in transaction for DDL",
                not self.in_transaction or not query.strip().upper().startswith(('CREATE', 'DROP', 'ALTER')))
        require("query is not empty", len(query.strip()) > 0)

        # Simulate query execution
        time.sleep(0.002)  # Simulate query time

        result = {
            'query': query,
            'rows_affected': 1 if query.strip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')) else 0,
            'result_set': [{'id': 1, 'value': 'test'}] if query.strip().upper().startswith('SELECT') else None,
            'execution_time': 0.002
        }

        require("result is valid", isinstance(result, dict))
        return result

    @protocol_method(dbconnection_protocol, "transaction")
    def begin_transaction(self) -> bool:
        """Begin database transaction"""
        require("is connected", self.is_connected)
        require("not already in transaction", not self.in_transaction)

        self.in_transaction = True

        require("now in transaction", self.in_transaction)
        return True

    @protocol_method(dbconnection_protocol, "connected")
    def commit_transaction(self) -> bool:
        """Commit transaction"""
        require("is connected", self.is_connected)
        require("is in transaction", self.in_transaction)

        # Simulate commit
        time.sleep(0.001)
        self.in_transaction = False

        require("transaction ended", not self.in_transaction)
        return True

    @protocol_method(dbconnection_protocol, "connected")
    def rollback_transaction(self) -> bool:
        """Rollback transaction"""
        require("is connected", self.is_connected)
        require("is in transaction", self.in_transaction)

        # Simulate rollback
        self.in_transaction = False

        require("transaction ended", not self.in_transaction)
        return True

    @protocol_method(dbconnection_protocol, "disconnected")
    def disconnect(self) -> None:
        """Disconnect from database"""
        require("is connected", self.is_connected)
        require("not in transaction", not self.in_transaction)

        self.is_connected = False

        require("now disconnected", not self.is_connected)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 3. STATE MACHINE PROTOCOL
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class VerifiedWorkerMachine:
    """Worker state machine with verified transitions"""

    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.current_state = "stopped"
        self.processed_items = 0
        self.error_count = 0

    @protocol_method(statemachine_protocol, "running")
    def start(self) -> bool:
        """Start the worker machine"""
        require("worker is stopped", self.current_state == "stopped")
        require("worker_id is valid", len(self.worker_id) > 0)

        self.current_state = "running"
        self.processed_items = 0
        self.error_count = 0

        require("worker is now running", self.current_state == "running")
        return True

    @protocol_method(statemachine_protocol, "process")
    def process_item(self, item: str) -> Dict[str, Any]:
        """Process an item - must be running"""
        require("worker is running", self.current_state == "running")
        require("item is not empty", len(item.strip()) > 0)

        # Simulate processing
        try:
            if "error" in item.lower():
                raise ValueError(f"Simulated error processing {item}")

            result = f"processed_{item}_{self.worker_id}"
            self.processed_items += 1

            require("item was processed", len(result) > 0)
            require("processed count increased", self.processed_items > 0)

            return {
                'status': 'success',
                'result': result,
                'processed_count': self.processed_items
            }

        except Exception as e:
            self.error_count += 1
            return {
                'status': 'error',
                'error': str(e),
                'error_count': self.error_count
            }

    @protocol_method(statemachine_protocol, "stopped")
    def stop(self) -> Dict[str, Any]:
        """Stop the worker machine"""
        require("worker is running or processing",
                self.current_state in ["running", "process"])

        old_state = self.current_state
        self.current_state = "stopped"

        summary = {
            'previous_state': old_state,
            'total_processed': self.processed_items,
            'total_errors': self.error_count,
            'success_rate': (self.processed_items / max(1, self.processed_items + self.error_count))
        }

        require("worker is now stopped", self.current_state == "stopped")
        require("summary is valid", isinstance(summary, dict))
        return summary

    @protocol_method(statemachine_protocol, "stopped")
    def reset(self) -> None:
        """Reset worker to initial state"""
        require("worker is stopped", self.current_state == "stopped")

        self.processed_items = 0
        self.error_count = 0

        require("counters reset", self.processed_items == 0 and self.error_count == 0)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 4. CUSTOM HTTP CLIENT PROTOCOL
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Define HTTP client protocol
http_protocol = Protocol("HttpClient", "idle")
http_protocol.add_state(ProtocolState("idle", ["requesting"]))
http_protocol.add_state(ProtocolState("requesting", ["response_received", "error"]))
http_protocol.add_state(ProtocolState("response_received", ["idle"]))
http_protocol.add_state(ProtocolState("error", ["idle"]))


class VerifiedHttpClient:
    """HTTP client with verified request/response protocol"""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.last_response = None
        self.request_count = 0

    @protocol_method(http_protocol, "requesting")
    def send_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> bool:
        """Send HTTP request - transitions to requesting state"""
        require("method is valid", method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH"])
        require("endpoint is valid", len(endpoint) > 0)
        require("base_url is set", len(self.base_url) > 0)

        # Simulate sending request
        self.request_count += 1
        time.sleep(0.001)  # Simulate network delay

        require("request count increased", self.request_count > 0)
        return True

    @protocol_method(http_protocol, "response_received")
    def receive_response(self) -> Dict[str, Any]:
        """Receive HTTP response - transitions to response_received state"""
        # Simulate receiving response
        response = {
            'status_code': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': {'message': 'success', 'data': [1, 2, 3]},
            'request_id': self.request_count
        }

        self.last_response = response

        require("response is valid", isinstance(response, dict))
        require("status code is set", 'status_code' in response)
        require("response was stored", self.last_response == response)
        return response

    @protocol_method(http_protocol, "error")
    def handle_error(self, error_message: str) -> Dict[str, Any]:
        """Handle request error - transitions to error state"""
        require("error message provided", len(error_message) > 0)

        error_response = {
            'error': True,
            'message': error_message,
            'request_id': self.request_count,
            'timestamp': time.time()
        }

        self.last_response = error_response

        require("error response is valid", isinstance(error_response, dict))
        require("error flag is set", error_response.get('error') is True)
        return error_response

    @protocol_method(http_protocol, "idle")
    def reset(self) -> None:
        """Reset client to idle state"""
        self.last_response = None

        require("response cleared", self.last_response is None)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 5. RESOURCE MANAGEMENT PROTOCOL
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Define resource lifecycle protocol
resource_protocol = Protocol("ResourceLifecycle", "unacquired")
resource_protocol.add_state(ProtocolState("unacquired", ["acquiring"]))
resource_protocol.add_state(ProtocolState("acquiring", ["acquired", "failed"]))
resource_protocol.add_state(ProtocolState("acquired", ["using", "releasing"]))
#resource_protocol.add_state(ProtocolState("using", ["acquired", "error"]))
resource_protocol.add_state(ProtocolState("using", ["acquired", "error", "releasing"]))  # Added "releasing"
resource_protocol.add_state(ProtocolState("releasing", ["unacquired"]))
resource_protocol.add_state(ProtocolState("failed", ["unacquired"]))
resource_protocol.add_state(ProtocolState("error", ["releasing", "failed"]))


class VerifiedResourceManager:
    """Resource manager with lifecycle protocol verification"""

    _active_resources = {}  # Track all active resources
    _resource_lock = threading.Lock()

    def __init__(self, resource_id: str, resource_type: str):
        self.resource_id = resource_id
        self.resource_type = resource_type
        self.handle = None
        self.acquisition_time = None

    @protocol_method(resource_protocol, "acquiring")
    def begin_acquire(self) -> bool:
        """Begin resource acquisition"""
        require("resource_id is valid", len(self.resource_id) > 0)

        with self._resource_lock:
            require("resource is available", self.resource_id not in self._active_resources)

        return True

    @protocol_method(resource_protocol, "acquired")
    def complete_acquire(self) -> bool:
        """Complete resource acquisition"""
        # Simulate resource acquisition
        self.handle = f"handle_{self.resource_id}_{int(time.time())}"
        self.acquisition_time = time.time()

        with self._resource_lock:
            self._active_resources[self.resource_id] = {
                'type': self.resource_type,
                'handle': self.handle,
                'acquired_at': self.acquisition_time,
                'thread': threading.get_ident()
            }

        require("resource acquired", self.handle is not None)
        require("acquisition time set", self.acquisition_time is not None)
        require("resource tracked", self.resource_id in self._active_resources)
        return True

    @protocol_method(resource_protocol, "failed")
    def fail_acquire(self, reason: str) -> str:
        """Handle acquisition failure"""
        require("reason provided", len(reason) > 0)

        # Clean up any partial state
        self.handle = None
        self.acquisition_time = None

        require("state cleaned", self.handle is None)
        return f"Failed to acquire {self.resource_id}: {reason}"

    @protocol_method(resource_protocol, "using")
    def use_resource(self, operation: str) -> Any:
        """Use the acquired resource"""
        require("resource is acquired", self.handle is not None)
        require("operation specified", len(operation) > 0)

        with self._resource_lock:
            require("resource is tracked", self.resource_id in self._active_resources)

        # Simulate resource usage
        time.sleep(0.001)  # Simulate work
        result = f"operation_{operation}_on_{self.resource_id}_completed"

        require("operation completed", len(result) > 0)
        return result

    @protocol_method(resource_protocol, "error")
    def handle_error(self, error: str) -> str:
        """Handle resource error during usage"""
        require("error message provided", len(error) > 0)
        require("resource was acquired", self.handle is not None)

        error_info = f"Error in {self.resource_id}: {error} at {time.time()}"

        require("error info created", len(error_info) > 0)
        return error_info

    @protocol_method(resource_protocol, "releasing")
    def begin_release(self) -> bool:
        """Begin resource release"""
        require("resource is acquired", self.handle is not None)

        with self._resource_lock:
            require("resource is tracked", self.resource_id in self._active_resources)

        return True

    @protocol_method(resource_protocol, "unacquired")
    def complete_release(self) -> Dict[str, Any]:
        """Complete resource release"""
        require("resource was acquired", self.handle is not None)

        # Calculate usage duration
        release_time = time.time()
        usage_duration = release_time - self.acquisition_time if self.acquisition_time else 0

        # Clean up
        old_handle = self.handle
        self.handle = None
        self.acquisition_time = None

        with self._resource_lock:
            resource_info = self._active_resources.pop(self.resource_id, {})

        summary = {
            'resource_id': self.resource_id,
            'handle': old_handle,
            'usage_duration': usage_duration,
            'released_at': release_time,
            'resource_info': resource_info
        }

        require("resource released", self.handle is None)
        require("resource untracked", self.resource_id not in self._active_resources)
        require("summary created", isinstance(summary, dict))
        return summary

    @classmethod
    def get_active_resources(cls) -> Dict:
        """Get all currently active resources"""
        with cls._resource_lock:
            return cls._active_resources.copy()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 6. DEMONSTRATION FUNCTION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def demonstrate_protocol_violation():
    """Demonstrate protocol violations and their detection"""
    print("\n~~~~~~~ Protocol Violation Examples ~~~~~~~")

    # File protocol violation
    print("1. File Protocol Violation:")
    try:
        fm = VerifiedFileManager("test.txt")
        fm.read()  # Try to read without opening - should fail
    except ProofFailure as e:
        print(f"   +++ - Caught file protocol violation: {e}")

    # Database protocol violation
    print("\n2. Database Protocol Violation:")
    try:
        db = VerifiedDatabaseConnection("test://localhost")
        db.execute_query("SELECT * FROM users")  # Try to query without connecting
    except ProofFailure as e:
        print(f"   +++ - Caught database protocol violation: {e}")

    # State machine protocol violation
    print("\n3. State Machine Protocol Violation:")
    try:
        worker = VerifiedWorkerMachine("worker1")
        worker.process_item("test")  # Try to process without starting
    except ProofFailure as e:
        print(f"   +++ - Caught state machine protocol violation: {e}")


def demonstrate_correct_protocols():
    """Demonstrate correct protocol usage"""
    print("\n~~~~~~~ Correct Protocol Usage ~~~~~~~")

    # 1. File Protocol
    print("1. File Protocol:")
    fm = VerifiedFileManager("data.txt")
    fm.open("w+")
    bytes_written = fm.write("Hello, Protocol World!")
    print(f"   Wrote {bytes_written} bytes")

    fm.close()   # Close the file first before reopening

    fm.open("r") # Then reopen for reading
    content = fm.read()
    print(f"   Read: '{content[:20]}...'")
    fm.close()
    print("   +++ - File protocol completed successfully")

    # 2. Database Protocol
    print("\n2. Database Protocol:")
    db = VerifiedDatabaseConnection("postgresql://localhost:5432/testdb")
    db.connect()

    # Execute some queries
    result = db.execute_query("SELECT COUNT(*) FROM users")
    print(f"   Query result: {result['rows_affected']} rows affected")

    # Use transaction
    db.begin_transaction()
    db.execute_query("INSERT INTO users (name) VALUES ('Alice')")
    db.commit_transaction()

    db.disconnect()
    print("   +++ - Database protocol completed successfully")

    # 3. State Machine Protocol
    print("\n3. State Machine Protocol:")
    worker = VerifiedWorkerMachine("worker_001")
    worker.start()

    # Process items
    items = ["item1", "item2", "error_item", "item3"]
    for item in items:
        result = worker.process_item(item)
        status = result['status']
        print(f"   Processed '{item}': {status}")

    summary = worker.stop()
    print(f"   +++ - Worker completed: {summary['total_processed']} processed, "
          f"{summary['total_errors']} errors")

    # 4. HTTP Protocol
    print("\n4. HTTP Protocol:")
    client = VerifiedHttpClient("https://api.example.com")

    # Successful request
    client.send_request("GET", "/users")
    response = client.receive_response()
    print(f"   GET /users: {response['status_code']}")
    client.reset()

    # Error case
    client.send_request("POST", "/invalid")
    error_response = client.handle_error("404 Not Found")
    print(f"   POST /invalid: Error - {error_response['message']}")
    client.reset()
    print("   +++ - HTTP protocol completed successfully")

    # 5. Resource Management Protocol
    print("\n5. Resource Management Protocol:")
    resource = VerifiedResourceManager("db_connection_1", "database")

    # Acquire resource
    resource.begin_acquire()
    resource.complete_acquire()
    print(f"   Acquired resource: {resource.handle}")

    # Use resource
    result = resource.use_resource("SELECT")
    print(f"   Used resource: {result[:30]}...")

    # Release resource
    resource.begin_release()
    summary = resource.complete_release()
    print(f"   Released resource, used for {summary['usage_duration']:.3f}s")
    print("   +++ - Resource protocol completed successfully")

    # Check no resource leaks
    active = VerifiedResourceManager.get_active_resources()
    print(f"   Active resources: {len(active)} (should be 0)")


def demonstrate_protocol_verification():
    """Main demonstration function"""
    print("PyProof Protocol Verification Examples")
    print("~" * 80)

    # Show correct usage
    demonstrate_correct_protocols()

    # Show violation detection
    demonstrate_protocol_violation()

    # Show protocol states
    print("\n~~~~~~~ Protocol State Information ~~~~~~~")
    print(f"File Protocol States: {list(filemanager_protocol.states.keys())}")
    print(f"State Machine States: {list(statemachine_protocol.states.keys())}")
    print(f"Database States: {list(dbconnection_protocol.states.keys())}")
    print(f"HTTP Protocol States: {list(http_protocol.states.keys())}")
    print(f"Resource Protocol States: {list(resource_protocol.states.keys())}")

    # Show verification summary
    print("\n~~~~~~~ Verification Summary ~~~~~~~")
    summary = pyproof.pyproof._proof.get_summary()
    print(f"Total proof steps: {summary['total_steps']}")
    print(f"Contexts verified: {len(summary['contexts'])}")
    print(f"Thread safety: {summary['thread_count']} threads involved")

    if pyproof.pyproof._proof.steps:
        print("\nRecent protocol verifications:")
        protocol_steps = [s for s in pyproof.pyproof._proof.steps[-10:]
                          if 'protocol' in s.context.lower()]
        for step in protocol_steps[-5:]:
            print(f"  - {step.claim} ({step.context})")


if __name__ == "__main__":
    demonstrate_protocol_verification()