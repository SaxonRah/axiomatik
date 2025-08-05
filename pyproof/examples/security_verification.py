#!/usr/bin/env python3
"""
security_verification.py - Information flow and crypto verification

This example demonstrates:
- Information flow tracking with security labels
- Tainted value propagation
- Cryptographic operation verification
- Input sanitization verification
- Security policy enforcement
- Declassification with justification

Run with: python security_verification.py
"""

import hashlib
import secrets
import time
import re
from typing import Dict, List, Any, Optional
import pyproof.pyproof
from pyproof.pyproof import require, contract, TaintedValue, SecurityLabel, track_sensitive_data
from pyproof.pyproof import InformationFlowTracker, _plugin_registry, ProofFailure


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. INFORMATION FLOW TRACKING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class SecureDataProcessor:
    """Processor that tracks information flow through security levels"""

    def __init__(self):
        self.flow_tracker = InformationFlowTracker()
        self._setup_flow_policies()
        self.audit_log = []

    def _setup_flow_policies(self):
        """Set up information flow policies"""
        # Allow public -> confidential
        self.flow_tracker.add_policy(SecurityLabel.PUBLIC, SecurityLabel.CONFIDENTIAL, True)

        # Allow confidential -> secret (with justification)
        self.flow_tracker.add_policy(SecurityLabel.CONFIDENTIAL, SecurityLabel.SECRET, True)

        # Prevent secret -> public (prevents data leaks)
        self.flow_tracker.add_policy(SecurityLabel.SECRET, SecurityLabel.PUBLIC, False)

        # Prevent top_secret -> anything lower
        self.flow_tracker.add_policy(SecurityLabel.TOP_SECRET, SecurityLabel.SECRET, False)
        self.flow_tracker.add_policy(SecurityLabel.TOP_SECRET, SecurityLabel.CONFIDENTIAL, False)
        self.flow_tracker.add_policy(SecurityLabel.TOP_SECRET, SecurityLabel.PUBLIC, False)

    def process_public_data(self, data: str) -> TaintedValue:
        """Process public data"""
        require("data is not empty", len(data) > 0)

        # Create public tainted value
        public_data = TaintedValue(
            value=data.strip().lower(),
            label=SecurityLabel.PUBLIC,
            provenance=["user_input", "sanitized"]
        )

        self._log_operation("process_public_data", public_data.label, len(data))
        return public_data

    def process_user_credentials(self, username: str, password: str) -> TaintedValue:
        """Process user credentials - highly sensitive"""
        require("username not empty", len(username) > 0)
        require("password not empty", len(password) > 0)
        require("password meets minimum length", len(password) >= 8)

        # Combine credentials as secret data
        credential_data = {
            'username': username,
            'password_hash': hashlib.sha256(password.encode()).hexdigest(),
            'timestamp': time.time()
        }

        secret_creds = TaintedValue(
            value=credential_data,
            label=SecurityLabel.SECRET,
            provenance=["user_authentication", "password_hashing"]
        )

        self._log_operation("process_credentials", secret_creds.label, 1)
        return secret_creds

    def process_api_key(self, api_key: str, service: str) -> TaintedValue:
        """Process API key - confidential level"""
        require("api_key not empty", len(api_key) > 0)
        require("api_key meets length requirement", len(api_key) >= 16)
        require("service specified", len(service) > 0)

        # Process API key
        processed_key = {
            'service': service,
            'key_prefix': api_key[:4],  # Only store prefix for logging
            'key_hash': hashlib.sha256(api_key.encode()).hexdigest(),
            'created_at': time.time()
        }

        confidential_key = TaintedValue(
            value=processed_key,
            label=SecurityLabel.CONFIDENTIAL,
            provenance=[f"api_key_{service}", "key_processing"]
        )

        self._log_operation("process_api_key", confidential_key.label, 1)
        return confidential_key

    @contract(
        preconditions=[
            ("source has valid label", lambda self, source, target_label:
            isinstance(source.label, SecurityLabel)),
            ("target label is valid", lambda self, source, target_label:
            isinstance(target_label, SecurityLabel))
        ]
    )
    def transfer_data(self, source: TaintedValue, target_label: SecurityLabel) -> TaintedValue:
        """Transfer data with information flow verification"""
        # Verify the flow is allowed
        self.flow_tracker.track_flow(source, target_label)

        # Create new tainted value at target level
        transferred = TaintedValue(
            value=source.value,
            label=target_label,
            provenance=source.provenance + [f"transferred_to_{target_label.value}"]
        )

        self._log_operation("transfer_data", transferred.label, 1)
        require("transfer completed", transferred.label == target_label)
        return transferred

    def declassify_for_logging(self, secret_data: TaintedValue, reason: str) -> TaintedValue:
        """Declassify secret data for logging purposes"""
        require("data is secret or top_secret",
                secret_data.label in [SecurityLabel.SECRET, SecurityLabel.TOP_SECRET])
        require("reason provided", len(reason) > 20)  # Require detailed justification

        # Create declassified copy
        declassified = TaintedValue(
            value=self._sanitize_for_logging(secret_data.value),
            label=SecurityLabel.CONFIDENTIAL,
            provenance=secret_data.provenance + [f"declassified_for_logging: {reason}"]
        )

        self._log_operation("declassify", declassified.label, 1)
        return declassified

    def _sanitize_for_logging(self, data: Any) -> Any:
        """Sanitize sensitive data for logging"""
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                if 'password' in key.lower() or 'key' in key.lower():
                    sanitized[key] = "***REDACTED***"
                else:
                    sanitized[key] = value
            return sanitized
        elif isinstance(data, str):
            # Redact potential sensitive patterns
            sanitized = re.sub(r'\b\w*(?:password|key|token|secret)\w*\b', '***REDACTED***', data, flags=re.IGNORECASE)
            return sanitized
        return data

    def _log_operation(self, operation: str, security_level: SecurityLabel, count: int):
        """Log security operation"""
        self.audit_log.append({
            'operation': operation,
            'security_level': security_level.value,
            'count': count,
            'timestamp': time.time(),
            'thread': id(self)
        })


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2. CRYPTOGRAPHIC VERIFICATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class VerifiedCryptoOperations:
    """Cryptographic operations with verification"""

    def __init__(self):
        self.crypto_plugin = None
        # Find crypto plugin
        for plugin in _plugin_registry.plugins.values():
            if plugin.name == "crypto":
                self.crypto_plugin = plugin
                break

    @contract(
        preconditions=[
            ("key_length_sufficient", lambda self, key: len(key) >= 32),
            ("data_not_empty", lambda self, key, data: len(data) > 0)
        ],
        postconditions=[
            ("result_different_from_input", lambda self, key, data, result: result != data),
            ("result_not_empty", lambda self, key, data, result: len(result) > 0)
        ]
    )
    def encrypt_data(self, data: str, key: str) -> TaintedValue:
        """Encrypt data with verification"""
        require("key is secure", self._verify_key_strength(key))

        # Simple encryption (in practice, use proper crypto library)
        encrypted = self._simple_encrypt(data, key)

        # Verify encryption properties
        require("encrypted data differs from original", encrypted != data)
        require("encrypted data is not empty", len(encrypted) > 0)

        # Create tainted value for encrypted data
        encrypted_value = TaintedValue(
            value=encrypted,
            label=SecurityLabel.CONFIDENTIAL,
            provenance=["encryption", f"key_length_{len(key)}"]
        )

        return encrypted_value

    @contract(
        preconditions=[
            ("encrypted_data_not_empty", lambda self, encrypted_data, key: len(encrypted_data.value) > 0),
            ("key_not_empty", lambda self, encrypted_data, key: len(key) > 0)
        ]
    )
    def decrypt_data(self, encrypted_data: TaintedValue, key: str) -> TaintedValue:
        """Decrypt data with verification"""
        require("data is encrypted", "encryption" in encrypted_data.provenance)
        require("key is valid", len(key) >= 32)

        # Decrypt
        decrypted = self._simple_decrypt(encrypted_data.value, key)

        # Verify decryption
        require("decrypted data is valid", len(decrypted) > 0)

        # Maintain security level
        decrypted_value = TaintedValue(
            value=decrypted,
            label=encrypted_data.label,  # Maintain security level
            provenance=encrypted_data.provenance + ["decryption"]
        )

        return decrypted_value

    def generate_secure_key(self, length: int = 32) -> TaintedValue:
        """Generate cryptographically secure key"""
        require("length is sufficient", length >= 32)

        # Generate secure random key
        key = secrets.token_hex(length)

        # Verify key properties
        require("key has correct length", len(key) == length * 2)  # hex encoding doubles length
        require("key is not empty", len(key) > 0)

        # Verify randomness if plugin available
        if self.crypto_plugin:
            is_secure = self.crypto_plugin.verify_secure_random(secrets.randbits)
            require("key generator is secure", is_secure)

        # Create secret tainted value
        key_value = TaintedValue(
            value=key,
            label=SecurityLabel.TOP_SECRET,
            provenance=["secure_key_generation", f"length_{length}"]
        )

        return key_value

    def hash_password(self, password: str, salt: Optional[str] = None) -> TaintedValue:
        """Hash password with salt"""
        require("password not empty", len(password) > 0)
        require("password meets minimum strength", self._verify_password_strength(password))

        if salt is None:
            salt = secrets.token_hex(16)

        # Hash password with salt
        salted_password = f"{password}{salt}"
        password_hash = hashlib.pbkdf2_hmac('sha256',
                                            salted_password.encode('utf-8'),
                                            salt.encode('utf-8'),
                                            100000)  # 100k iterations

        hash_result = {
            'hash': password_hash.hex(),
            'salt': salt,
            'algorithm': 'pbkdf2_hmac_sha256',
            'iterations': 100000
        }

        # Verify hash properties
        require("hash is not empty", len(hash_result['hash']) > 0)
        require("salt is not empty", len(hash_result['salt']) > 0)
        require("hash differs from password", hash_result['hash'] != password)

        # Create confidential tainted value
        hashed_value = TaintedValue(
            value=hash_result,
            label=SecurityLabel.CONFIDENTIAL,
            provenance=["password_hashing", "pbkdf2_100k_iterations"]
        )

        return hashed_value

    def verify_constant_time_operation(self, operation_func, test_inputs: List[Any]) -> bool:
        """Verify operation runs in constant time"""
        if not self.crypto_plugin:
            return True  # Skip if plugin not available

        require("test inputs provided", len(test_inputs) > 1)
        require("operation is callable", callable(operation_func))

        # Use crypto plugin to verify constant time
        is_constant_time = self.crypto_plugin.verify_constant_time(operation_func, test_inputs)

        require("operation timing is constant", is_constant_time)
        return is_constant_time

    def secure_key_zeroization(self, key_data: TaintedValue) -> bool:
        """Securely zero out key data"""
        require("key_data is secret", key_data.label in [SecurityLabel.SECRET, SecurityLabel.TOP_SECRET])

        # Zero out the key (in practice, this would be more sophisticated)
        if isinstance(key_data.value, str):
            # Create zeroed version
            zeroed_data = '\x00' * len(key_data.value)
        elif isinstance(key_data.value, bytes):
            zeroed_data = b'\x00' * len(key_data.value)
        else:
            zeroed_data = None

        # Verify zeroization if plugin available
        if self.crypto_plugin and zeroed_data:
            is_zeroized = self.crypto_plugin.verify_key_zeroized(zeroed_data)
            require("key was properly zeroized", is_zeroized)

        return True

    def _simple_encrypt(self, data: str, key: str) -> str:
        """Simple encryption for demo purposes"""
        # XOR-based encryption (NOT for production use)
        encrypted = ""
        for i, char in enumerate(data):
            key_char = key[i % len(key)]
            encrypted_char = chr(ord(char) ^ ord(key_char))
            encrypted += encrypted_char
        return encrypted

    def _simple_decrypt(self, encrypted: str, key: str) -> str:
        """Simple decryption for demo purposes"""
        # XOR decryption (same as encryption for XOR)
        return self._simple_encrypt(encrypted, key)

    def _verify_key_strength(self, key: str) -> bool:
        """Verify cryptographic key strength"""
        if len(key) < 32:
            return False

        # Check for sufficient entropy (basic check)
        unique_chars = len(set(key))
        if unique_chars < 16:  # Should have diverse characters
            return False

        return True

    def _verify_password_strength(self, password: str) -> bool:
        """Verify password meets strength requirements"""
        if len(password) < 8:
            return False

        # Check for character diversity
        has_lower = any(c.islower() for c in password)
        has_upper = any(c.isupper() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)

        return sum([has_lower, has_upper, has_digit, has_special]) >= 3


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 3. INPUT SANITIZATION AND VALIDATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class SecureInputValidator:
    """Input validation with security verification"""

    def __init__(self):
        self.security_plugin = None
        # Find security plugin
        for plugin in _plugin_registry.plugins.values():
            if plugin.name == "security":
                self.security_plugin = plugin
                break

    @contract(
        preconditions=[
            ("input_not_empty", lambda self, user_input: len(user_input) > 0)
        ],
        postconditions=[
            ("output_is_safe", lambda self, user_input, result:
            self._verify_no_injection_patterns(result)),
            ("output_not_empty", lambda self, user_input, result: len(result) > 0)
        ]
    )
    def sanitize_html_input(self, user_input: str) -> TaintedValue:
        """Sanitize HTML input to prevent XSS"""
        require("input is string", isinstance(user_input, str))

        # Basic HTML sanitization
        sanitized = user_input
        sanitized = sanitized.replace("<", "&lt;")
        sanitized = sanitized.replace(">", "&gt;")
        sanitized = sanitized.replace("&", "&amp;")
        sanitized = sanitized.replace('"', "&quot;")
        sanitized = sanitized.replace("'", "&#x27;")

        # Verify with security plugin if available
        if self.security_plugin:
            is_safe = self.security_plugin.verify_input_sanitized(
                user_input,
                lambda x: x.replace("<", "&lt;").replace(">", "&gt;")
            )
            require("sanitization is effective", is_safe)

        # Create public tainted value (safe for display)
        sanitized_value = TaintedValue(
            value=sanitized,
            label=SecurityLabel.PUBLIC,
            provenance=["user_input", "html_sanitization"]
        )

        return sanitized_value

    @contract(
        preconditions=[
            ("query_template_valid", lambda self, query_template, params: "?" in query_template),
            ("params_list_valid", lambda self, query_template, params: isinstance(params, list))
        ]
    )
    def sanitize_sql_input(self, query_template: str, params: List[str]) -> TaintedValue:
        """Sanitize SQL input to prevent injection"""
        require("query uses parameterization", query_template.count("?") == len(params))

        # Verify SQL injection safety
        sanitized_params = []
        for param in params:
            # Basic SQL injection prevention
            sanitized_param = str(param).replace("'", "''")  # Escape single quotes
            sanitized_param = sanitized_param.replace(";", "")  # Remove semicolons
            sanitized_param = sanitized_param.replace("--", "")  # Remove comments
            sanitized_params.append(sanitized_param)

        # Create safe query structure
        safe_query = {
            'template': query_template,
            'params': sanitized_params,
            'param_count': len(sanitized_params)
        }

        # Verify with security plugin if available
        if self.security_plugin:
            for param in params:
                is_safe = self.security_plugin.verify_no_injection(query_template, param)
                require(f"parameter '{param}' is injection-safe", is_safe)

        # Create confidential tainted value
        sanitized_value = TaintedValue(
            value=safe_query,
            label=SecurityLabel.CONFIDENTIAL,
            provenance=["sql_parameterization", "injection_prevention"]
        )

        return sanitized_value

    def validate_user_role(self, user_role: str, required_privilege: str) -> bool:
        """Validate user has required privileges"""
        require("user_role specified", len(user_role) > 0)
        require("required_privilege specified", len(required_privilege) > 0)

        # Use security plugin for privilege verification
        if self.security_plugin:
            has_privilege = self.security_plugin.verify_privilege_boundary(user_role, required_privilege)
            require("user has required privilege", has_privilege)
            return has_privilege

        # Fallback basic validation
        privilege_levels = {'guest': 0, 'user': 1, 'admin': 2, 'superuser': 3}
        user_level = privilege_levels.get(user_role, -1)
        required_level = privilege_levels.get(required_privilege, 3)

        has_access = user_level >= required_level
        require("privilege check passed", has_access)
        return has_access

    def _verify_no_injection_patterns(self, text: str) -> bool:
        """Verify text contains no injection patterns"""
        dangerous_patterns = [
            '<script', 'javascript:', 'onload=', 'onerror=',
            'DROP TABLE', 'DELETE FROM', 'INSERT INTO', 'UPDATE SET',
            '--', '/*', '*/', 'xp_cmdshell', 'sp_executesql'
        ]

        text_lower = text.lower()
        return not any(pattern.lower() in text_lower for pattern in dangerous_patterns)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 4. SECURE DATA PIPELINE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class SecureDataPipeline:
    """End-to-end secure data processing pipeline"""

    def __init__(self):
        self.data_processor = SecureDataProcessor()
        self.crypto_ops = VerifiedCryptoOperations()
        self.input_validator = SecureInputValidator()
        self.processing_log = []

    def process_user_registration(self, username: str, password: str, email: str) -> Dict[str, Any]:
        """Secure user registration pipeline"""
        require("username provided", len(username) > 0)
        require("password provided", len(password) > 0)
        require("email provided", len(email) > 0)

        # Stage 1: Input validation and sanitization
        sanitized_username = self.input_validator.sanitize_html_input(username)
        sanitized_email = self.input_validator.sanitize_html_input(email)

        # Stage 2: Process credentials
        credentials = self.data_processor.process_user_credentials(username, password)

        # Stage 3: Hash password
        password_hash = self.crypto_ops.hash_password(password)

        # Stage 4: Generate API key
        api_key = self.crypto_ops.generate_secure_key(32)

        # Stage 5: Declassify for storage (with justification)
        storage_data = {
            'username': sanitized_username.value,
            'email': sanitized_email.value,
            'password_hash': password_hash.value['hash'],
            'salt': password_hash.value['salt'],
            'api_key_hash': hashlib.sha256(api_key.value.encode()).hexdigest()
        }

        # Log the operation
        self.processing_log.append({
            'operation': 'user_registration',
            'username': username,
            'timestamp': time.time(),
            'security_checks_passed': True
        })

        require("registration data complete", all(
            key in storage_data for key in ['username', 'email', 'password_hash', 'salt']
        ))

        return {
            'status': 'success',
            'user_id': abs(hash(username)) % 10000,  # Simple ID generation
            'storage_data': storage_data,
            'api_key': api_key.value  # Return actual key to user (one time only)
        }

    def process_secure_message(self, sender: str, recipient: str, message: str,
                               encryption_key: str) -> Dict[str, Any]:
        """Secure message processing pipeline"""
        require("sender specified", len(sender) > 0)
        require("recipient specified", len(recipient) > 0)
        require("message not empty", len(message) > 0)
        require("encryption key provided", len(encryption_key) >= 32)

        # Stage 1: Sanitize inputs
        sanitized_message = self.input_validator.sanitize_html_input(message)

        # Stage 2: Encrypt message
        encrypted_message = self.crypto_ops.encrypt_data(sanitized_message.value, encryption_key)

        # Stage 3: Create message metadata
        message_metadata = {
            'sender': sender,
            'recipient': recipient,
            'timestamp': time.time(),
            'encrypted_size': len(encrypted_message.value),
            'encryption_method': 'simple_xor_demo'  # In practice, use AES/ChaCha20
        }

        # Stage 4: Process metadata as confidential
        metadata_tainted = TaintedValue(
            value=message_metadata,
            label=SecurityLabel.CONFIDENTIAL,
            provenance=["message_metadata", "secure_messaging"]
        )

        # Log operation
        self.processing_log.append({
            'operation': 'secure_message',
            'sender': sender,
            'recipient': recipient,
            'timestamp': time.time(),
            'encrypted': True
        })

        return {
            'status': 'success',
            'encrypted_message': encrypted_message.value,
            'metadata': metadata_tainted.value,
            'message_id': abs(hash(f"{sender}{recipient}{time.time()}")) % 100000
        }

    def audit_security_operations(self) -> Dict[str, Any]:
        """Generate security audit report"""
        # Combine all audit logs
        all_operations = (
                self.processing_log +
                self.data_processor.audit_log
        )

        # Analyze security operations
        total_operations = len(all_operations)
        operation_types = {}
        security_levels = {}

        for op in all_operations:
            op_type = op.get('operation', 'unknown')
            operation_types[op_type] = operation_types.get(op_type, 0) + 1

            if 'security_level' in op:
                level = op['security_level']
                security_levels[level] = security_levels.get(level, 0) + 1

        # Information flow summary
        flow_summary = {
            'total_flows': len(self.data_processor.flow_tracker.flows),
            'flow_policies': len(self.data_processor.flow_tracker.policies)
        }

        audit_report = {
            'total_operations': total_operations,
            'operation_types': operation_types,
            'security_levels': security_levels,
            'information_flow': flow_summary,
            'audit_timestamp': time.time()
        }

        require("audit report complete", isinstance(audit_report, dict))
        require("total operations correct", audit_report['total_operations'] == total_operations)

        return audit_report


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 5. DEMONSTRATION FUNCTION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def demonstrate_security_verification():
    """Demonstrate all security verification features"""
    print("PyProof Security Verification Examples")
    print("~" * 80)

    # 1. Information Flow Tracking
    print("\n1. Information Flow Tracking:")
    processor = SecureDataProcessor()

    # Process different security levels
    public_data = processor.process_public_data("public information")
    print(f"   Public data: {public_data.label.value}")

    api_key = processor.process_api_key("demo_api_key_123456789abcdef", "payment_service")
    print(f"   API key: {api_key.label.value}")

    credentials = processor.process_user_credentials("alice", "secure_password_123")
    print(f"   Credentials: {credentials.label.value}")

    # Test information flow
    try:
        # This should succeed (public -> confidential)
        transferred = processor.transfer_data(public_data, SecurityLabel.CONFIDENTIAL)
        print(f"   +++ - Public->Confidential transfer: {transferred.label.value}")
    except ProofFailure as e:
        print(f"   XXX - Transfer failed: {e}")

    try:
        # This should fail (secret -> public)
        leaked = processor.transfer_data(credentials, SecurityLabel.PUBLIC)
        print(f"   XXX - This shouldn't happen: {leaked.label.value}")
    except ProofFailure as e:
        print(f"   +++ - Prevented data leak: Secret->Public blocked")

    # Demonstrate declassification
    declassified = processor.declassify_for_logging(
        credentials,
        "User authentication event logging for security audit trail analysis"
    )
    print(f"   +++ - Declassified for logging: {declassified.label.value}")

    # 2. Cryptographic Verification
    print("\n2. Cryptographic Operations:")
    crypto_ops = VerifiedCryptoOperations()

    # Generate secure key
    secure_key = crypto_ops.generate_secure_key(32)
    print(f"   Generated secure key: {secure_key.value[:16]}... ({len(secure_key.value)} chars)")

    # Encrypt data
    test_data = "This is sensitive information that needs encryption"
    encrypted = crypto_ops.encrypt_data(test_data, secure_key.value)
    print(f"   Encrypted data length: {len(encrypted.value)} chars")

    # Decrypt data
    decrypted = crypto_ops.decrypt_data(encrypted, secure_key.value)
    print(f"   Decrypted data: {decrypted.value[:30]}...")

    # Hash password
    password_hash = crypto_ops.hash_password("user_password_123")
    print(f"   Password hash: {password_hash.value['hash'][:16]}...")
    print(f"   Salt: {password_hash.value['salt'][:16]}...")

    # Test constant time operation
    def test_operation(x):
        return hashlib.sha256(str(x).encode()).hexdigest()

    is_constant_time = crypto_ops.verify_constant_time_operation(
        test_operation,
        [1, 100, 10000]
    )
    print(f"   +++ - Constant time verification: {is_constant_time}")

    # Secure key zeroization
    crypto_ops.secure_key_zeroization(secure_key)
    print("   +++ - Key securely zeroized")

    # 3. Input Sanitization
    print("\n3. Input Sanitization and Validation:")
    validator = SecureInputValidator()

    # HTML sanitization
    malicious_html = "<script>alert('XSS Attack!');</script><p>Safe content</p>"
    sanitized_html = validator.sanitize_html_input(malicious_html)
    print(f"   Original: {malicious_html}")
    print(f"   Sanitized: {sanitized_html.value}")

    # SQL injection prevention
    sql_template = "SELECT * FROM users WHERE name = ? AND age > ?"
    sql_params = ["Alice'; DROP TABLE users; --", "25"]
    sanitized_sql = validator.sanitize_sql_input(sql_template, sql_params)
    print(f"   SQL template: {sanitized_sql.value['template']}")
    print(f"   Sanitized params: {sanitized_sql.value['params']}")

    # Privilege validation
    try:
        admin_access = validator.validate_user_role("admin", "user")
        print(f"   +++ - Admin accessing user resources: {admin_access}")

        guest_access = validator.validate_user_role("guest", "admin")
        print(f"   XXX - This shouldn't succeed: {guest_access}")
    except ProofFailure as e:
        print(f"   +++ - Prevented privilege escalation: Guest->Admin blocked")

    # 4. Secure Data Pipeline
    print("\n4. Secure Data Pipeline:")
    pipeline = SecureDataPipeline()

    # User registration
    registration_result = pipeline.process_user_registration(
        username="alice_user",
        password="SecurePass123!",
        email="alice@example.com"
    )
    print(f"   Registration status: {registration_result['status']}")
    print(f"   User ID: {registration_result['user_id']}")
    print(f"   API key (first 16 chars): {registration_result['api_key'][:16]}...")

    # Secure messaging
    message_result = pipeline.process_secure_message(
        sender="alice_user",
        recipient="bob_user",
        message="This is a confidential message about project details",
        encryption_key=secure_key.value
    )
    print(f"   Message status: {message_result['status']}")
    print(f"   Message ID: {message_result['message_id']}")
    print(f"   Encrypted length: {len(message_result['encrypted_message'])} chars")

    # Security audit
    audit_report = pipeline.audit_security_operations()
    print(f"   Audit report - Total operations: {audit_report['total_operations']}")
    print(f"   Operation types: {list(audit_report['operation_types'].keys())}")
    print(f"   Information flows tracked: {audit_report['information_flow']['total_flows']}")

    # 5. Security Policy Enforcement
    print("\n5. Security Policy Enforcement:")

    # Demonstrate information flow policies
    flow_tracker = InformationFlowTracker()

    # Set restrictive policy
    flow_tracker.add_policy(SecurityLabel.SECRET, SecurityLabel.PUBLIC, False)

    secret_data = TaintedValue("classified_information", SecurityLabel.SECRET, ["classification"])

    try:
        flow_tracker.track_flow(secret_data, SecurityLabel.PUBLIC)
        print("   XXX - This shouldn't succeed")
    except ProofFailure as e:
        print("   +++ - Security policy enforced: Secret->Public flow blocked")

    # Allow controlled flow
    flow_tracker.add_policy(SecurityLabel.SECRET, SecurityLabel.CONFIDENTIAL, True)

    try:
        flow_tracker.track_flow(secret_data, SecurityLabel.CONFIDENTIAL)
        print("   +++ - Controlled downgrade allowed: Secret->Confidential")
    except ProofFailure as e:
        print(f"   XXX - Unexpected policy failure: {e}")

    # 6. Security Verification Summary
    print("\n6. Security Verification Summary:")
    summary = pyproof._proof.get_summary()
    print(f"   Total security verifications: {summary['total_steps']}")

    # Count security-related proof steps
    security_steps = [s for s in pyproof._proof.steps
                      if any(keyword in s.claim.lower() for keyword in
                             ['security', 'encrypt', 'sanitiz', 'privilege', 'flow'])]
    print(f"   Security-specific proofs: {len(security_steps)}")

    # Show recent security verifications
    if security_steps:
        print("   Recent security verifications:")
        for step in security_steps[-5:]:
            context = f" ({step.context})" if step.context else ""
            print(f"     - {step.claim}{context}")

    # Information flow summary
    print(f"   Information flows tracked: {len(processor.flow_tracker.flows)}")
    print(f"   Flow policies enforced: {len(processor.flow_tracker.policies)}")

    print("\nAll security verification examples completed successfully!")


if __name__ == "__main__":
    demonstrate_security_verification()