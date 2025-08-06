"""
Axiomatik: Performant Runtime Verification for Python
A comprehensive system with performance optimization, integration helpers, domain-specific extensions,
adaptive monitoring, performance introspection, recovery framework, and user-friendly interface.

This package provides three levels of API:
1. Core API (axiomatik.py) - Full-featured verification system
2. Future API (future_axiomatik.py) - Advanced features like adaptive monitoring and recovery
3. Simple API (simple_axiomatik.py) - User-friendly interface for easy adoption
"""

# Import from core axiomatik module
from .axiomatik import (
    # Core system
    ProofFailure,
    Config,
    Proof,
    GhostState,
    require,
    proof_context,

    # Refinement types
    RefinementType,
    RangeInt,
    NonEmptyList as CoreNonEmptyList,  # Renamed to avoid conflict
    ValidatedString,
    PositiveInt as CorePositiveInt,    # Renamed to avoid conflict
    Natural,
    Percentage as CorePercentage,      # Renamed to avoid conflict

    # Protocol verification
    Protocol,
    ProtocolState,
    protocol_method,

    # Information flow tracking
    TaintedValue,
    InformationFlowTracker,
    SecurityLabel,
    track_sensitive_data,

    # Temporal properties
    TemporalProperty,
    EventuallyProperty,
    AlwaysProperty,
    TemporalVerifier,
    record_temporal_event,
    add_temporal_property,
    verify_temporal_properties,
    get_temporal_history,

    # Plugin system
    Plugin,
    PluginRegistry,
    CryptoPlugin,
    FinancePlugin,
    SecurityPlugin,
    ConcurrencyPlugin,

    # Function contracts
    contract,
    auto_contract,
    gradually_verify,
    verification_mode as core_verification_mode,  # Renamed to avoid conflict

    # Predefined protocols
    filemanager_protocol,
    statemachine_protocol,
    dbconnection_protocol,
    httpclient_protocol,

    # Global instances (for advanced users)
    _plugin_registry,
    _proof,
    _ghost,
    _config,
    _temporal_verifier,

    # Verification levels
    VerificationLevel,
)

# Import from future axiomatik module (advanced features)
from future.future_axiomatik import (
    # Adaptive monitoring
    AdaptiveMonitor,
    PropertyManager,
    adaptive_require,
    adaptive_verification_context,

    # Performance introspection
    PerformanceAnalyzer,
    VerificationHotspot,
    auto_tune_verification_level,
    generate_performance_report,
    get_performance_hotspots,
    visualize_performance,

    # Recovery framework
    RecoveryPolicy,
    RecoveryStrategy,
    RecoveryManager,
    contract_with_recovery,

    # Global instances
    _performance_analyzer,
    _adaptive_monitor,
    _future_require,
)

# Import from simple axiomatik module (user-friendly interface)
from simple.simple_axiomatik import (
    # User-friendly errors
    VerificationError,

    # Simple decorators
    verify,
    checked,

    # Type aliases (these are the recommended ones for users)
    Positive,
    NonEmpty,
    Range,
    PositiveInt,      # Simple version (recommended)
    PositiveFloat,
    NonEmptyList,     # Simple version (recommended)
    NonEmptyStr,
    Percentage,       # Simple version (recommended)

    # Protocol verification decorators
    stateful,
    state,

    # Integration helpers
    enable_for_dataclass,
    expect_approximately,
    approx_equal,

    # Context managers
    verification_context,
    production_mode,
    no_verification,

    # Configuration
    set_mode,
    get_mode,
    report,
    performance_report,
    clear_performance_data,

    # Convenience aliases
    req,              # Alias for require
    ensure,           # Alias for require (for postconditions)

    # Simple config
    _config as _simple_config,
)

# Import CLI tool
try:
    from .axiomatikify import cli as axiomatikify
except ImportError:
    # Handle case where libcst is not available
    axiomatikify = None

# Package metadata
__version__ = "0.2.0"
__author__ = "Robert Valentine"
__email__ = "paraboliclabs@gmail.com"
__url__ = "https://github.com/SaxonRah/axiomatik"

# Convenience function for CLI
def axiomatikify_cli():
    """Run the axiomatikify code transformation tool"""
    if axiomatikify is None:
        raise ImportError("axiomatikify requires libcst. Install with: pip install libcst")
    return axiomatikify()

# Demo functions from each module
from .axiomatik import demo_advanced_features
from simple.simple_axiomatik import demo as demo_simple_features
try:
    from future.future_axiomatik import demo_future_features
except ImportError:
    demo_future_features = None

def demo_all_features():
    """Run all demo functions"""
    print("=== Simple Axiomatik Demo ===")
    demo_simple_features()

    print("\n\n=== Advanced Axiomatik Demo ===")
    demo_advanced_features()

    if demo_future_features:
        print("\n\n=== Future Axiomatik Demo ===")
        demo_future_features()
    else:
        print("\n\n=== Future Axiomatik Demo: Not Available ===")
        print("Some dependencies may be missing")

# Define the comprehensive public API
__all__ = [
    # === CORE API (axiomatik.py) ===

    # Core system
    "ProofFailure",
    "Config",
    "Proof",
    "GhostState",
    "require",
    "proof_context",
    "VerificationLevel",

    # Refinement types (core versions)
    "RefinementType",
    "RangeInt",
    "CoreNonEmptyList",
    "ValidatedString",
    "CorePositiveInt",
    "Natural",
    "CorePercentage",

    # Protocol verification
    "Protocol",
    "ProtocolState",
    "protocol_method",
    "filemanager_protocol",
    "statemachine_protocol",
    "dbconnection_protocol",
    "httpclient_protocol",

    # Information flow tracking
    "TaintedValue",
    "InformationFlowTracker",
    "SecurityLabel",
    "track_sensitive_data",

    # Temporal properties
    "TemporalProperty",
    "EventuallyProperty",
    "AlwaysProperty",
    "TemporalVerifier",
    "record_temporal_event",
    "add_temporal_property",
    "verify_temporal_properties",
    "get_temporal_history",

    # Plugin system
    "Plugin",
    "PluginRegistry",
    "CryptoPlugin",
    "FinancePlugin",
    "SecurityPlugin",
    "ConcurrencyPlugin",

    # Function contracts
    "contract",
    "auto_contract",
    "gradually_verify",
    "core_verification_mode",

    # === FUTURE API (future_axiomatik.py) ===

    # Adaptive monitoring
    "AdaptiveMonitor",
    "PropertyManager",
    "adaptive_require",
    "adaptive_verification_context",

    # Performance introspection
    "PerformanceAnalyzer",
    "VerificationHotspot",
    "auto_tune_verification_level",
    "generate_performance_report",
    "get_performance_hotspots",
    "visualize_performance",

    # Recovery framework
    "RecoveryPolicy",
    "RecoveryStrategy",
    "RecoveryManager",
    "contract_with_recovery",

    # === SIMPLE API (simple_axiomatik.py) - RECOMMENDED FOR MOST USERS ===

    # User-friendly errors
    "VerificationError",

    # Simple decorators (RECOMMENDED)
    "verify",
    "checked",

    # Type aliases (RECOMMENDED - these are easier to use)
    "Positive",
    "NonEmpty",
    "Range",
    "PositiveInt",          # Simple version - use this one
    "PositiveFloat",
    "NonEmptyList",         # Simple version - use this one
    "NonEmptyStr",
    "Percentage",           # Simple version - use this one

    # Protocol verification (RECOMMENDED)
    "stateful",
    "state",

    # Integration helpers
    "enable_for_dataclass",
    "expect_approximately",
    "approx_equal",

    # Context managers (RECOMMENDED)
    "verification_context",
    "production_mode",
    "no_verification",

    # Configuration (RECOMMENDED)
    "set_mode",
    "get_mode",
    "report",
    "performance_report",
    "clear_performance_data",

    # Convenience aliases
    "req",                  # Alias for require
    "ensure",              # For postconditions

    # === TOOLS ===

    # CLI tool
    "axiomatikify_cli",

    # Demo functions
    "demo_simple_features",
    "demo_advanced_features",
    "demo_future_features",
    "demo_all_features",
]

# Provide guidance for new users
def get_started():
    """Print getting started guide for new users"""
    print("""
Axiomatik Getting Started Guide
==============================

For new users, start with the Simple API:

1. Basic verification:
   import axiomatik as ax
   
   @ax.verify
   def safe_divide(a: float, b: float) -> float:
       ax.require(b != 0, "Cannot divide by zero")
       return a / b

2. Type-based verification:
   @ax.checked  
   def process(items: ax.NonEmpty[list]) -> ax.PositiveInt:
       return len(items)

3. Set verification mode:
   ax.set_mode("dev")    # Development
   ax.set_mode("prod")   # Production  
   ax.set_mode("test")   # Testing
   ax.set_mode("off")    # Disabled

4. Run demos:
   ax.demo_simple_features()    # Start here
   ax.demo_advanced_features()  # Advanced patterns
   ax.demo_future_features()    # Cutting edge features

For complete examples: see simple_quick.py and simple_usage.py
For advanced features: see axiomatik.py examples  
For future features: see future_axiomatik.py examples
""")

# Make the simple API the default recommendation
# Users should generally import axiomatik and use the simple API
# Power users can access core and future APIs when needed