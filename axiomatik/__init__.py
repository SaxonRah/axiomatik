"""
Axiomatik: Performant Runtime Verification for Python
A comprehensive system with performance optimization, integration helpers, and domain-specific extensions.
"""


# Import core components
from .axiomatik import (
    ProofFailure,
    Config,
    Proof,
    GhostState,
    require,
    proof_context,
    RefinementType,
    RangeInt,
    NonEmptyList,
    ValidatedString,
    PositiveInt,
    Natural,
    Percentage,
    Protocol,
    protocol_method,
    TaintedValue,
    InformationFlowTracker,
    TemporalProperty,
    EventuallyProperty,
    AlwaysProperty,
    TemporalVerifier,
    record_temporal_event,
    add_temporal_property,
    verify_temporal_properties,
    get_temporal_history,
    Plugin,
    PluginRegistry,
    CryptoPlugin,
    FinancePlugin,
    SecurityPlugin,
    ConcurrencyPlugin,
    contract,
    auto_contract,
    gradually_verify,
    verification_mode,
    track_sensitive_data,
)

# Import CLI tool from axiomatikify.py
from .axiomatikify import cli as axiomatikify

# Import future features from future/future_axiomatik.py
from .future.future_axiomatik import (
    VerificationLevel,
    adaptive_require,
    adaptive_verification_context,
    auto_tune_verification_level,
    generate_performance_report,
    get_performance_hotspots,
    visualize_performance,
    contract_with_recovery,
    RecoveryPolicy,
    RecoveryStrategy,
    RecoveryManager,
    PropertyManager,
    PerformanceAnalyzer,
    VerificationHotspot,
    _performance_analyzer,
    _adaptive_monitor,
    _future_require,
)

# Set up the package version
__version__ = "0.1.3"
__author__ = "Robert Valentine"
__email__ = "paraboliclabs@gmail.com"
__url__ = "https://github.com/SaxonRah/axiomatik"

# Expose the CLI tool as a package-level function
def axiomatikify_cli():
    return axiomatikify()

# Define the package's public API
__all__ = [
    "ProofFailure",
    "Config",
    "Proof",
    "GhostState",
    "require",
    "proof_context",
    "RefinementType",
    "RangeInt",
    "NonEmptyList",
    "ValidatedString",
    "PositiveInt",
    "Natural",
    "Percentage",
    "Protocol",
    "protocol_method",
    "TaintedValue",
    "InformationFlowTracker",
    "TemporalProperty",
    "EventuallyProperty",
    "AlwaysProperty",
    "TemporalVerifier",
    "record_temporal_event",
    "add_temporal_property",
    "verify_temporal_properties",
    "get_temporal_history",
    "Plugin",
    "PluginRegistry",
    "CryptoPlugin",
    "FinancePlugin",
    "SecurityPlugin",
    "ConcurrencyPlugin",
    "contract",
    "auto_contract",
    "gradually_verify",
    "verification_mode",
    "track_sensitive_data",
    "axiomatikify_cli",
    "VerificationLevel",
    "adaptive_require",
    "adaptive_verification_context",
    "auto_tune_verification_level",
    "generate_performance_report",
    "get_performance_hotspots",
    "visualize_performance",
    "contract_with_recovery",
    "RecoveryPolicy",
    "RecoveryStrategy",
    "RecoveryManager",
    "PropertyManager",
    "PerformanceAnalyzer",
    "VerificationHotspot",
]