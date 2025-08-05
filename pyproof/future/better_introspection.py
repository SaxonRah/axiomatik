import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
import matplotlib.pyplot as plt  # Optional visualization

from pyproof.pyproof import VerificationLevel, _config, require


@dataclass
class VerificationHotspot:
    """Represents an expensive verification check"""
    property_name: str
    total_time: float
    call_count: int
    average_time: float
    percentage_of_total: float
    context: str


class PerformanceAnalyzer:
    """Analyzes verification performance and identifies hotspots"""

    def __init__(self):
        self.verification_times = defaultdict(list)
        self.context_times = defaultdict(list)
        self.total_verification_time = 0.0
        self.auto_tuning_enabled = False
        self.target_overhead_percent = 5.0

    def record_verification(self, property_name: str, context: str,
                            execution_time: float, verification_time: float):
        """Record performance data for a verification"""
        self.verification_times[property_name].append(verification_time)
        self.context_times[context].append(verification_time)
        self.total_verification_time += verification_time

        # Auto-tune if enabled
        if self.auto_tuning_enabled:
            self._check_auto_tune(execution_time, verification_time)

    def get_performance_hotspots(self, top_n: int = 10) -> List[VerificationHotspot]:
        """Identify the most expensive verification checks"""
        hotspots = []

        for prop_name, times in self.verification_times.items():
            total_time = sum(times)
            call_count = len(times)
            avg_time = total_time / call_count
            percentage = (total_time / self.total_verification_time) * 100

            # Find most common context for this property
            context = self._find_primary_context(prop_name)

            hotspots.append(VerificationHotspot(
                property_name=prop_name,
                total_time=total_time,
                call_count=call_count,
                average_time=avg_time,
                percentage_of_total=percentage,
                context=context
            ))

        # Sort by total time and return top N
        hotspots.sort(key=lambda x: x.total_time, reverse=True)
        return hotspots[:top_n]

    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        hotspots = self.get_performance_hotspots()

        report = ["PyProof Performance Analysis", "=" * 50, ""]

        # Overall statistics
        total_properties = len(self.verification_times)
        total_calls = sum(len(times) for times in self.verification_times.values())
        avg_verification_time = self.total_verification_time / max(1, total_calls)

        report.extend([
            f"Total properties verified: {total_properties}",
            f"Total verification calls: {total_calls}",
            f"Total verification time: {self.total_verification_time:.3f}s",
            f"Average per verification: {avg_verification_time * 1000:.3f}ms",
            ""
        ])

        # Top hotspots
        report.append("Top Performance Hotspots:")
        for i, hotspot in enumerate(hotspots, 1):
            report.append(
                f"  {i:2d}. {hotspot.property_name[:40]:40} "
                f"{hotspot.total_time * 1000:6.1f}ms "
                f"({hotspot.percentage_of_total:4.1f}%) "
                f"[{hotspot.call_count} calls]"
            )

        # Context analysis
        report.extend(["", "Performance by Context:"])
        context_totals = {
            ctx: sum(times) for ctx, times in self.context_times.items()
        }
        for context, total_time in sorted(context_totals.items(),
                                          key=lambda x: x[1], reverse=True):
            percentage = (total_time / self.total_verification_time) * 100
            report.append(f"  {context:20}: {total_time * 1000:6.1f}ms ({percentage:4.1f}%)")

        return "\n".join(report)

    def auto_tune_verification_level(self, target_overhead_percent: float = 5.0):
        """Automatically tune verification level based on measured overhead"""
        self.auto_tuning_enabled = True
        self.target_overhead_percent = target_overhead_percent

        print(f"Auto-tuning enabled: target overhead {target_overhead_percent}%")

    def _check_auto_tune(self, execution_time: float, verification_time: float):
        """Check if auto-tuning adjustment is needed"""
        if execution_time <= 0:
            return

        overhead_percent = (verification_time / execution_time) * 100

        if overhead_percent > self.target_overhead_percent * 1.5:
            # Reduce verification level
            self._reduce_verification_intensity()
        elif overhead_percent < self.target_overhead_percent * 0.5:
            # Can increase verification level
            self._increase_verification_intensity()

    def _reduce_verification_intensity(self):
        """Reduce verification intensity to meet performance targets"""
        current_level = _config.level

        if current_level == VerificationLevel.DEBUG:
            _config.level = VerificationLevel.FULL
        elif current_level == VerificationLevel.FULL:
            _config.level = VerificationLevel.INVARIANTS
        elif current_level == VerificationLevel.INVARIANTS:
            _config.level = VerificationLevel.CONTRACTS

        print(f"Auto-tune: Reduced verification level to {_config.level.value}")

    def _increase_verification_intensity(self):
        """Increase verification intensity when performance allows"""
        current_level = _config.level

        if current_level == VerificationLevel.CONTRACTS:
            _config.level = VerificationLevel.INVARIANTS
        elif current_level == VerificationLevel.INVARIANTS:
            _config.level = VerificationLevel.FULL
        elif current_level == VerificationLevel.FULL:
            _config.level = VerificationLevel.DEBUG

        print(f"Auto-tune: Increased verification level to {_config.level.value}")

    def visualize_hotspots(self, save_path: str = None):
        """Create visualization of performance hotspots"""
        try:
            hotspots = self.get_performance_hotspots()

            if not hotspots:
                print("No performance data available for visualization")
                return

            # Create bar chart of top hotspots
            names = [h.property_name[:20] for h in hotspots[:10]]
            times = [h.total_time * 1000 for h in hotspots[:10]]  # Convert to ms

            plt.figure(figsize=(12, 6))
            plt.bar(names, times)
            plt.title("Top 10 Verification Performance Hotspots")
            plt.xlabel("Property Name")
            plt.ylabel("Total Time (ms)")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()

        except ImportError:
            print("Matplotlib not available for visualization")


# Global performance analyzer
_performance_analyzer = PerformanceAnalyzer()


# Enhanced global functions
def get_performance_hotspots(top_n: int = 10) -> List[VerificationHotspot]:
    """Get the most expensive verification checks"""
    return _performance_analyzer.get_performance_hotspots(top_n)


def auto_tune_verification_level(target_overhead_percent: float = 5.0):
    """Automatically tune verification level based on measured overhead"""
    _performance_analyzer.auto_tune_verification_level(target_overhead_percent)


def generate_performance_report() -> str:
    """Generate comprehensive performance report"""
    return _performance_analyzer.generate_performance_report()


def visualize_performance(save_path: str = None):
    """Visualize verification performance"""
    _performance_analyzer.visualize_hotspots(save_path)


# Integration with existing require function
def _enhanced_require(claim: str, evidence: Any, context: str = "") -> Any:
    """Enhanced require with performance tracking"""
    start_time = time.perf_counter()

    try:
        result = require(claim, evidence)
        return result
    finally:
        verification_time = time.perf_counter() - start_time
        _performance_analyzer.record_verification(
            property_name=claim,
            context=context or "global",
            execution_time=0.001,  # Placeholder - would need actual execution timing
            verification_time=verification_time
        )


# Usage examples:
if __name__ == "__main__":
    # Enable auto-tuning for 5% overhead target
    auto_tune_verification_level(target_overhead_percent=5.0)

    # Run some verification-heavy code...

    # Analyze performance
    print(generate_performance_report())

    # Show top hotspots
    hotspots = get_performance_hotspots(5)
    for hotspot in hotspots:
        print(f"Expensive: {hotspot.property_name} - {hotspot.average_time * 1000:.2f}ms avg")

    # Visualize (if matplotlib available)
    visualize_performance("verification_hotspots.png")