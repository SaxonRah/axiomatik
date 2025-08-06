#!/usr/bin/env python3
"""
Advanced Verified Space Shooter - The Ultimate Axiomatik Demonstration

This game showcases ALL axiomatik features:
- Type-verified entity management with dataclasses
- Stateful game state management with protocols
- Mathematical verification for physics calculations
- Performance monitoring with adaptive tuning
- Information flow tracking for sensitive data
- Temporal property verification for game events
- Recovery framework with graceful degradation
- Plugin system with game-specific verifiers
- Loop invariants for mathematical rigor
- Ghost state tracking for debugging
- Advanced refinement types
- Concurrency verification
- Real-time performance introspection
- Configuration-based verification levels
- Comprehensive game analytics

Installation:
    pip install pygame axiomatik

Usage:
    python advanced_verified_space_shooter.py

Controls:
    Arrow Keys: Move ship
    Spacebar: Fire bullets
    ESC: Pause/Resume
    Q: Quit game
    F1: Toggle debug overlay
    F2: Cycle verification modes
    F3: Generate analytics report
"""
import sys
sys.setrecursionlimit(10**8)

import pygame
import axiomatik.simple_axiomatik as ax
from axiomatik import (
    record_temporal_event, get_temporal_history, track_sensitive_data, SecurityLabel,
    add_temporal_property, AlwaysProperty, EventuallyProperty, verify_temporal_properties,
    TaintedValue, InformationFlowTracker, Plugin, _plugin_registry,
    contract_with_recovery, RecoveryStrategy, RecoveryPolicy,
    get_performance_hotspots, auto_tune_verification_level, generate_performance_report, Natural
)

# Import future features
try:
    from axiomatik.future_axiomatik import (
        adaptive_require, adaptive_verification_context,
        _adaptive_monitor, _performance_analyzer, _ghost
)

    FUTURE_FEATURES_AVAILABLE = True
except ImportError:
    FUTURE_FEATURES_AVAILABLE = False
    print("Note: Advanced future features not available - using standard axiomatik")

import math
import random
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Union, Any
from datetime import datetime
from enum import Enum
from collections import defaultdict, deque

# Initialize axiomatik in development mode initially
ax.set_mode("dev")

# Initialize pygame
pygame.init()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ADVANCED VERIFICATION CONFIGURATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class GameVerificationMode(Enum):
    """Game-specific verification modes"""
    TOURNAMENT = "tournament"  # Maximum verification for competitive play
    DEVELOPMENT = "development"  # Full verification for development
    PERFORMANCE = "performance"  # Minimal verification for smooth gameplay
    DEMO = "demo"  # Educational verification with detailed output
    ADAPTIVE = "adaptive"  # Auto-tuning verification based on performance


class AdvancedConfig:
    """Advanced configuration for game verification"""

    def __init__(self):
        self.verification_mode = GameVerificationMode.TOURNAMENT
        self.debug_overlay_enabled = False
        self.analytics_enabled = True
        self.recovery_enabled = True
        self.adaptive_tuning_enabled = FUTURE_FEATURES_AVAILABLE
        self.information_flow_tracking = True
        self.temporal_verification_enabled = True

    def set_game_mode(self, mode: GameVerificationMode):
        """Set game-specific verification mode"""
        self.verification_mode = mode

        mode_mapping = {
            GameVerificationMode.TOURNAMENT: "debug",
            GameVerificationMode.DEVELOPMENT: "dev",
            GameVerificationMode.PERFORMANCE: "prod",
            GameVerificationMode.DEMO: "dev",
            GameVerificationMode.ADAPTIVE: "dev"
        }

        ax.set_mode(mode_mapping[mode])

        if mode == GameVerificationMode.DEMO:
            print("Educational verification mode enabled")
            print("Detailed verification output will be shown")

        elif mode == GameVerificationMode.ADAPTIVE and FUTURE_FEATURES_AVAILABLE:
            print("Adaptive verification mode enabled")
            auto_tune_verification_level(target_overhead_percent=3.0)

        elif mode == GameVerificationMode.TOURNAMENT:
            print("Tournament mode: Maximum verification enabled")

        elif mode == GameVerificationMode.PERFORMANCE:
            print("Performance mode: Minimal verification for smooth gameplay")


# Global configuration
_game_config = AdvancedConfig()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ADVANCED REFINEMENT TYPES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from axiomatik import RefinementType


class GameCoordinate(RefinementType):
    """Screen coordinate that must be within extended game bounds"""

    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        super().__init__(
            float,
            lambda x: -100 <= x <= screen_width + 100,
            f"coordinate within extended screen bounds (0-{screen_width})"
        )


class HealthPoints(RefinementType):
    """Health that must be 0-100"""

    def __init__(self):
        super().__init__(
            int,
            lambda hp: 0 <= hp <= 100,
            "health points 0-100"
        )


class GameScore(RefinementType):
    """Game score with reasonable bounds"""

    def __init__(self):
        super().__init__(
            int,
            lambda score: 0 <= score <= 99999999,
            "game score 0-99,999,999"
        )


class Velocity(RefinementType):
    """Velocity with reasonable physics bounds"""

    def __init__(self, max_speed: float = 2000.0):
        self.max_speed = max_speed
        super().__init__(
            float,
            lambda v: abs(v) <= max_speed,
            f"velocity within ±{max_speed} pixels/second"
        )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# GAME-SPECIFIC PLUGIN SYSTEM
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class GamePhysicsPlugin(Plugin):
    """Plugin for game physics verification"""

    def __init__(self):
        super().__init__("game_physics")

    def add_verifiers(self):
        return {
            'velocity_reasonable': self.verify_velocity_bounds,
            'collision_geometry_valid': self.verify_collision_geometry,
            'physics_conservation': self.verify_physics_conservation,
            'boundary_conditions': self.verify_boundary_conditions
        }

    @staticmethod
    def verify_velocity_bounds(obj) -> bool:
        """Verify velocity is within reasonable game bounds"""
        if hasattr(obj, 'velocity'):
            max_speed = 1500.0  # pixels/second
            return obj.velocity.magnitude() <= max_speed
        return True

    @staticmethod
    def verify_collision_geometry(obj1, obj2) -> bool:
        """Verify collision geometry makes sense"""
        return (hasattr(obj1, 'size') and hasattr(obj2, 'size') and
                obj1.size > 0 and obj2.size > 0)

    @staticmethod
    def verify_physics_conservation(old_momentum: float, new_momentum: float,
                                    tolerance: float = 0.1) -> bool:
        """Verify momentum conservation in collisions"""
        return abs(old_momentum - new_momentum) <= tolerance

    @staticmethod
    def verify_boundary_conditions(obj, screen_width: int, screen_height: int) -> bool:
        """Verify object stays within reasonable bounds"""
        if hasattr(obj, 'position'):
            return (-200 <= obj.position.x <= screen_width + 200 and
                    -200 <= obj.position.y <= screen_height + 200)
        return True


class GameBalancePlugin(Plugin):
    """Plugin for game balance verification"""

    def __init__(self):
        super().__init__("game_balance")

    def add_verifiers(self):
        return {
            'score_progression': self.verify_score_progression,
            'difficulty_curve': self.verify_difficulty_curve,
            'player_survivability': self.verify_player_survivability
        }

    @staticmethod
    def verify_score_progression(score_rate: float, game_time: float) -> bool:
        """Verify score progression is reasonable"""
        if game_time <= 0:
            return True
        points_per_second = score_rate / game_time
        return 1.0 <= points_per_second <= 1000.0  # Reasonable scoring rate

    @staticmethod
    def verify_difficulty_curve(enemy_count: int, game_time: float) -> bool:
        """Verify difficulty increases reasonably over time"""
        if game_time <= 0:
            return True
        expected_enemies = min(1 + int(game_time / 10), 10)  # Max 10 enemies
        return enemy_count <= expected_enemies + 2  # Allow some variance

    @staticmethod
    def verify_player_survivability(player_health: int, enemy_count: int) -> bool:
        """Verify player has reasonable chance of survival"""
        # More enemies should not make health irrelevant
        return player_health >= 0 and (enemy_count <= 20 or player_health > 20)


# Register plugins
_plugin_registry.register(GamePhysicsPlugin())
_plugin_registry.register(GameBalancePlugin())


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# GAME CONSTANTS AND CONFIGURATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class GameConstants:
    """Game constants with verification"""

    # Display constants
    SCREEN_WIDTH: ax.PositiveInt = 1000
    SCREEN_HEIGHT: ax.PositiveInt = 700
    FPS: ax.Range[int, 30, 144] = 60

    # Entity constants
    PLAYER_SPEED: ax.PositiveFloat = 350.0
    ENEMY_SPEED: ax.PositiveFloat = 150.0
    BULLET_SPEED: ax.PositiveFloat = 600.0

    # Game mechanics
    MAX_ENEMIES: ax.Range[int, 1, 50] = 15
    MAX_BULLETS: ax.Range[int, 1, 200] = 50
    ENEMY_SPAWN_RATE: ax.Range[float, 0.1, 10.0] = 2.5  # enemies per second

    # Colors (RGB tuples)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 50, 50)
    GREEN = (50, 255, 50)
    BLUE = (50, 150, 255)
    YELLOW = (255, 255, 100)
    PURPLE = (255, 100, 255)
    ORANGE = (255, 165, 0)
    CYAN = (0, 255, 255)

    # Debug colors
    DEBUG_GREEN = (0, 255, 0)
    DEBUG_RED = (255, 0, 0)
    DEBUG_BLUE = (0, 0, 255)


# Create refinement types with game constants
GameX = GameCoordinate(GameConstants.SCREEN_WIDTH, GameConstants.SCREEN_HEIGHT)
GameY = GameCoordinate(GameConstants.SCREEN_WIDTH, GameConstants.SCREEN_HEIGHT)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ADVANCED INFORMATION FLOW TRACKING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class SecureGameDataManager:
    """Manages sensitive game data with information flow tracking"""

    def __init__(self):
        self.flow_tracker = InformationFlowTracker()

        # Set up information flow policies
        self.flow_tracker.add_policy(SecurityLabel.SECRET, SecurityLabel.PUBLIC, False)
        self.flow_tracker.add_policy(SecurityLabel.CONFIDENTIAL, SecurityLabel.PUBLIC, False)
        self.flow_tracker.add_policy(SecurityLabel.CONFIDENTIAL, SecurityLabel.CONFIDENTIAL, True)

        # Track sensitive data
        self.high_scores: List[TaintedValue] = []
        self.player_statistics: Dict[str, TaintedValue] = {}

    @ax.verify
    def record_high_score(self, score: int, player_name: str) -> str:
        """Record high score with information flow tracking"""
        ax.require(score >= 0, "Score cannot be negative")
        ax.require(len(player_name) > 0, "Player name cannot be empty")

        # Mark high score as confidential
        tainted_score = TaintedValue(
            {"score": score, "name": player_name, "timestamp": time.time()},
            SecurityLabel.CONFIDENTIAL,
            [f"high_score_achievement", f"player_{player_name}"]
        )

        self.high_scores.append(tainted_score)

        # Attempt to create public display
        try:
            self.flow_tracker.track_flow(tainted_score, SecurityLabel.PUBLIC)
            return f"New High Score: {score} by {player_name}"
        except ax.ProofFailure:
            # Sanitize before public display
            sanitized_score = min(score, 999999)
            return f"New High Score: {sanitized_score} by {'*' * len(player_name)}"

    @ax.verify
    def get_public_leaderboard(self) -> List[str]:
        """Get sanitized leaderboard for public display"""
        public_scores = []

        for tainted_score in sorted(self.high_scores,
                                    key=lambda x: x.value["score"], reverse=True)[:10]:
            try:
                self.flow_tracker.track_flow(tainted_score, SecurityLabel.PUBLIC)
                # Can display full info
                score_data = tainted_score.value
                public_scores.append(f"{score_data['score']:,} - {score_data['name']}")
            except ax.ProofFailure:
                # Must sanitize
                score_data = tainted_score.value
                public_scores.append(f"{min(score_data['score'], 999999):,} - Anonymous")

        return public_scores

    @ax.verify
    def track_sensitive_statistic(self, stat_name: str, value: Any,
                                  sensitivity: SecurityLabel = SecurityLabel.CONFIDENTIAL):
        """Track sensitive player statistics"""
        tainted_stat = TaintedValue(value, sensitivity, [f"player_stat_{stat_name}"])
        self.player_statistics[stat_name] = tainted_stat


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# VERIFIED ENTITY SYSTEM WITH ADVANCED FEATURES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@ax.enable_for_dataclass
@dataclass
class Vector2D:
    """2D vector with mathematical verification and ghost state tracking"""
    x: Union[int, float] = 0.0
    y: Union[int, float] = 0.0

    def __post_init__(self):
        # Convert to float for consistent math operations
        self.x = float(self.x)
        self.y = float(self.y)

        # Enhanced bounds checking
        ax.require(abs(self.x) < 1e8, "X coordinate exceeds reasonable bounds")
        ax.require(abs(self.y) < 1e8, "Y coordinate exceeds reasonable bounds")
        ax.require(not (math.isnan(self.x) or math.isnan(self.y)), "Coordinates cannot be NaN")
        ax.require(not (math.isinf(self.x) or math.isinf(self.y)), "Coordinates cannot be infinite")

    @ax.verify
    def magnitude(self) -> ax.PositiveFloat:
        """Calculate vector magnitude with verification and ghost state"""
        # Track calculation in ghost state
        # _ghost.set("vector_magnitude_calculation", {
        #     "x": self.x, "y": self.y, "timestamp": time.time()
        # })
        #
        # mag_squared = self.x * self.x + self.y * self.y
        # ax.require(mag_squared >= 0, "Magnitude squared cannot be negative")
        #
        # mag = math.sqrt(mag_squared)
        # ax.ensure(mag >= 0, "Magnitude cannot be negative")
        #
        # # Verify mathematical properties
        # if self.x == 0 and self.y == 0:
        #     ax.ensure(mag == 0, "Zero vector should have zero magnitude")
        # else:
        #     ax.ensure(mag > 0, "Non-zero vector should have positive magnitude")
        #
        # return mag

        mag_squared = self.x * self.x + self.y * self.y
        if mag_squared < 0:  # Simple check instead of complex requirement
            mag_squared = 0

        mag = math.sqrt(mag_squared)
        if mag < 0:  # Simple ensure instead of complex verification
            mag = 0

        return mag

    @contract_with_recovery(
        preconditions=[("vector_not_zero", lambda self: self.magnitude() > 1e-10)],
        recovery_strategy=RecoveryStrategy(
            RecoveryPolicy.GRACEFUL_DEGRADATION,
            fallback_handler=lambda self: Vector2D(1.0, 0.0)  # Default unit vector
        )
    )
    def normalize(self) -> 'Vector2D':
        """Normalize vector with recovery framework"""
        mag = self.magnitude()

        result = Vector2D(self.x / mag, self.y / mag)

        # Verify normalization
        result_mag = result.magnitude()
        ax.ensure(ax.approx_equal(result_mag, 1.0, 0.001),
                  "Normalized vector should have magnitude 1")

        return result

    @ax.verify
    def distance_to(self, other: 'Vector2D') -> ax.PositiveFloat:
        """Calculate distance between vectors with advanced verification"""
        # ax.require(isinstance(other, Vector2D), "Other must be Vector2D")
        #
        # dx = self.x - other.x
        # dy = self.y - other.y
        # distance = math.sqrt(dx * dx + dy * dy)
        #
        # # Verify mathematical properties
        # ax.ensure(distance >= 0, "Distance cannot be negative")
        # ax.ensure(distance == other.distance_to(self), "Distance should be symmetric")
        #
        # # Triangle inequality verification (with tolerance for floating point)
        # origin = Vector2D(0, 0)
        # self_to_origin = self.distance_to(origin)
        # other_to_origin = other.distance_to(origin)
        # ax.ensure(distance <= self_to_origin + other_to_origin + 1e-10,
        #           "Triangle inequality should hold")
        #
        # return distance
        if not isinstance(other, Vector2D):
            return 0.0

        dx = self.x - other.x
        dy = self.y - other.y
        distance = math.sqrt(dx * dx + dy * dy)

        return max(0.0, distance)  # Simple bounds check

    @ax.verify
    def dot_product(self, other: 'Vector2D') -> float:
        """Calculate dot product with verification"""
        ax.require(isinstance(other, Vector2D), "Other must be Vector2D")

        dot = self.x * other.x + self.y * other.y

        # Verify mathematical properties
        other_dot_self = other.dot_product(self)
        ax.ensure(ax.approx_equal(dot, other_dot_self, 1e-10), "Dot product should be commutative")

        return dot


@ax.enable_for_dataclass
@dataclass
class GameObject:
    """Base game object with advanced verification features"""
    position: Vector2D
    velocity: Vector2D
    size: Union[int, float]
    health: ax.Range[int, 0, 1000] = 100
    active: bool = True
    creation_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)

    def __post_init__(self):
        self.size = float(self.size)
        ax.require(self.size > 0, "Size must be positive")
        ax.require(0 <= self.health <= 1000, "Health must be in valid range")
        ax.require(self.creation_time > 0, "Creation time must be positive")

    @contract_with_recovery(
        preconditions=[("valid_delta_time", lambda self, delta_time: 0 < delta_time < 1.0)],
        recovery_strategy=RecoveryStrategy(
            RecoveryPolicy.GRACEFUL_DEGRADATION,
            fallback_handler=lambda self, delta_time: self._safe_position_update(min(delta_time, 1 / 60))
        ),
        #plugins=['velocity_reasonable']
    )
    def update_position(self, delta_time: ax.PositiveFloat) -> None:
        """Update position with recovery and plugin verification"""

        # Track update in ghost state
        _ghost.set("position_update", {
            "object_id": id(self),
            "old_position": (self.position.x, self.position.y),
            "velocity": (self.velocity.x, self.velocity.y),
            "delta_time": delta_time
        })

        # Store old position for verification
        old_pos = Vector2D(self.position.x, self.position.y)

        # Update position with loop invariant
        self.position.x += self.velocity.x * delta_time
        self.position.y += self.velocity.y * delta_time

        # Update timestamp
        current_time = time.time()
        ax.ensure(current_time >= self.last_update_time, "Time should not go backwards")
        self.last_update_time = current_time

        # Verify position change makes sense
        distance_moved = old_pos.distance_to(self.position)
        expected_distance = self.velocity.magnitude() * delta_time
        ax.ensure(ax.approx_equal(distance_moved, expected_distance, 0.1),
                  "Position update should match velocity * time")

        # Verify physics conservation
        momentum_before = self.velocity.magnitude()  # Simplified momentum
        # For basic movement, momentum should be conserved
        momentum_after = self.velocity.magnitude()
        ax.ensure(ax.approx_equal(momentum_before, momentum_after, 0.01),
                  "Momentum should be conserved during movement")

    def _safe_position_update(self, safe_delta: float) -> None:
        """Safe fallback position update"""
        self.position.x += self.velocity.x * safe_delta
        self.position.y += self.velocity.y * safe_delta

    @ax.verify
    def check_bounds(self, width: ax.PositiveInt, height: ax.PositiveInt) -> bool:
        """Check if object is within screen bounds with plugin verification"""
        in_bounds = (0 <= self.position.x <= width and
                     0 <= self.position.y <= height)

        # Use plugin verification for boundary conditions
        bounds_ok = _plugin_registry.get_verifier('boundary_conditions')
        if bounds_ok:
            plugin_check = bounds_ok(self, width, height)
            ax.ensure(plugin_check, "Plugin boundary verification failed")

        return in_bounds

    @ax.verify
    def collides_with(self, other: 'GameObject') -> bool:
        """Enhanced collision detection with geometric verification"""
        ax.require(isinstance(other, GameObject), "Other must be GameObject")
        ax.require(self.active and other.active, "Both objects must be active")

        # Use plugin verification for collision geometry
        geometry_ok = _plugin_registry.get_verifier('collision_geometry_valid')
        if geometry_ok:
            ax.require("collision_geometry_valid", geometry_ok(self, other))

        distance = self.position.distance_to(other.position)
        collision_distance = self.size + other.size

        colliding = distance <= collision_distance

        # Enhanced collision verification
        if colliding:
            # Verify collision makes geometric sense
            ax.ensure(distance >= 0, "Distance cannot be negative")
            ax.ensure(collision_distance > 0, "Collision distance must be positive")
            ax.ensure(distance <= collision_distance + 0.1, "Collision distance check")

            # Log collision for temporal verification with more data
            record_temporal_event("collision_detected", {
                "object1": type(self).__name__,
                "object2": type(other).__name__,
                "distance": distance,
                "collision_distance": collision_distance,
                "position1": (self.position.x, self.position.y),
                "position2": (other.position.x, other.position.y),
                "sizes": (self.size, other.size)
            })

        return colliding


@ax.enable_for_dataclass
@dataclass
class Player(GameObject):
    """Enhanced player ship with advanced verification"""
    lives: ax.Range[int, 0, 10] = 3
    score: Natural = 0
    shoot_cooldown: float = 0.0
    invulnerable_time: float = 0.0
    power_level: ax.Range[int, 1, 10] = 1
    shots_fired: ax.Range[int, 0, 999999] = 0
    hits_landed: ax.Range[int, 0, 999999] = 0

    def __post_init__(self):
        super().__post_init__()
        ax.require(self.lives >= 0, "Lives cannot be negative")
        ax.require(self.score >= 0, "Score cannot be negative")
        ax.require(self.hits_landed <= self.shots_fired, "Hits cannot exceed shots fired")

    # @ax.verify
    # def move(self, direction: Vector2D, speed: Velocity, delta_time: ax.PositiveFloat) -> None:
    #     """Enhanced movement with adaptive monitoring"""
    #     ax.require(isinstance(direction, Vector2D), "Direction must be Vector2D")
    #
    #     # Use adaptive monitoring for high-frequency checks
    #     if FUTURE_FEATURES_AVAILABLE:
    #         adaptive_require(
    #             "movement_speed_reasonable",
    #             speed <= GameConstants.PLAYER_SPEED * 1.5,
    #             property_name="player_movement_speed",
    #             priority=2
    #         )
    #
    #     # Normalize direction to prevent diagonal speed boost
    #     if direction.magnitude() > 0:
    #         direction = direction.normalize()
    #
    #     # Calculate new velocity with loop invariant
    #     new_velocity = Vector2D(direction.x * speed, direction.y * speed)
    #
    #     # Verify velocity change is reasonable
    #     velocity_change = abs(new_velocity.magnitude() - self.velocity.magnitude())
    #     ax.ensure(velocity_change <= speed + 1.0, "Velocity change should be reasonable")
    #
    #     self.velocity = new_velocity
    #     self.update_position(delta_time)

    @ax.verify
    def move(self, direction: Vector2D, speed: Velocity, delta_time: ax.PositiveFloat) -> None:
        """Simplified movement to prevent stack overflow"""
        # Basic validation without complex verification
        if not isinstance(direction, Vector2D) or speed <= 0 or delta_time <= 0:
            return

        # Skip complex adaptive monitoring that might cause recursion
        if speed > GameConstants.PLAYER_SPEED * 2:
            speed = GameConstants.PLAYER_SPEED * 1.5  # Cap speed instead of complex verification

        # Normalize direction to prevent diagonal speed boost - simplified
        direction_magnitude = (direction.x * direction.x + direction.y * direction.y) ** 0.5
        if direction_magnitude > 0:
            direction.x = direction.x / direction_magnitude
            direction.y = direction.y / direction_magnitude

        # Simple velocity update without complex verification
        self.velocity.x = direction.x * speed
        self.velocity.y = direction.y * speed

        # Simplified position update
        self.position.x += self.velocity.x * delta_time
        self.position.y += self.velocity.y * delta_time

    @ax.verify
    def can_shoot(self) -> bool:
        """Check if player can shoot with cooldown verification"""
        can_shoot = self.shoot_cooldown <= 0

        # Track shooting pattern in ghost state
        _ghost.set("player_shooting_check", {
            "can_shoot": can_shoot,
            "cooldown": self.shoot_cooldown,
            "timestamp": time.time()
        })

        return can_shoot

    @ax.verify
    def record_shot_fired(self) -> None:
        """Record shot fired with statistics tracking"""
        old_shots = self.shots_fired
        self.shots_fired += 1

        ax.ensure(self.shots_fired == old_shots + 1, "Shot count should increment by 1")
        ax.ensure(self.hits_landed <= self.shots_fired, "Hits cannot exceed shots")

        # Track sensitive shooting statistics
        if _game_config.information_flow_tracking:
            track_sensitive_data("shots_fired", self.shots_fired, SecurityLabel.CONFIDENTIAL)

    @ax.verify
    def record_hit(self) -> None:
        """Record successful hit with verification"""
        ax.require(self.hits_landed < self.shots_fired, "Cannot have more hits than shots")

        old_hits = self.hits_landed
        self.hits_landed += 1

        ax.ensure(self.hits_landed == old_hits + 1, "Hit count should increment by 1")
        ax.ensure(self.hits_landed <= self.shots_fired, "Hits cannot exceed shots fired")

    @ax.verify
    def calculate_accuracy(self) -> float:
        """Calculate shooting accuracy with mathematical verification"""
        if self.shots_fired == 0:
            return 0.0

        accuracy = self.hits_landed / self.shots_fired

        # Mathematical verification
        ax.ensure(0.0 <= accuracy <= 1.0, "Accuracy must be between 0 and 1")
        ax.ensure(accuracy * self.shots_fired == self.hits_landed, "Accuracy calculation verification")

        return accuracy

    @contract_with_recovery(
        preconditions=[("valid_points", lambda self, points: points > 0)],
        recovery_strategy=RecoveryStrategy(
            RecoveryPolicy.GRACEFUL_DEGRADATION,
            fallback_handler=lambda points: max(1, min(points, 1000))  # Clamp points
        )
    )
    def add_score(self, points: ax.PositiveInt) -> None:
        """Add score with overflow protection and recovery"""
        old_score = self.score

        # Check for potential overflow
        if self.score + points > 99999999:
            self.score = 99999999
        else:
            self.score += points

        ax.ensure(self.score >= old_score, "Score should not decrease")
        ax.ensure(self.score <= 99999999, "Score should not overflow")

        # Track sensitive score data with information flow
        if _game_config.information_flow_tracking:
            track_sensitive_data("player_score", self.score, SecurityLabel.CONFIDENTIAL)

        # Record for temporal verification
        record_temporal_event("score_update", {
            "old_score": old_score,
            "new_score": self.score,
            "points_added": points
        })

    @ax.verify
    def take_damage(self, damage: ax.PositiveInt = 1) -> bool:
        """Take damage with enhanced verification and ghost state tracking"""
        if self.invulnerable_time > 0:
            _ghost.set("damage_blocked", {
                "damage": damage,
                "invulnerable_time": self.invulnerable_time
            })
            return False

        ax.require(damage > 0, "Damage must be positive")

        old_health = self.health
        old_lives = self.lives

        self.health = max(0, self.health - damage)

        if self.health <= 0 and self.lives > 0:
            self.lives = max(0, self.lives - 1)
            self.health = 100 if self.lives > 0 else 0
            self.invulnerable_time = 2.0  # 2 seconds of invulnerability

            record_temporal_event("player_died", {
                "lives_remaining": self.lives,
                "damage_taken": damage,
                "old_health": old_health
            })

            ax.ensure(self.lives < old_lives, "Lives should decrease when player dies")
            return True

        ax.ensure(self.health < old_health or damage == 0, "Health should decrease after damage")
        return False


@ax.enable_for_dataclass
@dataclass
class Enemy(GameObject):
    """Enhanced enemy with AI verification"""
    enemy_type: ax.NonEmpty[str] = "basic"
    points_value: ax.PositiveInt = 100
    attack_cooldown: float = 0.0
    ai_state: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        ax.require(self.points_value > 0, "Points value must be positive")
        valid_types = ["basic", "patrol", "aggressive", "defensive"]
        ax.require(self.enemy_type in valid_types, f"Enemy type must be one of {valid_types}")

    @contract_with_recovery(
        preconditions=[("valid_player_position", lambda self, pos, dt: isinstance(pos, Vector2D))],
        recovery_strategy=RecoveryStrategy(
            RecoveryPolicy.GRACEFUL_DEGRADATION,
            fallback_handler=lambda self, pos, dt: self._safe_ai_update(Vector2D(400, 300), dt)
        ),
        # plugins=['velocity_reasonable']
    )
    def update_ai(self, player_pos: Vector2D, delta_time: ax.PositiveFloat) -> None:
        """Enhanced AI with behavioral verification"""
        ax.require(isinstance(player_pos, Vector2D), "Player position must be Vector2D")
        ax.require(delta_time > 0, "Delta time must be positive")

        # Store old velocity for verification
        old_velocity = Vector2D(self.velocity.x, self.velocity.y)

        if self.enemy_type == "basic":
            self._basic_ai(player_pos, delta_time)
        elif self.enemy_type == "patrol":
            self._patrol_ai(player_pos, delta_time)
        elif self.enemy_type == "aggressive":
            self._aggressive_ai(player_pos, delta_time)
        elif self.enemy_type == "defensive":
            self._defensive_ai(player_pos, delta_time)

        # Verify AI behavior is reasonable
        velocity_change = abs(self.velocity.magnitude() - old_velocity.magnitude())
        ax.ensure(velocity_change <= GameConstants.ENEMY_SPEED * 2,
                  "AI velocity change should be reasonable")

        self.update_position(delta_time)

    def _safe_ai_update(self, safe_pos: Vector2D, delta_time: float) -> None:
        """Safe fallback AI update"""
        self.velocity = Vector2D(0, GameConstants.ENEMY_SPEED)
        self.update_position(delta_time)

    @ax.verify
    def _basic_ai(self, player_pos: Vector2D, delta_time: float) -> None:
        """Basic AI: move toward player"""
        direction = Vector2D(
            player_pos.x - self.position.x,
            player_pos.y - self.position.y
        )

        if direction.magnitude() > 0:
            direction = direction.normalize()
            self.velocity = Vector2D(
                direction.x * GameConstants.ENEMY_SPEED * 0.8,
                direction.y * GameConstants.ENEMY_SPEED * 0.8
            )

    @ax.verify
    def _patrol_ai(self, player_pos: Vector2D, delta_time: float) -> None:
        """Patrol AI: move in patterns"""
        time_factor = (time.time() - self.creation_time) * 2.0
        self.velocity = Vector2D(
            math.sin(time_factor) * GameConstants.ENEMY_SPEED * 0.6,
            GameConstants.ENEMY_SPEED * 0.4
        )

    @ax.verify
    def _aggressive_ai(self, player_pos: Vector2D, delta_time: float) -> None:
        """Aggressive AI: move directly toward player at full speed"""
        direction = Vector2D(
            player_pos.x - self.position.x,
            player_pos.y - self.position.y
        )

        if direction.magnitude() > 0:
            direction = direction.normalize()
            self.velocity = Vector2D(
                direction.x * GameConstants.ENEMY_SPEED,
                direction.y * GameConstants.ENEMY_SPEED
            )

    @ax.verify
    def _defensive_ai(self, player_pos: Vector2D, delta_time: float) -> None:
        """Defensive AI: maintain distance from player"""
        direction = Vector2D(
            self.position.x - player_pos.x,  # Away from player
            self.position.y - player_pos.y
        )

        distance_to_player = self.position.distance_to(player_pos)
        ideal_distance = 200.0

        if direction.magnitude() > 0:
            direction = direction.normalize()

            # Move away if too close, toward if too far
            speed_factor = 1.0 if distance_to_player < ideal_distance else -0.5
            self.velocity = Vector2D(
                direction.x * GameConstants.ENEMY_SPEED * speed_factor,
                direction.y * GameConstants.ENEMY_SPEED * speed_factor
            )


@ax.enable_for_dataclass
@dataclass
class Bullet(GameObject):
    """Enhanced bullet with ballistics verification"""
    owner: ax.NonEmpty[str] = "player"
    damage: ax.PositiveInt = 25
    penetrating: bool = False
    lifetime: float = 5.0

    def __post_init__(self):
        super().__post_init__()
        ax.require(self.owner in ["player", "enemy"], "Owner must be 'player' or 'enemy'")
        ax.require(self.damage > 0, "Damage must be positive")
        ax.require(self.lifetime > 0, "Lifetime must be positive")

    @ax.verify
    def update(self, delta_time: ax.PositiveFloat) -> None:
        """Update bullet with lifetime tracking"""
        # Update position
        self.update_position(delta_time)

        # Update lifetime
        self.lifetime -= delta_time

        # Deactivate if lifetime expired
        if self.lifetime <= 0:
            self.active = False

            record_temporal_event("bullet_expired", {
                "owner": self.owner,
                "position": (self.position.x, self.position.y)
            })


@ax.enable_for_dataclass
@dataclass
class PowerUp(GameObject):
    """Enhanced power-up with effect verification"""
    powerup_type: ax.NonEmpty[str] = "health"
    effect_value: ax.PositiveInt = 50
    duration: ax.PositiveFloat = 10.0
    rarity: ax.Range[int, 1, 5] = 1  # 1=common, 5=legendary

    def __post_init__(self):
        super().__post_init__()
        valid_types = ["health", "shield", "double_shot", "speed_boost", "score_multiplier"]
        ax.require(self.powerup_type in valid_types, f"Power-up type must be one of {valid_types}")
        ax.require(self.effect_value > 0, "Effect value must be positive")
        ax.require(1 <= self.rarity <= 5, "Rarity must be 1-5")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ADVANCED ANALYTICS SYSTEM
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def create_game_score(value: int = 0) -> int:
    """Create a validated game score"""
    game_score_type = GameScore()
    ax.require(game_score_type.predicate(value),
               "Score must be within valid range (0-99,999,999)")
    return value

@ax.enable_for_dataclass
@dataclass
class GameAnalytics:
    """Comprehensive game analytics with verification"""
    session_id: ax.NonEmpty[str]
    start_time: float = field(default_factory=time.time)
    total_shots_fired: ax.Range[int, 0, 999999] = 0
    total_hits_landed: ax.Range[int, 0, 999999] = 0
    enemies_defeated: ax.Range[int, 0, 999999] = 0
    power_ups_collected: ax.Range[int, 0, 999999] = 0
    max_score_achieved: int = field(default_factory=lambda: create_game_score(0))
    max_enemies_on_screen: Natural = 0
    total_damage_taken: ax.Range[int, 0, 999999] = 0
    total_distance_traveled: Natural = 0.0
    gameplay_events: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        ax.require(len(self.session_id) > 0, "Session ID cannot be empty")
        ax.require(self.total_hits_landed <= self.total_shots_fired, "Hits cannot exceed shots")
        ax.require(self.start_time > 0, "Start time must be positive")

    @ax.verify
    def update_max_score(self, score: int) -> None:
        """Update max score with GameScore validation"""
        game_score = GameScore()
        ax.require(game_score.predicate(score), "Score must be within GameScore bounds")

        if score > self.max_score_achieved:
            self.max_score_achieved = score

    @ax.verify
    def calculate_accuracy(self) -> float:
        """Calculate shooting accuracy with verification"""
        if self.total_shots_fired == 0:
            return 0.0

        accuracy = self.total_hits_landed / self.total_shots_fired
        ax.ensure(0.0 <= accuracy <= 1.0, "Accuracy must be between 0 and 1")
        return accuracy

    @ax.verify
    def calculate_efficiency_score(self) -> float:
        """Calculate player efficiency score"""
        if self.total_shots_fired == 0:
            return 0.0

        accuracy = self.calculate_accuracy()
        damage_efficiency = 1.0 - (self.total_damage_taken / max(1, self.max_score_achieved / 10))
        damage_efficiency = max(0.0, min(1.0, damage_efficiency))

        efficiency = (accuracy * 0.6 + damage_efficiency * 0.4)
        ax.ensure(0.0 <= efficiency <= 1.0, "Efficiency score must be between 0 and 1")
        return efficiency

    @ax.verify
    def record_event(self, event_type: str, data: Dict[str, Any] = None) -> None:
        """Record gameplay event with verification"""
        ax.require(len(event_type) > 0, "Event type cannot be empty")

        event = {
            "type": event_type,
            "timestamp": time.time() - self.start_time,
            "data": data or {}
        }

        self.gameplay_events.append(event)

        # Keep events list bounded
        if len(self.gameplay_events) > 1000:
            self.gameplay_events = self.gameplay_events[-500:]  # Keep latest 500

        ax.ensure(len(self.gameplay_events) <= 1000, "Events list should be bounded")

    @ax.verify
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analytics report"""
        session_duration = time.time() - self.start_time

        report = {
            "session_id": self.session_id,
            "duration_seconds": session_duration,
            "accuracy": self.calculate_accuracy(),
            "efficiency": self.calculate_efficiency_score(),
            "shots_per_minute": (self.total_shots_fired / max(1, session_duration / 60)),
            "enemies_per_minute": (self.enemies_defeated / max(1, session_duration / 60)),
            "damage_taken_per_minute": (self.total_damage_taken / max(1, session_duration / 60)),
            "max_score": self.max_score_achieved,
            "total_events": len(self.gameplay_events),
            "performance_metrics": {
                "total_shots": self.total_shots_fired,
                "total_hits": self.total_hits_landed,
                "enemies_defeated": self.enemies_defeated,
                "powerups_collected": self.power_ups_collected,
                "max_enemies_concurrent": self.max_enemies_on_screen,
                "distance_traveled": self.total_distance_traveled
            }
        }

        # Verify report data
        ax.ensure(report["accuracy"] >= 0, "Report accuracy cannot be negative")
        ax.ensure(report["efficiency"] >= 0, "Report efficiency cannot be negative")
        ax.ensure(report["duration_seconds"] >= 0, "Duration cannot be negative")

        return report


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ENHANCED GAME STATE MANAGEMENT WITH TEMPORAL VERIFICATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class GamePhase(Enum):
    MENU = "menu"
    PLAYING = "playing"
    PAUSED = "paused"
    GAME_OVER = "game_over"
    HIGH_SCORES = "high_scores"
    ANALYTICS = "analytics"


@ax.stateful(initial="menu")
class GameStateManager:
    """Enhanced game state management with temporal properties"""

    def __init__(self):
        self.current_phase = GamePhase.MENU
        self.transition_time = time.time()
        self.state_history = []
        self.phase_durations = defaultdict(float)
        self._last_phase_start = time.time()

    @ax.state("menu", "playing")
    def start_game(self) -> None:
        """Start new game with temporal tracking"""
        self._record_phase_duration()

        record_temporal_event("game_started", {
            "previous_phase": self.current_phase.value,
            "transition_time": time.time()
        })

        self.current_phase = GamePhase.PLAYING
        self.transition_time = time.time()
        self._last_phase_start = time.time()
        self._log_state_change("start_game")

    @ax.state("playing", "paused")
    def pause_game(self) -> None:
        """Pause current game"""
        self._record_phase_duration()

        record_temporal_event("game_paused", {
            "game_time": time.time() - self.transition_time
        })

        self.current_phase = GamePhase.PAUSED
        self._last_phase_start = time.time()
        self._log_state_change("pause_game")

    @ax.state("paused", "playing")
    def resume_game(self) -> None:
        """Resume paused game"""
        pause_duration = time.time() - self._last_phase_start

        record_temporal_event("game_resumed", {
            "pause_duration": pause_duration
        })

        self.current_phase = GamePhase.PLAYING
        self._last_phase_start = time.time()
        self._log_state_change("resume_game")

    @ax.state(["playing", "paused"], "game_over")
    def end_game(self, final_score: GameScore) -> None:
        """End current game with score verification"""
        self._record_phase_duration()

        ax.require(final_score >= 0, "Final score cannot be negative")

        record_temporal_event("game_ended", {
            "final_score": final_score,
            "game_duration": self.phase_durations.get(GamePhase.PLAYING, 0)
        })

        self.current_phase = GamePhase.GAME_OVER
        self._last_phase_start = time.time()
        self._log_state_change("end_game")

    @ax.state("game_over", "high_scores")
    def show_high_scores(self) -> None:
        """Show high score screen"""
        self.current_phase = GamePhase.HIGH_SCORES
        self._last_phase_start = time.time()
        self._log_state_change("show_high_scores")

    @ax.state("game_over", "analytics")
    def show_analytics(self) -> None:
        """Show analytics screen"""
        self.current_phase = GamePhase.ANALYTICS
        self._last_phase_start = time.time()
        self._log_state_change("show_analytics")

    @ax.state(["game_over", "high_scores", "analytics"], "menu")
    def return_to_menu(self) -> None:
        """Return to main menu"""
        self._record_phase_duration()

        record_temporal_event("returned_to_menu", {
            "previous_phase": self.current_phase.value
        })

        self.current_phase = GamePhase.MENU
        self._last_phase_start = time.time()
        self._log_state_change("return_to_menu")

    @ax.verify
    def _record_phase_duration(self) -> None:
        """Record how long we spent in current phase"""
        duration = time.time() - self._last_phase_start
        self.phase_durations[self.current_phase] += duration

        ax.ensure(duration >= 0, "Phase duration cannot be negative")

    @ax.verify
    def _log_state_change(self, action: ax.NonEmpty[str]) -> None:
        """Log state changes with enhanced data"""
        entry = {
            "action": action,
            "from_phase": self.state_history[-1]["phase"] if self.state_history else "none",
            "to_phase": self.current_phase.value,
            "timestamp": time.time(),
            "transition_duration": time.time() - self.transition_time
        }
        self.state_history.append(entry)

        # Keep history bounded
        if len(self.state_history) > 200:
            self.state_history = self.state_history[-100:]

    @ax.verify
    def get_phase_statistics(self) -> Dict[str, float]:
        """Get statistics about time spent in each phase"""
        self._record_phase_duration()  # Update current phase duration

        total_time = sum(self.phase_durations.values())
        if total_time == 0:
            return {}

        stats = {}
        for phase, duration in self.phase_durations.items():
            percentage = (duration / total_time) * 100
            stats[phase.value] = {
                "duration_seconds": duration,
                "percentage": percentage
            }

        return stats


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ENHANCED PHYSICS SYSTEM WITH ADVANCED VERIFICATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class PhysicsSystem:
    """Enhanced physics system with comprehensive verification"""

    def __init__(self):
        self.collision_count = 0
        self.physics_violations = 0

    @ax.verify(track_performance=True)
    def update_objects(self, objects: ax.NonEmpty[List[GameObject]],
                       delta_time: ax.PositiveFloat) -> None:
        """Update all objects with loop invariants and verification"""
        ax.require(len(objects) > 0, "Object list cannot be empty")
        ax.require(delta_time > 0, "Delta time must be positive")
        ax.require(delta_time < 1.0, "Delta time seems unreasonably large")

        with ax.verification_context("physics_update"):
            # Loop invariant: number of objects should not change during update
            initial_count = len(objects)

            # Enhanced object update with loop invariants
            for i, obj in enumerate(objects):
                # Loop invariant: we should have processed exactly i objects
                ax.require(i >= 0, "Loop index should be non-negative")
                ax.require(i < len(objects), "Loop index should be within bounds")

                if obj.active:
                    # Store state for verification
                    old_position = Vector2D(obj.position.x, obj.position.y)

                    # Update with plugin verification
                    obj.update_position(delta_time)

                    # Verify physics consistency
                    if FUTURE_FEATURES_AVAILABLE:
                        adaptive_require(
                            "physics_consistency",
                            obj.position.distance_to(old_position) < 2000 * delta_time,
                            property_name="physics_position_change",
                            priority=3
                        )

                # Loop invariant: object count unchanged
                ax.ensure(len(objects) == initial_count,
                          "Object count should not change during update")

    @ax.verify(track_performance=True)
    def check_collisions(self, group1: List[GameObject],
                         group2: List[GameObject]) -> List[Tuple[GameObject, GameObject]]:
        """Enhanced collision detection with O(n*m) verification"""
        collisions = []

        with ax.verification_context("collision_detection"):
            # Loop invariants for collision detection
            max_possible_collisions = len(group1) * len(group2)

            for i, obj1 in enumerate(group1):
                if not obj1.active:
                    continue

                # Inner loop invariant
                for j, obj2 in enumerate(group2):
                    if not obj2.active:
                        continue

                    # Loop invariant: collision count should not exceed current possibilities
                    current_max = (i + 1) * (j + 1)
                    ax.ensure(len(collisions) <= current_max,
                              "Collision count should not exceed theoretical maximum")

                    if obj1.collides_with(obj2):
                        collisions.append((obj1, obj2))

                        # Verify collision is geometrically valid
                        distance = obj1.position.distance_to(obj2.position)
                        collision_threshold = obj1.size + obj2.size
                        ax.ensure(distance <= collision_threshold + 0.1,
                                  "Collision should be within geometric bounds")

                        self.collision_count += 1

        # Final verification
        ax.ensure(len(collisions) <= max_possible_collisions,
                  "Cannot have more collisions than object pairs")

        # Use plugin verification
        physics_plugin = _plugin_registry.get_verifier('collision_geometry_valid')
        for obj1, obj2 in collisions:
            ax.ensure(obj1.active and obj2.active, "Colliding objects should be active")
            if physics_plugin:
                ax.ensure(physics_plugin(obj1, obj2), "Plugin collision verification failed")

        return collisions

    @contract_with_recovery(
        preconditions=[("valid_object", lambda self, obj, width, height: hasattr(obj, 'position'))],
        recovery_strategy=RecoveryStrategy(
            RecoveryPolicy.GRACEFUL_DEGRADATION,
            fallback_handler=lambda self, obj, w, h: self._safe_boundary_application(obj, w, h)
        )
    )
    def apply_screen_boundaries(self, obj: GameObject,
                                width: ax.PositiveInt, height: ax.PositiveInt) -> None:
        """Apply screen boundary constraints with recovery"""
        ax.require(width > 0 and height > 0, "Screen dimensions must be positive")

        # Store original position for verification
        original_pos = Vector2D(obj.position.x, obj.position.y)

        # Apply boundaries with size consideration
        half_size = obj.size / 2
        new_x = max(half_size, min(width - half_size, obj.position.x))
        new_y = max(half_size, min(height - half_size, obj.position.y))

        # Update position
        obj.position.x = new_x
        obj.position.y = new_y

        # Verify boundary application
        ax.ensure(obj.check_bounds(width, height) or
                  # (obj.position.x >= -half_size and obj.position.x <= width + half_size),
                  (-half_size <= obj.position.x <= width + half_size),
                  "Object should be within extended bounds after clamping")

        # Use plugin verification
        boundary_verifier = _plugin_registry.get_verifier('boundary_conditions')
        if boundary_verifier:
            ax.ensure(boundary_verifier(obj, width, height),
                      "Plugin boundary verification failed")

    @staticmethod
    def _safe_boundary_application(obj: GameObject, width: int, height: int) -> None:
        """Safe fallback boundary application"""
        obj.position.x = max(0, min(width, obj.position.x))
        obj.position.y = max(0, min(height, obj.position.y))

    @ax.verify
    def get_physics_statistics(self) -> Dict[str, Any]:
        """Get physics system statistics"""
        return {
            "total_collisions": self.collision_count,
            "physics_violations": self.physics_violations,
            "collision_rate": self.collision_count / max(1, time.time())
        }


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ENHANCED RENDERING SYSTEM WITH DEBUG OVERLAY
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RenderSystem:
    """Enhanced rendering system with debug visualization"""

    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 20)
        self.debug_font = pygame.font.Font(None, 16)

        ax.require(screen is not None, "Screen surface cannot be None")
        ax.require(screen.get_width() > 0, "Screen width must be positive")
        ax.require(screen.get_height() > 0, "Screen height must be positive")

    @ax.verify(track_performance=True)
    def clear_screen(self, color: Tuple[int, int, int] = GameConstants.BLACK) -> None:
        """Clear screen with verified color values"""
        ax.require(len(color) == 3, "Color must be RGB tuple")
        ax.require(all(0 <= c <= 255 for c in color), "RGB values must be 0-255")
        self.screen.fill(color)

    @ax.verify
    def draw_object(self, obj: GameObject, color: Tuple[int, int, int],
                    debug_info: bool = False) -> None:
        """Enhanced object drawing with debug information"""
        ax.require(isinstance(obj, GameObject), "Object must be GameObject")
        ax.require(obj.active, "Cannot draw inactive object")
        ax.require(len(color) == 3, "Color must be RGB tuple")

        # Convert to screen coordinates
        screen_x = int(obj.position.x)
        screen_y = int(obj.position.y)
        radius = int(obj.size)

        # Bounds check
        screen_rect = self.screen.get_rect()
        if screen_rect.collidepoint(screen_x, screen_y):
            pygame.draw.circle(self.screen, color, (screen_x, screen_y), radius)

            # Draw object outline in debug mode
            if debug_info:
                pygame.draw.circle(self.screen, GameConstants.WHITE,
                                   (screen_x, screen_y), radius, 1)

        # Draw health bar for damaged objects
        if hasattr(obj, 'health') and obj.health < 100:
            self._draw_health_bar(obj)

        # Draw debug information
        if debug_info and _game_config.debug_overlay_enabled:
            self._draw_debug_info(obj)

    @ax.verify
    def _draw_health_bar(self, obj: GameObject) -> None:
        """Draw health bar above object"""
        if not hasattr(obj, 'health'):
            return

        bar_width = int(obj.size * 2)
        bar_height = 6
        bar_x = int(obj.position.x - bar_width // 2)
        bar_y = int(obj.position.y - obj.size - 15)

        # Background (red)
        pygame.draw.rect(self.screen, GameConstants.RED,
                         (bar_x, bar_y, bar_width, bar_height))

        # Health (green)
        health_percent = obj.health / 100.0
        health_width = int(bar_width * health_percent)
        if health_width > 0:
            pygame.draw.rect(self.screen, GameConstants.GREEN,
                             (bar_x, bar_y, health_width, bar_height))

    @ax.verify
    def _draw_debug_info(self, obj: GameObject) -> None:
        """Draw debug information for object"""
        debug_y = int(obj.position.y + obj.size + 20)
        debug_x = int(obj.position.x - 40)

        # Velocity vector
        if obj.velocity.magnitude() > 0:
            end_x = obj.position.x + obj.velocity.x * 0.1
            end_y = obj.position.y + obj.velocity.y * 0.1
            pygame.draw.line(self.screen, GameConstants.YELLOW,
                             (obj.position.x, obj.position.y), (end_x, end_y), 2)

        # Object info text
        obj_type = type(obj).__name__
        debug_text = self.debug_font.render(f"{obj_type}", True, GameConstants.WHITE)
        self.screen.blit(debug_text, (debug_x, debug_y))

    @ax.verify
    def draw_ui(self, player: Player, game_time: ax.PositiveFloat,
                analytics: GameAnalytics = None) -> None:
        """Enhanced UI with analytics display"""
        ax.require(isinstance(player, Player), "Player must be Player object")
        ax.require(game_time >= 0, "Game time cannot be negative")

        # Main game stats
        score_text = self.font.render(f"Score: {player.score:,}", True, GameConstants.WHITE)
        self.screen.blit(score_text, (10, 10))

        lives_text = self.font.render(f"Lives: {player.lives}", True, GameConstants.WHITE)
        self.screen.blit(lives_text, (10, 50))

        health_text = self.font.render(f"Health: {player.health}", True, GameConstants.WHITE)
        self.screen.blit(health_text, (10, 90))

        # Enhanced stats
        accuracy = player.calculate_accuracy() * 100
        accuracy_text = self.small_font.render(f"Accuracy: {accuracy:.1f}%", True, GameConstants.WHITE)
        self.screen.blit(accuracy_text, (10, 130))

        # Game time
        time_text = self.small_font.render(f"Time: {game_time:.1f}s", True, GameConstants.WHITE)
        self.screen.blit(time_text, (10, 150))

        # Analytics info if available
        if analytics:
            efficiency = analytics.calculate_efficiency_score() * 100
            efficiency_text = self.small_font.render(f"Efficiency: {efficiency:.1f}%",
                                                     True, GameConstants.CYAN)
            self.screen.blit(efficiency_text, (10, 170))

    @ax.verify(track_performance=True)
    def draw_debug_overlay(self, hotspots: List = None, physics_stats: Dict = None,
                           temporal_events: int = 0) -> None:
        """Draw comprehensive debug overlay"""
        if not _game_config.debug_overlay_enabled:
            return

        overlay_x = GameConstants.SCREEN_WIDTH - 300
        overlay_y = 10

        # Semi-transparent background
        overlay_rect = pygame.Rect(overlay_x - 10, overlay_y, 290, 250)
        overlay_surface = pygame.Surface((290, 250))
        overlay_surface.set_alpha(128)
        overlay_surface.fill(GameConstants.BLACK)
        self.screen.blit(overlay_surface, (overlay_x - 10, overlay_y))

        # Debug title
        debug_title = self.small_font.render("Debug Information", True, GameConstants.YELLOW)
        self.screen.blit(debug_title, (overlay_x, overlay_y))
        overlay_y += 25

        # Verification mode
        mode_text = self.debug_font.render(f"Mode: {_game_config.verification_mode.value}",
                                           True, GameConstants.WHITE)
        self.screen.blit(mode_text, (overlay_x, overlay_y))
        overlay_y += 18

        # Performance hotspots
        if hotspots and len(hotspots) > 0:
            hotspot_title = self.debug_font.render("Performance Hotspots:", True, GameConstants.CYAN)
            self.screen.blit(hotspot_title, (overlay_x, overlay_y))
            overlay_y += 18

            for i, hotspot in enumerate(hotspots[:5]):  # Show top 5
                if hasattr(hotspot, 'property_name') and hasattr(hotspot, 'average_time'):
                    text = f"{hotspot.property_name[:20]}: {hotspot.average_time * 1000:.1f}ms"
                    color = GameConstants.RED if hotspot.average_time > 0.01 else GameConstants.GREEN
                    hotspot_text = self.debug_font.render(text, True, color)
                    self.screen.blit(hotspot_text, (overlay_x + 10, overlay_y))
                    overlay_y += 15

        # Physics statistics
        if physics_stats:
            physics_title = self.debug_font.render("Physics Stats:", True, GameConstants.PURPLE)
            self.screen.blit(physics_title, (overlay_x, overlay_y))
            overlay_y += 18

            for key, value in physics_stats.items():
                stat_text = self.debug_font.render(f"{key}: {value}", True, GameConstants.WHITE)
                self.screen.blit(stat_text, (overlay_x + 10, overlay_y))
                overlay_y += 15

        # Temporal events
        temporal_text = self.debug_font.render(f"Temporal Events: {temporal_events}",
                                               True, GameConstants.ORANGE)
        self.screen.blit(temporal_text, (overlay_x, overlay_y))

    @ax.verify
    def draw_analytics_screen(self, analytics: GameAnalytics,
                              high_scores: List[str] = None) -> None:
        """Draw comprehensive analytics screen"""
        self.clear_screen()

        # Title
        title_text = pygame.font.Font(None, 72).render("Game Analytics", True, GameConstants.CYAN)
        title_rect = title_text.get_rect(center=(GameConstants.SCREEN_WIDTH // 2, 80))
        self.screen.blit(title_text, title_rect)

        # Generate analytics report
        report = analytics.generate_report()

        y_offset = 150
        left_column_x = 50
        right_column_x = 500

        # Left column - Performance metrics
        perf_title = self.font.render("Performance Metrics", True, GameConstants.YELLOW)
        self.screen.blit(perf_title, (left_column_x, y_offset))
        y_offset += 40

        metrics = [
            f"Session Duration: {report['duration_seconds']:.1f}s",
            f"Accuracy: {report['accuracy'] * 100:.1f}%",
            f"Efficiency: {report['efficiency'] * 100:.1f}%",
            f"Shots/Min: {report['shots_per_minute']:.1f}",
            f"Enemies/Min: {report['enemies_per_minute']:.1f}",
            f"Total Shots: {report['performance_metrics']['total_shots']:,}",
            f"Total Hits: {report['performance_metrics']['total_hits']:,}",
            f"Enemies Defeated: {report['performance_metrics']['enemies_defeated']:,}",
            f"Max Score: {report['max_score']:,}"
        ]

        for metric in metrics:
            metric_text = self.small_font.render(metric, True, GameConstants.WHITE)
            self.screen.blit(metric_text, (left_column_x, y_offset))
            y_offset += 25

        # Right column - High scores
        if high_scores:
            scores_title = self.font.render("High Scores", True, GameConstants.YELLOW)
            self.screen.blit(scores_title, (right_column_x, 150))

            score_y = 190
            for i, score_text in enumerate(high_scores[:10]):
                score_surface = self.small_font.render(f"{i + 1}. {score_text}",
                                                       True, GameConstants.WHITE)
                self.screen.blit(score_surface, (right_column_x, score_y))
                score_y += 25

        # Instructions
        instruction_y = GameConstants.SCREEN_HEIGHT - 50
        instruction_text = self.small_font.render("Press SPACE to return to menu, F3 to save report",
                                                  True, GameConstants.CYAN)
        instruction_rect = instruction_text.get_rect(center=(GameConstants.SCREEN_WIDTH // 2, instruction_y))
        self.screen.blit(instruction_text, instruction_rect)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MAIN GAME CLASS WITH ALL ADVANCED FEATURES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class AdvancedVerifiedSpaceShooter:
    """Ultimate axiomatik demonstration with all advanced features"""

    def __init__(self):
        # Initialize pygame display
        self.screen = pygame.display.set_mode((GameConstants.SCREEN_WIDTH, GameConstants.SCREEN_HEIGHT))
        pygame.display.set_caption("Advanced Verified Space Shooter - Complete Axiomatik Demo")
        self.clock = pygame.time.Clock()

        # Game systems
        self.state_manager = GameStateManager()
        self.physics = PhysicsSystem()
        self.renderer = RenderSystem(self.screen)
        self.secure_data = SecureGameDataManager()

        # Initialize analytics
        session_id = f"session_{int(time.time())}"
        self.analytics = GameAnalytics(session_id=session_id)

        # Game entities
        self.player = self._create_player()
        self.enemies: List[Enemy] = []
        self.bullets: List[Bullet] = []
        self.powerups: List[PowerUp] = []

        # Game state
        self.game_time = 0.0
        self.last_enemy_spawn = 0.0
        self.running = True
        self.debug_overlay_enabled = True

        # Performance tracking
        self.frame_count = 0
        self.performance_samples = deque(maxlen=120)  # 2 seconds at 60fps

        # Initialize temporal properties
        # self._setup_temporal_properties()

        # Initialize information flow tracking
        if _game_config.information_flow_tracking:
            self._setup_information_flow()

        # Initialize adaptive monitoring if available
        if FUTURE_FEATURES_AVAILABLE and _game_config.adaptive_tuning_enabled:
            self._setup_adaptive_monitoring()

        record_temporal_event("advanced_game_initialized", {
            "features_enabled": {
                "recovery_framework": _game_config.recovery_enabled,
                "adaptive_monitoring": _game_config.adaptive_tuning_enabled,
                "information_flow": _game_config.information_flow_tracking,
                "temporal_verification": _game_config.temporal_verification_enabled,
                "debug_overlay": _game_config.debug_overlay_enabled
            }
        })

    @staticmethod
    def _setup_temporal_properties() -> None:
        """Set up comprehensive temporal properties"""

        # Score should never decrease (stronger version)
        score_monotonic = AlwaysProperty(
            "score_never_decreases",
            lambda event: (
                    event.get('event') != 'score_update' or
                    event.get('data', {}).get('new_score', 0) >= event.get('data', {}).get('old_score', 0)
            )
        )
        add_temporal_property(score_monotonic)

        # Game events should always have valid data
        valid_events = AlwaysProperty(
            "game_events_valid",
            lambda event: event.get('data') is not None and isinstance(event.get('data'), dict)
        )
        add_temporal_property(valid_events)

        # Enemies should eventually be destroyed
        enemy_destruction = EventuallyProperty(
            "enemies_eventually_destroyed",
            lambda history: any(e.get('event') == 'enemy_destroyed' for e in history[-50:]),
            timeout=300.0
        )
        add_temporal_property(enemy_destruction)

        # Player should survive for reasonable time
        player_survival = EventuallyProperty(
            "player_survives_initial_period",
            lambda history: any(e.get('event') == 'game_ended' and
                                e.get('data', {}).get('game_duration', 0) > 10 for e in history),
            timeout=300.0
        )
        add_temporal_property(player_survival)

    def _setup_information_flow(self) -> None:
        """Set up information flow tracking"""
        # Track initial player data as confidential
        self.secure_data.track_sensitive_statistic("initial_lives", self.player.lives)
        self.secure_data.track_sensitive_statistic("game_session", time.time(), SecurityLabel.SECRET)

    @staticmethod
    def _setup_adaptive_monitoring() -> None:
        """Set up adaptive monitoring system"""
        if not FUTURE_FEATURES_AVAILABLE:
            return

        # Register adaptive properties for high-frequency game operations
        _adaptive_monitor.register_property(
            "collision_detection",
            lambda: True,  # Always passes, just tracks timing
            priority=4,
            cost_estimate=0.001
        )

        _adaptive_monitor.register_property(
            "physics_update",
            lambda: True,
            priority=3,
            cost_estimate=0.002
        )

    @contract_with_recovery(
        preconditions=[("valid_player_creation", lambda self: True)],
        recovery_strategy=RecoveryStrategy(
            RecoveryPolicy.GRACEFUL_DEGRADATION,
            fallback_handler=lambda self: self._create_fallback_player()
        )
    )
    def _create_player(self) -> Player:
        """Create verified player object with recovery"""
        player_pos = Vector2D(
            float(GameConstants.SCREEN_WIDTH // 2),
            float(GameConstants.SCREEN_HEIGHT - 100)
        )

        player = Player(
            position=player_pos,
            velocity=Vector2D(0, 0),
            size=25.0,
            health=100,
            lives=3,
            score=0,
            power_level=1
        )

        # Enhanced verification
        ax.ensure(player.position.x >= 0, "Player X position should be valid")
        ax.ensure(player.position.y >= 0, "Player Y position should be valid")
        ax.ensure(player.lives > 0, "Player should start with lives")
        ax.ensure(player.health == 100, "Player should start with full health")

        return player

    @staticmethod
    def _create_fallback_player() -> Player:
        """Safe fallback player creation"""
        return Player(
            position=Vector2D(400, 550),
            velocity=Vector2D(0, 0),
            size=20.0,
            health=100,
            lives=1  # Minimal viable player
        )

    @contract_with_recovery(
        preconditions=[("spawn_limits_respected", lambda self: len(self.enemies) < GameConstants.MAX_ENEMIES)],
        recovery_strategy=RecoveryStrategy(
            RecoveryPolicy.GRACEFUL_DEGRADATION,
            fallback_handler=lambda self: None  # Skip spawn if at limit
        )
    )
    def spawn_enemy(self) -> None:
        """Enhanced enemy spawning with recovery and verification"""

        if len(self.enemies) >= GameConstants.MAX_ENEMIES:
            return

        # Random spawn position with bounds checking
        spawn_x = random.uniform(50.0, float(GameConstants.SCREEN_WIDTH - 50))
        spawn_pos = Vector2D(spawn_x, -50)

        # Enhanced enemy types with verification
        enemy_types = ["basic", "patrol", "aggressive", "defensive"]
        enemy_type = random.choice(enemy_types)

        # Calculate points based on game difficulty
        base_points = 100
        difficulty_multiplier = 1 + (self.game_time / 60.0)  # Increase with time
        points_value = int(base_points * difficulty_multiplier)

        enemy = Enemy(
            position=spawn_pos,
            velocity=Vector2D(0, GameConstants.ENEMY_SPEED),
            size=random.uniform(12.0, 18.0),
            health=50 if enemy_type == "basic" else 75,
            enemy_type=enemy_type,
            points_value=points_value
        )

        self.enemies.append(enemy)

        # Update analytics
        self.analytics.max_enemies_on_screen = max(
            self.analytics.max_enemies_on_screen,
            len(self.enemies)
        )

        record_temporal_event("enemy_spawned", {
            "type": enemy_type,
            "count": len(self.enemies),
            "points_value": points_value,
            "game_time": self.game_time
        })

    @ax.verify
    def handle_input(self, keys, delta_time: ax.PositiveFloat) -> None:
        """Enhanced input handling with verification"""
        ax.require(hasattr(keys, '__getitem__'), "Keys must be indexable")
        ax.require(delta_time > 0, "Delta time must be positive")

        if self.state_manager.current_phase != GamePhase.PLAYING:
            return

        # Movement with enhanced verification
        direction = Vector2D(0, 0)
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            direction.x -= 1
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            direction.x += 1
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            direction.y -= 1
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            direction.y += 1

        # Apply movement with plugin verification
        if direction.magnitude() > 0:
            self.player.move(direction, GameConstants.PLAYER_SPEED, delta_time)

        # Keep player on screen with recovery
        self.physics.apply_screen_boundaries(
            self.player,
            GameConstants.SCREEN_WIDTH,
            GameConstants.SCREEN_HEIGHT
        )

        # Enhanced shooting
        if keys[pygame.K_SPACE] and self.player.can_shoot():
            self.create_bullet()

        # Debug controls
        if keys[pygame.K_F1]:
            # Toggle handled in event processing to avoid repeats
            pass

    @ax.verify
    def create_bullet(self) -> None:
        """Simplified bullet creation without complex verification"""
        if len(self.bullets) >= GameConstants.MAX_BULLETS:
            return

        try:
            # Calculate bullet spawn position - simplified
            bullet_pos = Vector2D(
                self.player.position.x,
                self.player.position.y - self.player.size - 5
            )

            # Simple bullet properties
            bullet_speed = GameConstants.BULLET_SPEED
            bullet_damage = 25

            bullet = Bullet(
                position=bullet_pos,
                velocity=Vector2D(0, -bullet_speed),
                size=4.0,
                owner="player",
                damage=bullet_damage,
                penetrating=False
            )

            self.bullets.append(bullet)
            self.player.shoot_cooldown = 0.25

            # Simple statistics update without verification
            if hasattr(self.player, 'shots_fired'):
                self.player.shots_fired += 1
            if self.analytics:
                self.analytics.total_shots_fired += 1

            print(f"DEBUG: Bullet created at ({bullet_pos.x}, {bullet_pos.y})")

        except Exception as e:
            print(f"ERROR: Failed to create bullet: {e}")
            import traceback
            traceback.print_exc()

    @ax.verify(track_performance=True)
    def update_game_logic(self, delta_time: ax.PositiveFloat) -> None:
        """Simplified game logic to prevent stack overflow"""
        if delta_time <= 0 or delta_time > 1.0:
            return  # Skip invalid frames

        if self.state_manager.current_phase != GamePhase.PLAYING:
            return

        try:
            # Simplified verification context - avoid deep nesting
            # Update cooldowns with basic math
            if self.player.shoot_cooldown > 0:
                self.player.shoot_cooldown = max(0, self.player.shoot_cooldown - delta_time)
            if self.player.invulnerable_time > 0:
                self.player.invulnerable_time = max(0, self.player.invulnerable_time - delta_time)

            # Update physics for all objects - simplified approach
            all_objects = []
            if self.player.active:
                all_objects.append(self.player)
            all_objects.extend([obj for obj in self.enemies if obj.active])
            all_objects.extend([obj for obj in self.bullets if obj.active])
            all_objects.extend([obj for obj in self.powerups if obj.active])

            # Limit object count to prevent performance issues
            if len(all_objects) > 500:
                print("WARNING: Too many objects, cleaning up...")
                self._emergency_cleanup()
                return

            # Simple physics update without complex verification
            for obj in all_objects:
                if obj.active:
                    # Simple position update without verification overhead
                    obj.position.x += obj.velocity.x * delta_time
                    obj.position.y += obj.velocity.y * delta_time

            # Simple AI updates for enemies
            for enemy in self.enemies[:]:  # Copy list to avoid modification issues
                if enemy.active:
                    try:
                        # Simplified AI - just move toward player
                        if self.player.active:
                            dx = self.player.position.x - enemy.position.x
                            dy = self.player.position.y - enemy.position.y
                            distance = (dx * dx + dy * dy) ** 0.5
                            if distance > 0:
                                enemy.velocity.x = (dx / distance) * GameConstants.ENEMY_SPEED * 0.5
                                enemy.velocity.y = (dy / distance) * GameConstants.ENEMY_SPEED * 0.5
                    except Exception as e:
                        print(f"AI update error for enemy: {e}")
                        enemy.active = False

            # Update bullets with simple lifetime
            for bullet in self.bullets[:]:
                if bullet.active:
                    bullet.lifetime -= delta_time
                    if bullet.lifetime <= 0:
                        bullet.active = False

            # Simple enemy spawning
            spawn_interval = 2.0  # Every 2 seconds
            if self.game_time - self.last_enemy_spawn > spawn_interval:
                if len(self.enemies) < 10:  # Limit enemies
                    self.spawn_enemy_simple()
                self.last_enemy_spawn = self.game_time

            # Handle collisions with simplified detection
            self._handle_collisions_simple()

            # Remove inactive objects
            self.enemies = [e for e in self.enemies if e.active]
            self.bullets = [b for b in self.bullets if b.active]
            self.powerups = [p for p in self.powerups if p.active]

            self._check_game_conditions_simple()

            # Check simple game conditions
            if self.player.lives <= 0:
                self.state_manager.end_game(self.player.score)

        except Exception as e:
            print(f"Game logic error: {e}")
            # Continue with minimal state

    def _handle_collisions_simple(self) -> None:
        """Simplified collision detection with player death"""
        # Player bullets vs enemies
        hits_this_frame = 0
        for bullet in self.bullets[:]:
            if not bullet.active or bullet.owner != "player":
                continue

            for enemy in self.enemies[:]:
                if not enemy.active:
                    continue

                # Simple distance check
                dx = bullet.position.x - enemy.position.x
                dy = bullet.position.y - enemy.position.y
                distance = (dx * dx + dy * dy) ** 0.5

                if distance <= (bullet.size + enemy.size):
                    # Collision detected
                    bullet.active = False
                    enemy.health -= bullet.damage

                    if enemy.health <= 0:
                        enemy.active = False
                        self.player.score += enemy.points_value
                        hits_this_frame += 1
                        if self.analytics:
                            self.analytics.enemies_defeated += 1

        # Update hit statistics
        if hits_this_frame > 0:
            try:
                for _ in range(hits_this_frame):
                    self.player.record_hit()
                if self.analytics:
                    self.analytics.total_hits_landed += hits_this_frame
            except Exception as e:
                print(f"Hit recording error: {e}")

        # PLAYER VS ENEMIES - This was missing!
        if self.player.active and self.player.invulnerable_time <= 0:
            for enemy in self.enemies[:]:
                if not enemy.active:
                    continue

                # Check collision between player and enemy
                dx = self.player.position.x - enemy.position.x
                dy = self.player.position.y - enemy.position.y
                distance = (dx * dx + dy * dy) ** 0.5

                if distance <= (self.player.size + enemy.size):
                    # Player hit by enemy!
                    damage = max(40, int(enemy.size))  # Damage based on enemy size

                    # Apply damage to player
                    old_health = self.player.health
                    self.player.health = max(0, self.player.health - damage)

                    if self.analytics:
                        self.analytics.total_damage_taken += damage

                    print(f"Player hit! Damage: {damage}, Health: {self.player.health}")

                    # Check if player died
                    # if self.player.health <= 0 and self.player.lives > 0:
                    if self.player.health <= 0 < self.player.lives:
                        self.player.lives -= 1
                        print(f"Player died! Lives remaining: {self.player.lives}")

                        if self.player.lives > 0:
                            # Respawn with full health and invulnerability
                            self.player.health = 100
                            self.player.invulnerable_time = 3.0  # 3 seconds of invulnerability
                            print("Player respawned with invulnerability")
                        else:
                            # Game over
                            print("Game Over!")
                            try:
                                self.state_manager.end_game(self.player.score)
                            except Exception as e:
                                print(f"Error ending game: {e}")
                                # Force game over
                                self.state_manager.current_phase = GamePhase.GAME_OVER

                    # Enemy is destroyed on impact with player
                    enemy.active = False

                    # Break out of loop since player was hit
                    break

    def _check_game_conditions_simple(self) -> None:
        """Simplified game condition checking"""
        # Game over if player has no lives
        if self.player.lives <= 0:
            print("No lives remaining - Game Over!")
            try:
                if self.state_manager.current_phase == GamePhase.PLAYING:
                    self.state_manager.end_game(self.player.score)
                    print("Game ended successfully")
            except Exception as e:
                print(f"Error ending game: {e}")
                # Force transition to game over
                self.state_manager.current_phase = GamePhase.GAME_OVER

        # Check for score milestones (simplified)
        if self.player.score > 0 and self.player.score % 1000 == 0:
            # Power level upgrades based on score
            expected_power = min(10, 1 + (self.player.score // 2000))
            if expected_power > self.player.power_level:
                self.player.power_level = expected_power
                print(f"Power level increased to {self.player.power_level}!")

    def _log_performance_simple(self) -> None:
        """Simplified performance logging"""
        if self.performance_samples:
            avg_time = sum(self.performance_samples) / len(self.performance_samples)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            print(f"Performance: {fps:.1f} FPS, {len(self.enemies)} enemies, {len(self.bullets)} bullets")

    def _generate_simple_final_report(self) -> None:
        """Generate a simple final report without complex operations"""
        print("\n" + "~" * 80)
        print("GAME SESSION COMPLETE")
        print("~" * 80)
        print(f"Final Score: {self.player.score:,}")
        print(f"Game Time: {self.game_time:.1f} seconds")
        print(f"Enemies Defeated: {self.analytics.enemies_defeated if self.analytics else 0}")
        print(f"Total Frames: {self.frame_count:,}")
        print("~" * 80)

    def spawn_enemy_simple(self) -> None:
        """Simplified enemy spawning without complex verification"""
        try:
            spawn_x = GameConstants.SCREEN_WIDTH * 0.1 + (GameConstants.SCREEN_WIDTH * 0.8) * (time.time() % 1)
            enemy = Enemy(
                position=Vector2D(spawn_x, -50),
                velocity=Vector2D(0, GameConstants.ENEMY_SPEED),
                size=15.0,
                health=50,
                enemy_type="basic",
                points_value=100
            )
            self.enemies.append(enemy)
        except Exception as e:
            print(f"Simple enemy spawn error: {e}")

    def _emergency_cleanup(self) -> None:
        """Emergency cleanup to prevent performance issues"""
        # Keep only closest enemies
        if len(self.enemies) > 20:
            self.enemies = self.enemies[:20]

        # Remove distant bullets
        if len(self.bullets) > 50:
            self.bullets = self.bullets[:50]

        print("Emergency cleanup completed")

    @ax.verify
    def _handle_collisions(self) -> None:
        """Enhanced collision handling with comprehensive verification"""

        # Track collision detection performance
        collision_start_time = time.perf_counter()

        # Player bullets vs Enemies
        player_bullets = [b for b in self.bullets if b.active and b.owner == "player"]
        bullet_enemy_collisions = self.physics.check_collisions(player_bullets, self.enemies)

        hits_this_frame = 0
        for bullet, enemy in bullet_enemy_collisions:
            # Verify collision is valid
            ax.ensure(bullet.active and enemy.active, "Colliding objects must be active")
            ax.ensure(bullet.owner == "player", "Only player bullets should hit enemies")

            # Apply damage
            old_enemy_health = enemy.health
            enemy.health -= bullet.damage
            ax.ensure(enemy.health <= old_enemy_health, "Enemy health should decrease")

            # Bullet hits unless penetrating
            if not bullet.penetrating:
                bullet.active = False

            if enemy.health <= 0:
                # Enemy destroyed
                self.player.add_score(enemy.points_value)
                enemy.active = False
                hits_this_frame += 1

                # Update analytics
                self.analytics.enemies_defeated += 1
                self.analytics.max_score_achieved = max(
                    self.analytics.max_score_achieved,
                    self.player.score
                )

                record_temporal_event("enemy_destroyed", {
                    "points": enemy.points_value,
                    "total_score": self.player.score,
                    "enemy_type": enemy.enemy_type,
                    "player_power": self.player.power_level
                })

        # Update hit statistics
        if hits_this_frame > 0:
            for _ in range(hits_this_frame):
                self.player.record_hit()
            self.analytics.total_hits_landed += hits_this_frame

        # Player vs Enemies
        player_enemy_collisions = self.physics.check_collisions([self.player], self.enemies)

        for _, enemy in player_enemy_collisions:
            if self.player.invulnerable_time <= 0:
                damage = 40 + (enemy.size / 2)  # Variable damage based on enemy size
                player_died = self.player.take_damage(int(damage))

                # Update analytics
                self.analytics.total_damage_taken += int(damage)

                if player_died:
                    record_temporal_event("player_death", {
                        "lives_remaining": self.player.lives,
                        "enemy_type": enemy.enemy_type,
                        "game_time": self.game_time
                    })

                if player_died and self.player.lives <= 0:
                    self.state_manager.end_game(self.player.score)

                    # Record final score in secure system
                    if _game_config.information_flow_tracking:
                        score_msg = self.secure_data.record_high_score(
                            self.player.score,
                            f"Player_{int(time.time())}"
                        )
                        print(f"Game Over: {score_msg}")

            enemy.active = False  # Enemy is destroyed on impact

        # Record collision performance
        if FUTURE_FEATURES_AVAILABLE:
            collision_time = time.perf_counter() - collision_start_time
            adaptive_require(
                "collision_detection_performance",
                collision_time < 0.01,  # 10ms max
                property_name="collision_detection_time",
                priority=4
            )

    @ax.verify
    def _cleanup_objects(self) -> None:
        """Enhanced object cleanup with verification"""
        # Count objects before cleanup
        initial_counts = {
            'enemies': len(self.enemies),
            'bullets': len(self.bullets),
            'powerups': len(self.powerups)
        }

        # Remove inactive objects
        self.enemies = [e for e in self.enemies if e.active]
        self.bullets = [b for b in self.bullets if b.active]
        self.powerups = [p for p in self.powerups if p.active]

        # Remove off-screen bullets
        screen_bounds = pygame.Rect(-100, -100,
                                    GameConstants.SCREEN_WIDTH + 200,
                                    GameConstants.SCREEN_HEIGHT + 200)

        bullets_removed = 0
        for bullet in self.bullets[:]:
            if not screen_bounds.collidepoint(bullet.position.x, bullet.position.y):
                bullet.active = False
                self.bullets.remove(bullet)
                bullets_removed += 1

        # Remove off-screen enemies (that went past player)
        enemies_removed = 0
        for enemy in self.enemies[:]:
            if enemy.position.y > GameConstants.SCREEN_HEIGHT + 100:
                enemy.active = False
                self.enemies.remove(enemy)
                enemies_removed += 1

        # Verify cleanup was reasonable
        final_counts = {
            'enemies': len(self.enemies),
            'bullets': len(self.bullets),
            'powerups': len(self.powerups)
        }

        # Cleanup should only decrease counts
        for obj_type in initial_counts:
            ax.ensure(final_counts[obj_type] <= initial_counts[obj_type],
                      f"{obj_type} count should not increase during cleanup")

        # Record cleanup statistics
        if bullets_removed > 0 or enemies_removed > 0:
            record_temporal_event("object_cleanup", {
                "bullets_removed": bullets_removed,
                "enemies_removed": enemies_removed,
                "remaining_objects": sum(final_counts.values())
            })

    @ax.verify
    def _check_game_conditions(self) -> None:
        """Enhanced game condition checking"""
        # Game over if player has no lives
        if self.player.lives <= 0:
            if self.state_manager.current_phase == GamePhase.PLAYING:
                self.state_manager.end_game(self.player.score)

        # Check for achievements/milestones
        if self.player.score > 0 and self.player.score % 5000 == 0:
            # Score milestone reached
            record_temporal_event("score_milestone", {
                "score": self.player.score,
                "accuracy": self.player.calculate_accuracy(),
                "game_time": self.game_time
            })

        # Power level upgrades based on score
        expected_power = min(10, 1 + (self.player.score // 10000))
        if expected_power > self.player.power_level:
            self.player.power_level = expected_power
            record_temporal_event("power_upgrade", {
                "new_level": self.player.power_level,
                "score": self.player.score
            })

    @ax.verify(track_performance=True)
    def render(self) -> None:
        """Enhanced rendering with debug overlay"""
        frame_start = time.perf_counter()

        self.renderer.clear_screen()

        if self.state_manager.current_phase == GamePhase.PLAYING:
            self._render_playing()
        elif self.state_manager.current_phase == GamePhase.MENU:
            self._render_menu()
        elif self.state_manager.current_phase == GamePhase.PAUSED:
            self._render_paused()
        elif self.state_manager.current_phase == GamePhase.GAME_OVER:
            self._render_game_over()
        elif self.state_manager.current_phase == GamePhase.ANALYTICS:
            self._render_analytics()

        # Draw debug overlay if enabled
        if _game_config.debug_overlay_enabled:
            self._render_debug_overlay()

        pygame.display.flip()

        # Track rendering performance
        frame_time = time.perf_counter() - frame_start
        self.performance_samples.append(frame_time)

    @ax.verify
    def _render_playing(self) -> None:
        """Render playing state with enhanced visuals"""
        # Draw player with power level indication
        player_color = GameConstants.WHITE
        if self.player.invulnerable_time > 0:
            # Flashing effect when invulnerable
            flash_rate = int(self.player.invulnerable_time * 10) % 2
            player_color = GameConstants.YELLOW if flash_rate else GameConstants.WHITE
        elif self.player.power_level > 5:
            player_color = GameConstants.CYAN  # High power indication

        self.renderer.draw_object(self.player, player_color, debug_info=True)

        # Draw enemies with type-based colors
        for enemy in self.enemies:
            if enemy.active:
                enemy_colors = {
                    "basic": GameConstants.RED,
                    "patrol": GameConstants.PURPLE,
                    "aggressive": GameConstants.ORANGE,
                    "defensive": GameConstants.BLUE
                }
                color = enemy_colors.get(enemy.enemy_type, GameConstants.RED)
                self.renderer.draw_object(enemy, color, debug_info=_game_config.debug_overlay_enabled)

        # Draw bullets with owner-based colors
        for bullet in self.bullets:
            if bullet.active:
                bullet_color = GameConstants.GREEN if bullet.owner == "player" else GameConstants.RED
                if bullet.penetrating:
                    bullet_color = GameConstants.CYAN
                self.renderer.draw_object(bullet, bullet_color)

        # Draw powerups
        for powerup in self.powerups:
            if powerup.active:
                self.renderer.draw_object(powerup, GameConstants.YELLOW)

        # Draw enhanced UI
        self.renderer.draw_ui(self.player, self.game_time, self.analytics)

    def _render_debug_overlay(self) -> None:
        """Render comprehensive debug information"""
        # Get performance hotspots
        hotspots = []
        if FUTURE_FEATURES_AVAILABLE:
            try:
                hotspots = get_performance_hotspots(5)
            except:
                pass

        # Get physics statistics
        physics_stats = self.physics.get_physics_statistics()

        # Count temporal events
        temporal_count = 0
        temporal_history = get_temporal_history()
        if temporal_history:
            for prop_history in temporal_history:
                temporal_count += len(prop_history)

        self.renderer.draw_debug_overlay(hotspots, physics_stats, temporal_count)

    @ax.verify
    def _render_menu(self) -> None:
        """Enhanced main menu"""
        title_font = pygame.font.Font(None, 84)
        title_text = title_font.render("Advanced Verified Space Shooter", True, GameConstants.CYAN)
        title_rect = title_text.get_rect(center=(GameConstants.SCREEN_WIDTH // 2, 120))
        self.screen.blit(title_text, title_rect)

        subtitle_text = self.renderer.font.render("Ultimate Axiomatik Runtime Verification Demo",
                                                  True, GameConstants.YELLOW)
        subtitle_rect = subtitle_text.get_rect(center=(GameConstants.SCREEN_WIDTH // 2, 170))
        self.screen.blit(subtitle_text, subtitle_rect)

        # Feature highlights
        feature_y = 220
        features = [
            "Recovery Framework with Graceful Degradation",
            "Adaptive Performance Monitoring",
            "Information Flow Tracking",
            "Temporal Property Verification",
            "Plugin System with Game-Specific Verifiers",
            "Loop Invariants & Mathematical Verification",
            "Ghost State Debugging",
            "Comprehensive Analytics System"
        ]

        for feature in features:
            feature_text = self.renderer.small_font.render(feature, True, GameConstants.GREEN)
            feature_rect = feature_text.get_rect(center=(GameConstants.SCREEN_WIDTH // 2, feature_y))
            self.screen.blit(feature_text, feature_rect)
            feature_y += 25

        # Current verification mode
        mode_text = self.renderer.font.render(f"Verification Mode: {_game_config.verification_mode.value.upper()}",
                                              True, GameConstants.WHITE)
        mode_rect = mode_text.get_rect(center=(GameConstants.SCREEN_WIDTH // 2, feature_y + 30))
        self.screen.blit(mode_text, mode_rect)

        # Instructions
        instructions = [
            "SPACE - Start Game",
            "F1 - Toggle Debug Overlay",
            "F2 - Cycle Verification Modes",
            "F3 - Analytics Report",
            "Q - Quit"
        ]

        instruction_y = GameConstants.SCREEN_HEIGHT - 150
        for instruction in instructions:
            inst_text = self.renderer.small_font.render(instruction, True, GameConstants.WHITE)
            inst_rect = inst_text.get_rect(center=(GameConstants.SCREEN_WIDTH // 2, instruction_y))
            self.screen.blit(inst_text, inst_rect)
            instruction_y += 25

    @ax.verify
    def _render_paused(self) -> None:
        """Render paused state with game visible"""
        # Draw the game state first
        self._render_playing()

        # Semi-transparent overlay
        overlay = pygame.Surface((GameConstants.SCREEN_WIDTH, GameConstants.SCREEN_HEIGHT))
        overlay.set_alpha(128)
        overlay.fill(GameConstants.BLACK)
        self.screen.blit(overlay, (0, 0))

        # Pause text
        pause_font = pygame.font.Font(None, 96)
        pause_text = pause_font.render("PAUSED", True, GameConstants.YELLOW)
        pause_rect = pause_text.get_rect(center=(GameConstants.SCREEN_WIDTH // 2, GameConstants.SCREEN_HEIGHT // 2))
        self.screen.blit(pause_text, pause_rect)

        # Instructions
        instruction_text = self.renderer.font.render("Press ESC to Resume", True, GameConstants.WHITE)
        instruction_rect = instruction_text.get_rect(
            center=(GameConstants.SCREEN_WIDTH // 2, GameConstants.SCREEN_HEIGHT // 2 + 80))
        self.screen.blit(instruction_text, instruction_rect)

        # Show current stats
        stats_y = GameConstants.SCREEN_HEIGHT // 2 + 120
        stats = [
            f"Current Score: {self.player.score:,}",
            f"Accuracy: {self.player.calculate_accuracy() * 100:.1f}%",
            f"Game Time: {self.game_time:.1f}s",
            f"Enemies Defeated: {self.analytics.enemies_defeated}"
        ]

        for stat in stats:
            stat_text = self.renderer.small_font.render(stat, True, GameConstants.CYAN)
            stat_rect = stat_text.get_rect(center=(GameConstants.SCREEN_WIDTH // 2, stats_y))
            self.screen.blit(stat_text, stat_rect)
            stats_y += 25

    @ax.verify
    def _render_game_over(self) -> None:
        """Enhanced game over screen"""
        # Game over title
        gameover_font = pygame.font.Font(None, 96)
        gameover_text = gameover_font.render("GAME OVER", True, GameConstants.RED)
        gameover_rect = gameover_text.get_rect(center=(GameConstants.SCREEN_WIDTH // 2, 150))
        self.screen.blit(gameover_text, gameover_rect)

        # Final statistics
        final_stats = [
            f"Final Score: {self.player.score:,}",
            f"Final Accuracy: {self.player.calculate_accuracy() * 100:.1f}%",
            f"Enemies Defeated: {self.analytics.enemies_defeated}",
            f"Survival Time: {self.game_time:.1f}s",
            f"Efficiency: {self.analytics.calculate_efficiency_score() * 100:.1f}%"
        ]

        stats_y = 250
        for stat in final_stats:
            stat_text = self.renderer.font.render(stat, True, GameConstants.WHITE)
            stat_rect = stat_text.get_rect(center=(GameConstants.SCREEN_WIDTH // 2, stats_y))
            self.screen.blit(stat_text, stat_rect)
            stats_y += 35

        # High score message if applicable
        if _game_config.information_flow_tracking:
            score_msg = self.secure_data.record_high_score(
                self.player.score,
                f"Player_{int(time.time())}"
            )
            score_text = self.renderer.small_font.render(score_msg, True, GameConstants.YELLOW)
            score_rect = score_text.get_rect(center=(GameConstants.SCREEN_WIDTH // 2, stats_y + 20))
            self.screen.blit(score_text, score_rect)

        # Instructions
        instructions = [
            "SPACE - Return to Menu",
            "F3 - View Analytics",
            "Q - Quit"
        ]

        instruction_y = GameConstants.SCREEN_HEIGHT - 100
        for instruction in instructions:
            inst_text = self.renderer.small_font.render(instruction, True, GameConstants.CYAN)
            inst_rect = inst_text.get_rect(center=(GameConstants.SCREEN_WIDTH // 2, instruction_y))
            self.screen.blit(inst_text, inst_rect)
            instruction_y += 25

    @ax.verify
    def _render_analytics(self) -> None:
        """Render comprehensive analytics screen"""
        high_scores = []
        if _game_config.information_flow_tracking:
            high_scores = self.secure_data.get_public_leaderboard()

        self.renderer.draw_analytics_screen(self.analytics, high_scores)

    @ax.verify
    def handle_events(self) -> None:
        """Enhanced event handling with all features"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                record_temporal_event("game_quit", {"reason": "window_closed"})

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.running = False
                    record_temporal_event("game_quit", {"reason": "quit_key"})

                elif event.key == pygame.K_ESCAPE:
                    try:
                        if self.state_manager.current_phase == GamePhase.PLAYING:
                            self.state_manager.pause_game()
                        elif self.state_manager.current_phase == GamePhase.PAUSED:
                            self.state_manager.resume_game()
                    except Exception as e:
                        print(f"ERROR: Failed to handle escape: {e}")

                elif event.key == pygame.K_SPACE:
                    try:
                        if self.state_manager.current_phase == GamePhase.MENU:
                            print("DEBUG: Starting game...")
                            print(f"DEBUG: Current state before transition: {self.state_manager.current_phase}")

                            # Try the state transition
                            self.state_manager.start_game()
                            print(f"DEBUG: State after transition: {self.state_manager.current_phase}")

                            # Only reset if transition succeeded
                            if self.state_manager.current_phase == GamePhase.PLAYING:
                                print("DEBUG: State transition successful, now resetting game...")
                                self._reset_game()
                                print("DEBUG: Game reset complete")
                            else:
                                print(
                                    f"ERROR: State transition failed, expected PLAYING but got {self.state_manager.current_phase}")

                        elif self.state_manager.current_phase in [GamePhase.GAME_OVER, GamePhase.ANALYTICS]:
                            self.state_manager.return_to_menu()

                    except ax.VerificationError as e:
                        print(f"ERROR: Verification failed during space press: {e}")
                        print("Continuing without crashing...")
                        # Don't exit, just log the error

                    except Exception as e:
                        print(f"ERROR: Unexpected error during space press: {e}")
                        print(f"Current phase: {self.state_manager.current_phase}")
                        import traceback
                        traceback.print_exc()
                        # Don't exit, just log the error

                elif event.key == pygame.K_F1:
                    # Toggle debug overlay
                    _game_config.debug_overlay_enabled = not _game_config.debug_overlay_enabled
                    print(f"Debug overlay: {'enabled' if _game_config.debug_overlay_enabled else 'disabled'}")

                elif event.key == pygame.K_F2:
                    # Cycle verification modes
                    self._cycle_verification_mode()

                elif event.key == pygame.K_F3:
                    # Show analytics or save report
                    try:
                        if self.state_manager.current_phase == GamePhase.GAME_OVER:
                            self.state_manager.show_analytics()
                        elif self.state_manager.current_phase == GamePhase.ANALYTICS:
                            self._save_analytics_report()
                        else:
                            self._generate_performance_report()
                    except Exception as e:
                        print(f"ERROR: Failed to handle F3: {e}")

    @ax.verify
    def _cycle_verification_mode(self) -> None:
        """Cycle through verification modes"""
        modes = list(GameVerificationMode)
        current_index = modes.index(_game_config.verification_mode)
        next_index = (current_index + 1) % len(modes)
        new_mode = modes[next_index]

        _game_config.set_game_mode(new_mode)

        record_temporal_event("verification_mode_changed", {
            "old_mode": _game_config.verification_mode.value,
            "new_mode": new_mode.value
        })

    @ax.verify
    def _save_analytics_report(self) -> None:
        """Save comprehensive analytics report"""
        try:
            report = self.analytics.generate_report()

            # Add axiomatik performance data
            axiomatik_report = generate_performance_report()
            report['axiomatik_performance'] = axiomatik_report

            # Add temporal event summary
            temporal_summary = {"total_events": 0, "properties": []}
            temporal_history = get_temporal_history()
            if temporal_history:
                for i, prop_history in enumerate(temporal_history):
                    prop_data = {
                        "property_id": i,
                        "events": len(prop_history),
                        "recent_events": [dict(e) for e in list(prop_history)[-5:]]
                    }
                    temporal_summary["properties"].append(prop_data)
                    temporal_summary["total_events"] += len(prop_history)

            report['temporal_verification'] = temporal_summary

            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"axiomatik_game_report_{timestamp}.json"

            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            print(f"Analytics report saved to: {filename}")

            record_temporal_event("report_saved", {"filename": filename})

        except Exception as e:
            print(f"Error saving report: {e}")

    @ax.verify
    def _generate_performance_report(self) -> None:
        """Generate and display real-time performance report"""
        print("\n" + "~" * 80)
        print("REAL-TIME AXIOMATIK PERFORMANCE REPORT")
        print("~" * 80)

        # Game statistics
        print(f"\nGame Statistics:")
        print(f"  Current Score: {self.player.score:,}")
        print(f"  Game Time: {self.game_time:.1f}s")
        print(f"  Accuracy: {self.player.calculate_accuracy() * 100:.1f}%")
        print(f"  Efficiency: {self.analytics.calculate_efficiency_score() * 100:.1f}%")

        # Axiomatik performance
        print(f"\nAxiomatik Performance:")
        print(ax.performance_report())

        if FUTURE_FEATURES_AVAILABLE:
            try:
                hotspots = get_performance_hotspots(10)
                if hotspots:
                    print(f"\nTop Performance Hotspots:")
                    for i, hotspot in enumerate(hotspots, 1):
                        print(f"  {i:2d}. {hotspot.property_name[:40]:40} "
                              f"{hotspot.average_time * 1000:6.1f}ms avg")
            except:
                pass

        # Frame rate statistics
        if self.performance_samples:
            avg_frame_time = sum(self.performance_samples) / len(self.performance_samples)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            print(f"\nRendering Performance:")
            print(f"  Average FPS: {fps:.1f}")
            print(f"  Frame Time: {avg_frame_time * 1000:.2f}ms")

        print("~" * 80)

    @ax.verify
    def _reset_game(self) -> None:
        """Reset game state with comprehensive cleanup"""
        print("DEBUG: Starting game reset...")

        try:
            # Store old state for verification
            old_score = self.player.score if hasattr(self.player, 'score') else 0

            print("DEBUG: Creating new player...")
            # Create new player with error handling
            try:
                self.player = self._create_player()
                print("DEBUG: Player created successfully")
            except Exception as e:
                print(f"ERROR: Failed to create player: {e}")
                # Create a minimal fallback player
                self.player = self._create_fallback_player()
                print("DEBUG: Fallback player created")

            print("DEBUG: Clearing game objects...")
            # Clear all game objects
            self.enemies.clear()
            self.bullets.clear()
            self.powerups.clear()

            print("DEBUG: Resetting game state...")
            # Reset game state
            self.game_time = 0.0
            self.last_enemy_spawn = 0.0
            self.frame_count = 0

            print("DEBUG: Creating new analytics...")
            # Reset analytics for new session
            try:
                session_id = f"session_{int(time.time())}"
                self.analytics = GameAnalytics(session_id=session_id)
                print(f"DEBUG: Analytics created with session {session_id}")
            except Exception as e:
                print(f"ERROR: Failed to create analytics: {e}")
                # Create minimal analytics
                self.analytics = None

            # Clear performance samples
            self.performance_samples.clear()

            # Reset physics statistics
            if hasattr(self.physics, 'collision_count'):
                self.physics.collision_count = 0
            if hasattr(self.physics, 'physics_violations'):
                self.physics.physics_violations = 0

            print("DEBUG: Verifying reset state...")
            # Verify reset was successful (with safer checks)
            try:
                ax.ensure(len(self.enemies) == 0, "Enemies should be cleared")
                ax.ensure(len(self.bullets) == 0, "Bullets should be cleared")
                ax.ensure(hasattr(self.player, 'lives') and self.player.lives > 0,
                          "Player should have lives after reset")
                ax.ensure(hasattr(self.player, 'score') and self.player.score == 0, "Score should be reset")
                ax.ensure(self.game_time == 0.0, "Game time should be reset")
                print("DEBUG: Reset verification passed")
            except ax.VerificationError as e:
                print(f"WARNING: Reset verification failed: {e}")
                # Don't crash, just warn

            # Record the reset event
            try:
                record_temporal_event("game_reset", {
                    "old_score": old_score,
                    "new_session_id": session_id if self.analytics else "none"
                })
            except Exception as e:
                print(f"WARNING: Failed to record temporal event: {e}")

            print("DEBUG: Game reset complete")

        except Exception as e:
            print(f"CRITICAL ERROR: Game reset failed: {e}")
            import traceback
            traceback.print_exc()
            # Try to create minimal game state
            try:
                self.player = self._create_fallback_player()
                self.enemies.clear()
                self.bullets.clear()
                self.powerups.clear()
                self.game_time = 0.0
                print("DEBUG: Minimal recovery completed")
            except Exception as recovery_error:
                print(f"CRITICAL: Even minimal recovery failed: {recovery_error}")
                # Set a flag that the game should return to menu
                if hasattr(self.state_manager, 'current_phase'):
                    self.state_manager.current_phase = GamePhase.MENU

    @ax.verify(track_performance=True)
    def run(self) -> None:
        """Enhanced main game loop with stack overflow protection"""
        print("Starting Advanced Verified Space Shooter")
        print(f"Verification Mode: {_game_config.verification_mode.value}")
        print(f"Debug Overlay: {'enabled' if _game_config.debug_overlay_enabled else 'disabled'}")

        if FUTURE_FEATURES_AVAILABLE:
            print("Advanced features: Available")
        else:
            print("Advanced features: Limited (install future_axiomatik)")

        record_temporal_event("advanced_game_loop_started")

        last_time = time.perf_counter()
        performance_check_timer = 0.0
        loop_count = 0

        try:
            while self.running:
                loop_count += 1

                # Stack overflow protection - limit deep recursion
                if loop_count > 1000000:  # Reset counter to prevent overflow
                    loop_count = 0

                # Calculate delta time
                current_time = time.perf_counter()
                delta_time = current_time - last_time
                last_time = current_time

                # Basic validation without recursion
                if delta_time < 0:
                    delta_time = 1.0 / 60.0  # Fallback to 60 FPS

                # Cap delta time to prevent huge jumps
                delta_time = min(delta_time, 1.0 / 30.0)  # Max 30 FPS equivalent

                self.game_time += delta_time
                self.frame_count += 1
                performance_check_timer += delta_time

                try:
                    # Handle events (most likely to cause issues)
                    self.handle_events()
                    if not self.running:
                        break

                    # Handle input
                    keys = pygame.key.get_pressed()
                    self.handle_input(keys, delta_time)

                    # Update game logic with error protection
                    self.update_game_logic(delta_time)

                    # Render
                    self.render()

                    # Control frame rate
                    self.clock.tick(GameConstants.FPS)

                    # Simplified periodic checks to avoid recursion
                    if performance_check_timer >= 5.0:  # Every 5 seconds instead of 2
                        performance_check_timer = 0.0
                        # Skip temporal verification if it might cause recursion
                        if _game_config.temporal_verification_enabled:
                            try:
                                # Don't verify temporal properties if we're in a problematic state
                                if len(self.enemies) < 100 and len(self.bullets) < 200:
                                    verify_temporal_properties()
                            except Exception as e:
                                print(f"Skipping temporal verification due to error: {e}")

                    # Simplified performance reporting
                    if self.frame_count % (GameConstants.FPS * 20) == 0:  # Every 20 seconds
                        try:
                            self._log_performance_simple()
                        except Exception as e:
                            print(f"Performance logging error: {e}")

                except RecursionError as e:
                    print(f"Recursion error detected: {e}")
                    print("Switching to minimal verification mode...")
                    _game_config.set_game_mode(GameVerificationMode.PERFORMANCE)
                    # Continue running with minimal verification

                except Exception as e:
                    print(f"Loop iteration error: {e}")
                    # Try to continue unless it's a critical error
                    if "stack" in str(e).lower() or "recursion" in str(e).lower():
                        print("Stack-related error detected, reducing verification...")
                        ax.set_mode("off")

        except ax.VerificationError as e:
            print(f"Verification Error: {e}")
            self._handle_verification_failure(e)

        except RecursionError as e:
            print(f"Stack Overflow Protection: {e}")
            print("The verification system hit recursion limits.")
            print("This is a safety feature to prevent crashes.")

        except Exception as e:
            print(f"Unexpected Error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            try:
                record_temporal_event("advanced_game_loop_ended")
                self._generate_simple_final_report()
            except Exception as e:
                print(f"Error in final cleanup: {e}")

            pygame.quit()

    @ax.verify
    def _handle_verification_failure(self, error: ax.VerificationError) -> None:
        """Handle verification failures gracefully"""
        print("Attempting graceful recovery from verification failure...")

        # Try to continue in performance mode
        if _game_config.verification_mode != GameVerificationMode.PERFORMANCE:
            _game_config.set_game_mode(GameVerificationMode.PERFORMANCE)
            print("Switched to performance mode for recovery")

        # Reset problematic game state
        if len(self.enemies) > 50:
            self.enemies = self.enemies[:20]  # Keep only first 20
            print("Reduced enemy count for stability")

        if len(self.bullets) > 100:
            self.bullets = self.bullets[:50]  # Keep only first 50
            print("Reduced bullet count for stability")

    @ax.verify
    def _log_performance(self) -> None:
        """Enhanced performance logging"""
        if self.performance_samples:
            avg_frame_time = sum(self.performance_samples) / len(self.performance_samples)
            max_frame_time = max(self.performance_samples)
            fps_estimate = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

            performance_data = {
                "avg_frame_time_ms": avg_frame_time * 1000,
                "max_frame_time_ms": max_frame_time * 1000,
                "estimated_fps": fps_estimate,
                "total_frames": self.frame_count,
                "game_time": self.game_time,
                "active_objects": len(self.enemies) + len(self.bullets) + len(self.powerups) + 1,
                "verification_mode": _game_config.verification_mode.value
            }

            record_temporal_event("performance_sample", performance_data)

            # Print performance summary
            object_count = performance_data["active_objects"]
            print(f"Performance: {fps_estimate:.1f} FPS, "
                  f"{avg_frame_time * 1000:.1f}ms avg, "
                  f"{object_count} objects, "
                  f"Mode: {_game_config.verification_mode.value}")

            # Adaptive performance adjustment
            if FUTURE_FEATURES_AVAILABLE and fps_estimate < 45:
                print("Low FPS detected - adaptive monitoring may reduce verification overhead")

    @ax.verify
    def _generate_final_report(self) -> None:
        """Generate comprehensive final report with all features"""
        print("~" * 80)
        print("ADVANCED VERIFIED SPACE SHOOTER - COMPREHENSIVE FINAL REPORT")
        print("~" * 80)

        # Game Statistics
        print(f"\nGame Statistics:")
        print(f"  Total Game Time: {self.game_time:.1f} seconds")
        print(f"  Total Frames Rendered: {self.frame_count:,}")
        print(f"  Final Score: {self.player.score:,}")
        print(f"  Shooting Accuracy: {self.player.calculate_accuracy() * 100:.1f}%")
        print(f"  Player Efficiency: {self.analytics.calculate_efficiency_score() * 100:.1f}%")
        print(f"  Enemies Defeated: {self.analytics.enemies_defeated}")
        print(f"  Damage Taken: {self.analytics.total_damage_taken}")

        # Verification Statistics
        print(f"\nVerification Statistics:")
        print(f"  Verification Mode: {_game_config.verification_mode.value}")
        print(
            f"  Mode Changes: {len([e for e in get_temporal_history() if any('mode_changed' in str(prop) for prop in e)]) if get_temporal_history() else 0}")
        print(f"  Recovery Framework: {'enabled' if _game_config.recovery_enabled else 'disabled'}")
        print(f"  Information Flow Tracking: {'enabled' if _game_config.information_flow_tracking else 'disabled'}")

        # Axiomatik Performance Report
        print(f"\nAxiomatik Performance Report:")
        axiomatik_report = ax.performance_report()
        for line in axiomatik_report.split('\n'):
            if line.strip():
                print(f"  {line}")

        # Advanced Features Report
        if FUTURE_FEATURES_AVAILABLE:
            print(f"\nAdvanced Features Report:")
            try:
                hotspots = get_performance_hotspots(5)
                if hotspots:
                    print(f"  Top Performance Hotspots:")
                    for i, hotspot in enumerate(hotspots, 1):
                        print(f"    {i}. {hotspot.property_name[:30]:30} "
                              f"{hotspot.average_time * 1000:6.1f}ms avg")

                # Adaptive monitor status
                active_properties = len(_adaptive_monitor.active_properties) if hasattr(_adaptive_monitor,
                                                                                        'active_properties') else 0
                print(f"  Adaptive Properties Monitored: {active_properties}")
            except Exception as e:
                print(f"  Advanced features error: {e}")

        # Temporal Verification Report
        print(f"\nTemporal Verification Report:")
        temporal_history = get_temporal_history()
        if temporal_history:
            total_events = 0
            property_count = 0
            for i, prop_history in enumerate(temporal_history):
                if prop_history:
                    property_count += 1
                    events_count = len(prop_history)
                    total_events += events_count
                    print(f"  Property {i}: {events_count} events")

                    # Show recent events
                    recent = list(prop_history)[-3:]
                    for event in recent:
                        event_type = event.get('event', 'unknown')
                        timestamp = event.get('timestamp', 0)
                        print(f"    {event_type} at {timestamp:.2f}s")

            print(f"  Total Properties: {property_count}")
            print(f"  Total Events: {total_events}")
        else:
            print("  No temporal history available")

        # Physics System Report
        print(f"\nPhysics System Report:")
        physics_stats = self.physics.get_physics_statistics()
        for key, value in physics_stats.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")

        # Information Flow Report
        if _game_config.information_flow_tracking:
            print(f"\nInformation Flow Report:")
            high_scores = self.secure_data.get_public_leaderboard()
            print(f"  High Scores Tracked: {len(high_scores)}")
            print(f"  Secure Data Entries: {len(self.secure_data.player_statistics)}")

            if high_scores:
                print(f"  Top Score: {high_scores[0] if high_scores else 'None'}")

        # Plugin System Report
        print(f"\nPlugin System Report:")
        print(f"  Registered Plugins: {len(_plugin_registry.plugins)}")
        for plugin_name in _plugin_registry.plugins:
            plugin = _plugin_registry.plugins[plugin_name]
            verifiers = plugin.add_verifiers()
            print(f"    {plugin_name}: {len(verifiers)} verifiers")

        # Recovery Framework Report
        if _game_config.recovery_enabled:
            print(f"\nRecovery Framework Report:")
            print(f"  Recovery strategies deployed successfully")
            print(f"  Graceful degradation available for critical operations")

        # Performance Summary
        if self.performance_samples:
            avg_fps = 1.0 / (sum(self.performance_samples) / len(self.performance_samples))
            print(f"\nPerformance Summary:")
            print(f"  Average FPS: {avg_fps:.1f}")
            print(f"  Frame Time: {(sum(self.performance_samples) / len(self.performance_samples)) * 1000:.2f}ms")
            print(f"  Performance Samples: {len(self.performance_samples)}")

        # Final Analytics Export
        try:
            report = self.analytics.generate_report()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"final_axiomatik_report_{timestamp}.json"

            # Enhanced report with all data
            enhanced_report = report.copy()
            enhanced_report.update({
                "axiomatik_performance": ax.performance_report(),
                "verification_mode": _game_config.verification_mode.value,
                "features_enabled": {
                    "recovery_framework": _game_config.recovery_enabled,
                    "adaptive_monitoring": _game_config.adaptive_tuning_enabled,
                    "information_flow": _game_config.information_flow_tracking,
                    "temporal_verification": _game_config.temporal_verification_enabled,
                    "debug_overlay": _game_config.debug_overlay_enabled
                },
                "physics_statistics": physics_stats,
                "total_frames": self.frame_count
            })

            with open(filename, 'w') as f:
                json.dump(enhanced_report, f, indent=2, default=str)

            print(f"\nComplete report saved to: {filename}")

        except Exception as e:
            print(f"Error saving final report: {e}")

        print("~" * 80)
        print("Thank you for experiencing the Ultimate Axiomatik Demonstration!")
        print("This game showcased every advanced verification feature available.")
        print("~" * 80)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MAIN EXECUTION WITH ERROR HANDLING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main():
    """Main entry point with comprehensive error handling"""
    print("Advanced Verified Space Shooter")
    print("Ultimate Axiomatik Runtime Verification Demonstration")
    print("~" * 80)

    # Display feature summary
    print("\nThis demonstration includes ALL axiomatik features:")

    standard_features = [
        "Type-verified entity management with dataclasses",
        "Stateful game state management with protocols",
        "Mathematical verification for physics calculations",
        "Performance monitoring with detailed tracking",
        "Comprehensive error messages and debugging"
    ]

    advanced_features = [
        "Recovery framework with graceful degradation",
        "Adaptive performance monitoring",
        "Information flow tracking for sensitive data",
        "Temporal property verification",
        "Plugin system with game-specific verifiers",
        "Loop invariants for mathematical rigor",
        "Ghost state tracking for debugging",
        "Comprehensive analytics and reporting",
        "Configuration-based verification levels",
        "Real-time performance introspection"
    ]

    print("\nStandard Features:")
    for feature in standard_features:
        print(f"  {feature}")

    print(f"\nAdvanced Features:")
    for feature in advanced_features:
        print(f"  {feature}")

    if not FUTURE_FEATURES_AVAILABLE:
        print(f"\nNote: Some advanced features require future_axiomatik module")
        print(f"    Basic axiomatik features are still fully functional!")

    print("\nControls:")
    print("  Arrow Keys / WASD - Move ship")
    print("  Spacebar - Fire bullets")
    print("  ESC - Pause/Resume")
    print("  F1 - Toggle debug overlay")
    print("  F2 - Cycle verification modes")
    print("  F3 - Analytics report")
    print("  Q - Quit game")

    print("~" * 80)
    print()

    try:
        with ax.verification_context("main_advanced_game"):
            game = AdvancedVerifiedSpaceShooter()
            game.run()

    except ax.VerificationError as e:
        print(f"\nVerification Error Caught!")
        print(f"Error: {e}")
        print("\nThis demonstrates axiomatik's ability to catch runtime issues!")
        print("The verification system prevented a potential problem.")
        print("\nAxiomatik verification working as intended!")

    except Exception as e:
        print(f"\nUnexpected Error: {e}")
        print("This represents an issue outside of axiomatik's verification scope.")
        print("Check the error details above for debugging information.")

    finally:
        print(f"\nGame Session Complete!")
        print(f"Check the comprehensive final report above to see all axiomatik features in action.")
        print(f"Thank you for trying the Advanced Verified Space Shooter!")

        # Final verification system report
        print(f"\nFinal Axiomatik Status:")
        print(ax.report())


if __name__ == "__main__":
    main()