#!/usr/bin/env python3
"""
Verified Space Shooter - A comprehensive pygame game demonstrating extensive axiomatik features

This game showcases:
- Type-verified entity management with dataclasses
- Stateful game state management with protocols
- Mathematical verification for physics calculations
- Performance monitoring for game loops
- Information flow tracking for scoring
- Temporal property verification for game events
- Error handling with graceful degradation
- Context managers for different game phases

Installation:
    pip install pygame simple_axiomatik

Usage:
    python verified_space_shooter.py

Controls:
    Arrow Keys: Move ship
    Spacebar: Fire bullets
    ESC: Pause/Resume
    Q: Quit game
"""

import pygame
import axiomatik.simple_axiomatik as ax
from axiomatik import record_temporal_event, get_temporal_history, track_sensitive_data, SecurityLabel
import math
import random
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Union
from datetime import datetime
from enum import Enum
from pathlib import Path

# Initialize axiomatik in development mode for comprehensive verification
ax.set_mode("dev")

# Initialize pygame
pygame.init()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# GAME CONSTANTS AND CONFIGURATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class GameConstants:
    """Game constants with verification"""

    # Display constants
    SCREEN_WIDTH: ax.PositiveInt = 800
    SCREEN_HEIGHT: ax.PositiveInt = 600
    FPS: ax.Range[int, 30, 120] = 60

    # Entity constants
    PLAYER_SPEED: ax.PositiveFloat = 300.0
    ENEMY_SPEED: ax.PositiveFloat = 100.0
    BULLET_SPEED: ax.PositiveFloat = 500.0

    # Game mechanics
    MAX_ENEMIES: ax.Range[int, 1, 50] = 10
    MAX_BULLETS: ax.Range[int, 1, 100] = 20
    ENEMY_SPAWN_RATE: ax.Range[float, 0.1, 5.0] = 2.0  # enemies per second

    # Colors (RGB tuples)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    PURPLE = (255, 0, 255)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# VERIFIED ENTITY SYSTEM
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@ax.enable_for_dataclass
@dataclass
class Vector2D:
    """2D vector with mathematical verification"""
    x: Union[int, float] = 0.0
    y: Union[int, float] = 0.0

    def __post_init__(self):
        # Convert to float for consistent math operations
        self.x = float(self.x)
        self.y = float(self.y)
        ax.require(abs(self.x) < 1e6, "X coordinate seems unreasonably large")
        ax.require(abs(self.y) < 1e6, "Y coordinate seems unreasonably large")

    @ax.verify
    def magnitude(self) -> ax.PositiveFloat:
        """Calculate vector magnitude with verification"""
        mag = math.sqrt(self.x * self.x + self.y * self.y)
        ax.ensure(mag >= 0, "Magnitude cannot be negative")
        return mag

    @ax.verify
    def normalize(self) -> 'Vector2D':
        """Normalize vector with zero-division protection"""
        mag = self.magnitude()
        ax.require(mag > 0, "Cannot normalize zero vector")

        result = Vector2D(self.x / mag, self.y / mag)
        ax.ensure(ax.approx_equal(result.magnitude(), 1.0, 0.001), "Normalized vector should have magnitude 1")
        return result

    @ax.verify
    def distance_to(self, other: 'Vector2D') -> ax.PositiveFloat:
        """Calculate distance between vectors"""
        ax.require(isinstance(other, Vector2D), "Other must be Vector2D")

        dx = self.x - other.x
        dy = self.y - other.y
        distance = math.sqrt(dx * dx + dy * dy)

        ax.ensure(distance >= 0, "Distance cannot be negative")
        return distance


@ax.enable_for_dataclass
@dataclass
class GameObject:
    """Base game object with verified properties"""
    position: Vector2D
    velocity: Vector2D
    size: Union[int, float]  # Allow both int and float for size
    health: ax.Range[int, 0, 1000] = 100
    active: bool = True
    creation_time: float = field(default_factory=time.time)

    def __post_init__(self):
        self.size = float(self.size)  # Convert to float
        ax.require(self.size > 0, "Size must be positive")
        ax.require(0 <= self.health <= 1000, "Health must be in valid range")

    @ax.verify
    def update_position(self, delta_time: ax.PositiveFloat) -> None:
        """Update position based on velocity with bounds checking"""
        ax.require(delta_time > 0, "Delta time must be positive")
        ax.require(delta_time < 1.0, "Delta time seems unreasonably large")

        # Store old position for verification
        old_pos = Vector2D(self.position.x, self.position.y)

        # Update position
        self.position.x += self.velocity.x * delta_time
        self.position.y += self.velocity.y * delta_time

        # Verify position change makes sense
        distance_moved = old_pos.distance_to(self.position)
        expected_distance = self.velocity.magnitude() * delta_time
        ax.ensure(ax.approx_equal(distance_moved, expected_distance, 0.1),
                  "Position update should match velocity * time")

    @ax.verify
    def check_bounds(self, width: ax.PositiveInt, height: ax.PositiveInt) -> bool:
        """Check if object is within screen bounds"""
        in_bounds = (0 <= self.position.x <= width and
                     0 <= self.position.y <= height)
        return in_bounds

    @ax.verify
    def collides_with(self, other: 'GameObject') -> bool:
        """Check collision with another object using circle collision"""
        ax.require(isinstance(other, GameObject), "Other must be GameObject")
        ax.require(self.active and other.active, "Both objects must be active")

        distance = self.position.distance_to(other.position)
        collision_distance = self.size + other.size

        colliding = distance <= collision_distance

        # Log collision for temporal verification
        if colliding:
            record_temporal_event("collision_detected", {
                "object1": type(self).__name__,
                "object2": type(other).__name__,
                "distance": distance
            })

        return colliding


@ax.enable_for_dataclass
@dataclass
class Player(GameObject):
    """Player ship with verified movement and shooting"""
    lives: ax.Range[int, 0, 10] = 3
    score: ax.Range[int, 0, 9999999] = 0
    shoot_cooldown: float = 0.0
    invulnerable_time: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        ax.require(self.lives >= 0, "Lives cannot be negative")
        ax.require(self.score >= 0, "Score cannot be negative")

    @ax.verify
    def move(self, direction: Vector2D, speed: ax.PositiveFloat, delta_time: ax.PositiveFloat) -> None:
        """Move player with input validation"""
        ax.require(isinstance(direction, Vector2D), "Direction must be Vector2D")
        ax.require(speed > 0, "Speed must be positive")
        ax.require(delta_time > 0, "Delta time must be positive")

        # Normalize direction to prevent diagonal speed boost
        if direction.magnitude() > 0:
            direction = direction.normalize()

        self.velocity = Vector2D(direction.x * speed, direction.y * speed)
        self.update_position(delta_time)

    @ax.verify
    def can_shoot(self) -> bool:
        """Check if player can shoot based on cooldown"""
        can_shoot = self.shoot_cooldown <= 0
        return can_shoot

    @ax.verify
    def add_score(self, points: ax.PositiveInt) -> None:
        """Add score with overflow protection"""
        ax.require(points > 0, "Points must be positive")

        old_score = self.score
        self.score = min(9999999, self.score + points)

        ax.ensure(self.score >= old_score, "Score should not decrease")
        ax.ensure(self.score <= 9999999, "Score should not overflow")

        # Track sensitive score data
        track_sensitive_data("player_score", self.score, SecurityLabel.CONFIDENTIAL)

    @ax.verify
    def take_damage(self, damage: ax.PositiveInt = 1) -> bool:
        """Take damage and return if player died"""
        if self.invulnerable_time > 0:
            return False  # Invulnerable

        ax.require(damage > 0, "Damage must be positive")

        old_health = self.health
        self.health = max(0, self.health - damage)

        if self.health <= 0:
            self.lives = max(0, self.lives - 1)
            self.health = 100 if self.lives > 0 else 0
            self.invulnerable_time = 2.0  # 2 seconds of invulnerability

            record_temporal_event("player_died", {"lives_remaining": self.lives})
            return True

        ax.ensure(self.health < old_health, "Health should decrease after damage")
        return False


@ax.enable_for_dataclass
@dataclass
class Enemy(GameObject):
    """Enemy ship with verified AI behavior"""
    enemy_type: ax.NonEmpty[str] = "basic"
    points_value: ax.PositiveInt = 100
    attack_cooldown: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        ax.require(self.points_value > 0, "Points value must be positive")
        ax.require(len(self.enemy_type) > 0, "Enemy type cannot be empty")

    @ax.verify
    def update_ai(self, player_pos: Vector2D, delta_time: ax.PositiveFloat) -> None:
        """Update enemy AI with verified behavior"""
        ax.require(isinstance(player_pos, Vector2D), "Player position must be Vector2D")
        ax.require(delta_time > 0, "Delta time must be positive")

        if self.enemy_type == "basic":
            # Basic enemy: move toward player slowly
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

        elif self.enemy_type == "patrol":
            # Patrol enemy: move in patterns
            time_factor = self.creation_time + time.time()
            self.velocity = Vector2D(
                math.sin(time_factor) * GameConstants.ENEMY_SPEED,
                GameConstants.ENEMY_SPEED
            )

        self.update_position(delta_time)


@ax.enable_for_dataclass
@dataclass
class Bullet(GameObject):
    """Bullet projectile with verified physics"""
    owner: ax.NonEmpty[str] = "player"  # "player" or "enemy"
    damage: ax.PositiveInt = 25

    def __post_init__(self):
        super().__post_init__()
        ax.require(self.owner in ["player", "enemy"], "Owner must be 'player' or 'enemy'")
        ax.require(self.damage > 0, "Damage must be positive")


@ax.enable_for_dataclass
@dataclass
class PowerUp(GameObject):
    """Power-up collectible with verified effects"""
    powerup_type: ax.NonEmpty[str] = "health"
    effect_value: ax.PositiveInt = 50
    duration: ax.PositiveFloat = 10.0

    def __post_init__(self):
        super().__post_init__()
        valid_types = ["health", "shield", "double_shot", "speed_boost"]
        ax.require(self.powerup_type in valid_types, f"Power-up type must be one of {valid_types}")
        ax.require(self.effect_value > 0, "Effect value must be positive")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# VERIFIED GAME STATE MANAGEMENT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class GamePhase(Enum):
    MENU = "menu"
    PLAYING = "playing"
    PAUSED = "paused"
    GAME_OVER = "game_over"
    HIGH_SCORES = "high_scores"


@ax.stateful(initial="menu")
class GameStateManager:
    """Game state management with verified transitions"""

    def __init__(self):
        self.current_phase = GamePhase.MENU
        self.transition_time = time.time()
        self.state_history = []

    @ax.state("menu", "playing")
    def start_game(self) -> None:
        """Start new game"""
        record_temporal_event("game_started")
        self.current_phase = GamePhase.PLAYING
        self.transition_time = time.time()
        self._log_state_change("start_game")

    @ax.state("playing", "paused")
    def pause_game(self) -> None:
        """Pause current game"""
        record_temporal_event("game_paused")
        self.current_phase = GamePhase.PAUSED
        self.transition_time = time.time()
        self._log_state_change("pause_game")

    @ax.state("paused", "playing")
    def resume_game(self) -> None:
        """Resume paused game"""
        record_temporal_event("game_resumed")
        self.current_phase = GamePhase.PLAYING
        self.transition_time = time.time()
        self._log_state_change("resume_game")

    @ax.state(["playing", "paused"], "game_over")
    def end_game(self, final_score: ax.Range[int, 0, 9999999]) -> None:
        """End current game"""
        ax.require(final_score >= 0, "Final score cannot be negative")
        record_temporal_event("game_ended", {"final_score": final_score})
        self.current_phase = GamePhase.GAME_OVER
        self.transition_time = time.time()
        self._log_state_change("end_game")

    @ax.state("game_over", "high_scores")
    def show_high_scores(self) -> None:
        """Show high score screen"""
        self.current_phase = GamePhase.HIGH_SCORES
        self.transition_time = time.time()
        self._log_state_change("show_high_scores")

    @ax.state(["game_over", "high_scores"], "menu")
    def return_to_menu(self) -> None:
        """Return to main menu"""
        record_temporal_event("returned_to_menu")
        self.current_phase = GamePhase.MENU
        self.transition_time = time.time()
        self._log_state_change("return_to_menu")

    @ax.verify
    def _log_state_change(self, action: ax.NonEmpty[str]) -> None:
        """Log state changes for temporal verification"""
        entry = {
            "action": action,
            "phase": self.current_phase.value,
            "timestamp": time.time()
        }
        self.state_history.append(entry)

        # Keep history bounded
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-50:]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# VERIFIED PHYSICS AND COLLISION SYSTEMS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class PhysicsSystem:
    """Physics system with mathematical verification"""

    @ax.verify(track_performance=True)
    def update_objects(self, objects: ax.NonEmpty[List[GameObject]],
                       delta_time: ax.PositiveFloat) -> None:
        """Update all objects with verified physics"""
        ax.require(len(objects) > 0, "Object list cannot be empty")
        ax.require(delta_time > 0, "Delta time must be positive")
        ax.require(delta_time < 1.0, "Delta time seems unreasonably large")

        with ax.verification_context("physics_update"):
            for obj in objects:
                if obj.active:
                    obj.update_position(delta_time)

    @ax.verify(track_performance=True)
    def check_collisions(self, group1: List[GameObject],
                         group2: List[GameObject]) -> List[Tuple[GameObject, GameObject]]:
        """Check collisions between two groups with O(n*m) verification"""
        collisions = []

        with ax.verification_context("collision_detection"):
            for obj1 in group1:
                if not obj1.active:
                    continue

                for obj2 in group2:
                    if not obj2.active:
                        continue

                    if obj1.collides_with(obj2):
                        collisions.append((obj1, obj2))

        # Verify collision results
        ax.ensure(len(collisions) <= len(group1) * len(group2),
                  "Cannot have more collisions than object pairs")

        for obj1, obj2 in collisions:
            ax.ensure(obj1.active and obj2.active, "Colliding objects should be active")

        return collisions

    @ax.verify
    def apply_screen_boundaries(self, obj: GameObject,
                                width: ax.PositiveInt, height: ax.PositiveInt) -> None:
        """Apply screen boundary constraints"""
        ax.require(width > 0 and height > 0, "Screen dimensions must be positive")

        # Clamp position to screen bounds with size consideration
        half_size = obj.size / 2
        obj.position.x = max(half_size, min(width - half_size, obj.position.x))
        obj.position.y = max(half_size, min(height - half_size, obj.position.y))

        ax.ensure(obj.check_bounds(width, height), "Object should be within bounds after clamping")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# VERIFIED RENDERING SYSTEM
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RenderSystem:
    """Rendering system with bounds checking and performance tracking"""

    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

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
    def draw_object(self, obj: GameObject, color: Tuple[int, int, int]) -> None:
        """Draw game object with bounds checking"""
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

        # Draw health bar for damaged objects
        if hasattr(obj, 'health') and obj.health < 100:
            self._draw_health_bar(obj)

    @ax.verify
    def _draw_health_bar(self, obj: GameObject) -> None:
        """Draw health bar above object"""
        if not hasattr(obj, 'health'):
            return

        bar_width = int(obj.size * 2)
        bar_height = 4
        bar_x = int(obj.position.x - bar_width // 2)
        bar_y = int(obj.position.y - obj.size - 10)

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
    def draw_ui(self, player: Player, game_time: ax.PositiveFloat) -> None:
        """Draw UI elements with verification"""
        ax.require(isinstance(player, Player), "Player must be Player object")
        ax.require(game_time >= 0, "Game time cannot be negative")

        # Score
        score_text = self.font.render(f"Score: {player.score}", True, GameConstants.WHITE)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font.render(f"Lives: {player.lives}", True, GameConstants.WHITE)
        self.screen.blit(lives_text, (10, 50))

        # Health
        health_text = self.font.render(f"Health: {player.health}", True, GameConstants.WHITE)
        self.screen.blit(health_text, (10, 90))

        # Game time
        time_text = self.small_font.render(f"Time: {game_time:.1f}s", True, GameConstants.WHITE)
        self.screen.blit(time_text, (10, 130))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# VERIFIED GAME MANAGER
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class VerifiedSpaceShooter:
    """Main game class with comprehensive verification"""

    def __init__(self):
        # Initialize pygame display
        self.screen = pygame.display.set_mode((GameConstants.SCREEN_WIDTH, GameConstants.SCREEN_HEIGHT))
        pygame.display.set_caption("Verified Space Shooter - Axiomatik Demo")
        self.clock = pygame.time.Clock()

        # Game systems
        self.state_manager = GameStateManager()
        self.physics = PhysicsSystem()
        self.renderer = RenderSystem(self.screen)

        # Game entities
        self.player = self._create_player()
        self.enemies: List[Enemy] = []
        self.bullets: List[Bullet] = []
        self.powerups: List[PowerUp] = []

        # Game state
        self.game_time = 0.0
        self.last_enemy_spawn = 0.0
        self.running = True

        # Performance tracking
        self.frame_count = 0
        self.performance_samples = []

        # Simple event tracking for verification
        self.event_counter = {}

        # Add temporal properties for event tracking
        from axiomatik import add_temporal_property, AlwaysProperty, EventuallyProperty

        # Add property to track that game events are always valid
        always_valid_events = AlwaysProperty(
            "game_events_valid",
            lambda event: event.get('data', {}) is not None
        )
        add_temporal_property(always_valid_events)

        # Add property to track that enemies eventually get destroyed
        eventually_enemy_destroyed = EventuallyProperty(
            "enemies_destroyed",
            lambda history: any(e.get('event') == 'enemy_destroyed' for e in history),
            timeout=60.0
        )
        add_temporal_property(eventually_enemy_destroyed)

        record_temporal_event("game_initialized")

    def _record_game_event(self, event_name: str, data=None):
        """Helper to record events both temporally and for counting"""
        record_temporal_event(event_name, data)
        self.event_counter[event_name] = self.event_counter.get(event_name, 0) + 1

    @ax.verify
    def _create_player(self) -> Player:
        """Create verified player object"""
        player_pos = Vector2D(
            float(GameConstants.SCREEN_WIDTH // 2),
            float(GameConstants.SCREEN_HEIGHT - 100)
        )

        player = Player(
            position=player_pos,
            velocity=Vector2D(0, 0),
            size=20.0,
            health=100,
            lives=3,
            score=0
        )

        ax.ensure(player.position.x >= 0, "Player X position should be valid")
        ax.ensure(player.position.y >= 0, "Player Y position should be valid")
        ax.ensure(player.lives > 0, "Player should start with lives")

        return player

    @ax.verify
    def spawn_enemy(self) -> None:
        """Spawn enemy with verification"""

        if len(self.enemies) >= GameConstants.MAX_ENEMIES:
            return

        # Random spawn position at top of screen
        spawn_x = random.uniform(50.0, float(GameConstants.SCREEN_WIDTH - 50))
        spawn_pos = Vector2D(spawn_x, -50)

        enemy_types = ["basic", "patrol"]
        enemy_type = random.choice(enemy_types)

        enemy = Enemy(
            position=spawn_pos,
            velocity=Vector2D(0, GameConstants.ENEMY_SPEED),
            size=15.0,
            health=50,
            enemy_type=enemy_type,
            points_value=100 if enemy_type == "basic" else 200
        )

        self.enemies.append(enemy)
        record_temporal_event("enemy_spawned", {"type": enemy_type, "count": len(self.enemies)})

    @ax.verify
    def handle_input(self, keys, delta_time: ax.PositiveFloat) -> None:
        """Handle player input with verification"""
        ax.require(hasattr(keys, '__getitem__'), "Keys must be indexable (pygame key sequence)")
        ax.require(delta_time > 0, "Delta time must be positive")

        if self.state_manager.current_phase != GamePhase.PLAYING:
            return

        # Movement
        direction = Vector2D(0, 0)
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            direction.x -= 1
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            direction.x += 1
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            direction.y -= 1
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            direction.y += 1

        # Apply movement
        if direction.magnitude() > 0:
            self.player.move(direction, GameConstants.PLAYER_SPEED, delta_time)

        # Keep player on screen
        self.physics.apply_screen_boundaries(
            self.player,
            GameConstants.SCREEN_WIDTH,
            GameConstants.SCREEN_HEIGHT
        )

        # Shooting
        if keys[pygame.K_SPACE] and self.player.can_shoot():
            self.create_bullet()

    @ax.verify
    def create_bullet(self) -> None:
        """Create player bullet with verification"""
        if len(self.bullets) >= GameConstants.MAX_BULLETS:
            return

        bullet_pos = Vector2D(
            self.player.position.x,
            self.player.position.y - self.player.size
        )

        bullet = Bullet(
            position=bullet_pos,
            velocity=Vector2D(0, -GameConstants.BULLET_SPEED),
            size=3.0,
            owner="player",
            damage=25
        )

        self.bullets.append(bullet)
        self.player.shoot_cooldown = 0.2  # 200ms cooldown

        self._record_game_event("bullet_fired", {"owner": "player"})

    @ax.verify(track_performance=True)
    def update_game_logic(self, delta_time: ax.PositiveFloat) -> None:
        """Update all game logic with verification"""
        ax.require(delta_time > 0, "Delta time must be positive")

        if self.state_manager.current_phase != GamePhase.PLAYING:
            return

        with ax.verification_context("game_logic_update"):
            # Update cooldowns
            self.player.shoot_cooldown = max(0, self.player.shoot_cooldown - delta_time)
            self.player.invulnerable_time = max(0, self.player.invulnerable_time - delta_time)

            # Update physics for all objects
            all_objects = [self.player] + self.enemies + self.bullets + self.powerups
            active_objects = [obj for obj in all_objects if obj.active]

            if active_objects:
                self.physics.update_objects(active_objects, delta_time)

            # Update enemy AI
            for enemy in self.enemies:
                if enemy.active:
                    enemy.update_ai(self.player.position, delta_time)

            # Spawn enemies
            if self.game_time - self.last_enemy_spawn > (1.0 / GameConstants.ENEMY_SPAWN_RATE):
                self.spawn_enemy()
                self.last_enemy_spawn = self.game_time

            # Handle collisions
            self._handle_collisions()

            # Remove inactive objects
            self._cleanup_objects()

            # Check win/lose conditions
            self._check_game_conditions()

    @ax.verify
    def _handle_collisions(self) -> None:
        """Handle all collision detection and response"""
        # Bullets vs Enemies
        bullet_enemy_collisions = self.physics.check_collisions(
            [b for b in self.bullets if b.owner == "player"],
            self.enemies
        )

        for bullet, enemy in bullet_enemy_collisions:
            # Damage enemy
            enemy.health -= bullet.damage
            bullet.active = False

            if enemy.health <= 0:
                # Enemy destroyed
                self.player.add_score(enemy.points_value)
                enemy.active = False
                self._record_game_event("enemy_destroyed", {
                    "points": enemy.points_value,
                    "total_score": self.player.score
                })

        # Player vs Enemies
        player_enemy_collisions = self.physics.check_collisions([self.player], self.enemies)

        for _, enemy in player_enemy_collisions:
            if self.player.invulnerable_time <= 0:
                player_died = self.player.take_damage(50)
                if player_died:
                    self._record_game_event("player_died", {"lives_remaining": self.player.lives})
                enemy.active = False  # Enemy is destroyed on impact

                if player_died and self.player.lives <= 0:
                    self.state_manager.end_game(self.player.score)

    @ax.verify
    def _cleanup_objects(self) -> None:
        """Remove inactive or off-screen objects"""
        # Remove inactive objects
        self.enemies = [e for e in self.enemies if e.active]
        self.bullets = [b for b in self.bullets if b.active]
        self.powerups = [p for p in self.powerups if p.active]

        # Remove off-screen bullets
        for bullet in self.bullets[:]:  # Create copy for safe iteration
            if bullet.position.y < -50 or bullet.position.y > GameConstants.SCREEN_HEIGHT + 50:
                bullet.active = False

        # Remove off-screen enemies (that went past player)
        for enemy in self.enemies[:]:
            if enemy.position.y > GameConstants.SCREEN_HEIGHT + 50:
                enemy.active = False

    @ax.verify
    def _check_game_conditions(self) -> None:
        """Check for win/lose conditions"""
        # Game over if player has no lives
        if self.player.lives <= 0:
            if self.state_manager.current_phase == GamePhase.PLAYING:
                self.state_manager.end_game(self.player.score)

    @ax.verify(track_performance=True)
    def render(self) -> None:
        """Render all game elements with verification"""
        frame_start = time.perf_counter()

        self.renderer.clear_screen()

        if self.state_manager.current_phase == GamePhase.PLAYING:
            # Draw player (with invulnerability flashing)
            player_color = GameConstants.WHITE
            if self.player.invulnerable_time > 0 and int(self.player.invulnerable_time * 10) % 2:
                player_color = GameConstants.YELLOW
            self.renderer.draw_object(self.player, player_color)

            # Draw enemies
            for enemy in self.enemies:
                if enemy.active:
                    enemy_color = GameConstants.RED if enemy.enemy_type == "basic" else GameConstants.PURPLE
                    self.renderer.draw_object(enemy, enemy_color)

            # Draw bullets
            for bullet in self.bullets:
                if bullet.active:
                    bullet_color = GameConstants.GREEN if bullet.owner == "player" else GameConstants.RED
                    self.renderer.draw_object(bullet, bullet_color)

            # Draw powerups
            for powerup in self.powerups:
                if powerup.active:
                    self.renderer.draw_object(powerup, GameConstants.BLUE)

            # Draw UI
            self.renderer.draw_ui(self.player, self.game_time)

        elif self.state_manager.current_phase == GamePhase.MENU:
            self._render_menu()


        elif self.state_manager.current_phase == GamePhase.PAUSED:
            # Draw game state first WITHOUT recursive call
            # Copy the PLAYING rendering logic here instead
            player_color = GameConstants.WHITE
            if self.player.invulnerable_time > 0 and int(self.player.invulnerable_time * 10) % 2:
                player_color = GameConstants.YELLOW

            self.renderer.draw_object(self.player, player_color)

            # Draw enemies
            for enemy in self.enemies:
                if enemy.active:
                    enemy_color = GameConstants.RED if enemy.enemy_type == "basic" else GameConstants.PURPLE
                    self.renderer.draw_object(enemy, enemy_color)

            # Draw bullets
            for bullet in self.bullets:
                if bullet.active:
                    bullet_color = GameConstants.GREEN if bullet.owner == "player" else GameConstants.RED
                    self.renderer.draw_object(bullet, bullet_color)

            # Draw UI
            self.renderer.draw_ui(self.player, self.game_time)

            # Then draw pause overlay
            self._render_pause_overlay()

        elif self.state_manager.current_phase == GamePhase.GAME_OVER:
            self._render_game_over()

        pygame.display.flip()

        # Track rendering performance
        frame_time = time.perf_counter() - frame_start
        self.performance_samples.append(frame_time)

        if len(self.performance_samples) > 60:  # Keep last 60 frames
            self.performance_samples = self.performance_samples[-60:]

    @ax.verify
    def _render_menu(self) -> None:
        """Render main menu"""
        title_text = pygame.font.Font(None, 72).render("Verified Space Shooter", True, GameConstants.WHITE)
        title_rect = title_text.get_rect(center=(GameConstants.SCREEN_WIDTH // 2, 150))
        self.screen.blit(title_text, title_rect)

        subtitle_text = self.renderer.font.render("Axiomatik Runtime Verification Demo", True, GameConstants.YELLOW)
        subtitle_rect = subtitle_text.get_rect(center=(GameConstants.SCREEN_WIDTH // 2, 200))
        self.screen.blit(subtitle_text, subtitle_rect)

        instructions = [
            "Press SPACE to Start",
            "",
            "Controls:",
            "Arrow Keys / WASD - Move",
            "SPACE - Shoot",
            "ESC - Pause",
            "Q - Quit"
        ]

        y_offset = 280
        for instruction in instructions:
            if instruction:  # Skip empty lines
                text = self.renderer.small_font.render(instruction, True, GameConstants.WHITE)
                text_rect = text.get_rect(center=(GameConstants.SCREEN_WIDTH // 2, y_offset))
                self.screen.blit(text, text_rect)
            y_offset += 30

    @ax.verify
    def _render_pause_overlay(self) -> None:
        """Render pause overlay"""
        # Semi-transparent overlay
        overlay = pygame.Surface((GameConstants.SCREEN_WIDTH, GameConstants.SCREEN_HEIGHT))
        overlay.set_alpha(128)
        overlay.fill(GameConstants.BLACK)
        self.screen.blit(overlay, (0, 0))

        # Pause text
        pause_text = pygame.font.Font(None, 72).render("PAUSED", True, GameConstants.WHITE)
        pause_rect = pause_text.get_rect(center=(GameConstants.SCREEN_WIDTH // 2, GameConstants.SCREEN_HEIGHT // 2))
        self.screen.blit(pause_text, pause_rect)

        instruction_text = self.renderer.font.render("Press ESC to Resume", True, GameConstants.WHITE)
        instruction_rect = instruction_text.get_rect(
            center=(GameConstants.SCREEN_WIDTH // 2, GameConstants.SCREEN_HEIGHT // 2 + 100))
        self.screen.blit(instruction_text, instruction_rect)

    @ax.verify
    def _render_game_over(self) -> None:
        """Render game over screen"""
        # Game over text
        game_over_text = pygame.font.Font(None, 72).render("GAME OVER", True, GameConstants.RED)
        game_over_rect = game_over_text.get_rect(center=(GameConstants.SCREEN_WIDTH // 2, 200))
        self.screen.blit(game_over_text, game_over_rect)

        # Final score
        score_text = self.renderer.font.render(f"Final Score: {self.player.score}", True, GameConstants.WHITE)
        score_rect = score_text.get_rect(center=(GameConstants.SCREEN_WIDTH // 2, 280))
        self.screen.blit(score_text, score_rect)

        # Instructions
        restart_text = self.renderer.small_font.render("Press SPACE to Return to Menu", True, GameConstants.WHITE)
        restart_rect = restart_text.get_rect(center=(GameConstants.SCREEN_WIDTH // 2, 350))
        self.screen.blit(restart_text, restart_rect)

        quit_text = self.renderer.small_font.render("Press Q to Quit", True, GameConstants.WHITE)
        quit_rect = quit_text.get_rect(center=(GameConstants.SCREEN_WIDTH // 2, 380))
        self.screen.blit(quit_text, quit_rect)

    @ax.verify
    def handle_events(self) -> None:
        """Handle pygame events with verification"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                record_temporal_event("game_quit", {"reason": "window_closed"})

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.running = False
                    record_temporal_event("game_quit", {"reason": "quit_key"})

                elif event.key == pygame.K_ESCAPE:
                    if self.state_manager.current_phase == GamePhase.PLAYING:
                        self.state_manager.pause_game()
                    elif self.state_manager.current_phase == GamePhase.PAUSED:
                        self.state_manager.resume_game()

                elif event.key == pygame.K_SPACE:
                    if self.state_manager.current_phase == GamePhase.MENU:
                        self.state_manager.start_game()
                        self._reset_game()
                    elif self.state_manager.current_phase == GamePhase.GAME_OVER:
                        self.state_manager.return_to_menu()

    @ax.verify
    def _reset_game(self) -> None:
        """Reset game state for new game"""
        self.player = self._create_player()
        self.enemies.clear()
        self.bullets.clear()
        self.powerups.clear()
        self.game_time = 0.0
        self.last_enemy_spawn = 0.0
        self.frame_count = 0

        record_temporal_event("game_reset")

    @ax.verify(track_performance=True)
    def run(self) -> None:
        """Main game loop with comprehensive verification"""
        record_temporal_event("game_loop_started")

        last_time = time.perf_counter()

        try:
            while self.running:

                # First frame
                if self.frame_count == 1:
                    self._record_game_event("game_loop_started", {})

                # Calculate delta time
                current_time = time.perf_counter()
                delta_time = current_time - last_time
                last_time = current_time

                ax.require(delta_time >= 0, "Delta time cannot be negative")

                # Cap delta time to prevent huge jumps
                delta_time = min(delta_time, 1.0 / 60.0)  # Max 30 FPS equivalent

                self.game_time += delta_time
                self.frame_count += 1

                # Handle events
                self.handle_events()

                # Handle input
                keys = pygame.key.get_pressed()
                self.handle_input(keys, delta_time)

                # Update game logic
                self.update_game_logic(delta_time)

                # Render
                self.render()

                # Control frame rate
                self.clock.tick(GameConstants.FPS)

                # Performance reporting every 5 seconds
                if self.frame_count % (GameConstants.FPS * 5) == 0:
                    self._log_performance()

        except Exception as e:
            record_temporal_event("game_error", {"error": str(e)})
            print(f"Game error: {e}")
            raise

        finally:
            record_temporal_event("game_loop_ended")
            self._record_game_event("game_loop_ended", {"reason": "normal_exit"})
            self._generate_final_report()
            pygame.quit()

    @ax.verify
    def _log_performance(self) -> None:
        """Log performance metrics"""
        if self.performance_samples:
            avg_frame_time = sum(self.performance_samples) / len(self.performance_samples)
            max_frame_time = max(self.performance_samples)
            fps_estimate = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

            performance_data = {
                "avg_frame_time_ms": avg_frame_time * 1000,
                "max_frame_time_ms": max_frame_time * 1000,
                "estimated_fps": fps_estimate,
                "total_frames": self.frame_count,
                "game_time": self.game_time
            }

            record_temporal_event("performance_sample", performance_data)

            print(f"Performance: {fps_estimate:.1f} FPS, {avg_frame_time * 1000:.1f}ms avg frame time")

    @ax.verify
    def _generate_final_report(self) -> None:
        """Generate comprehensive final report"""
        print("\n" + "~" * 80)
        print("VERIFIED SPACE SHOOTER - FINAL REPORT")
        print("~" * 80)

        print(f"\nGame Statistics:")
        print(f"  Total Game Time: {self.game_time:.1f} seconds")
        print(f"  Total Frames: {self.frame_count}")
        print(f"  Final Score: {self.player.score}")
        print(f"  Enemies Defeated: {self.player.score // 100}")  # Rough estimate

        print(f"\nAxiomatik Performance Report:")
        print(ax.performance_report())

        print(f"\nAxiomatik System Status:")
        print(ax.report())

        print(f"\nTemporal Events Summary:")
        temporal_history = get_temporal_history()
        if temporal_history:
            total_events = 0
            for i, prop_history in enumerate(temporal_history):
                if prop_history:
                    print(f"  Property {i}: {len(prop_history)} events recorded")
                    total_events += len(prop_history)

                    # Show sample of recent events
                    recent_events = list(prop_history)[-3:]  # Last 3 events
                    for event in recent_events:
                        event_type = event.get('event', 'unknown')
                        timestamp = event.get('timestamp', 0)
                        data = event.get('data', {})
                        print(f"    - {event_type} at {timestamp:.2f}s: {data}")

            print(f"  Total temporal events: {total_events}")
        else:
            print("  No temporal history available")

        print(f"\nGame Event Summary:")
        if hasattr(self, 'event_counter') and self.event_counter:
            print(f"  Events recorded: {sum(self.event_counter.values())}")
            for event_name, count in sorted(self.event_counter.items()):
                print(f"    {event_name}: {count}")
        else:
            print("  No game events recorded")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MAIN EXECUTION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main():
    """Main entry point with error handling"""
    print("Verified Space Shooter - Axiomatik Runtime Verification Demo")
    print("~" * 80)
    print("This game demonstrates extensive axiomatik verification features:")
    print("- Type-verified entity management with dataclasses")
    print("- Stateful game state management with protocols")
    print("- Mathematical verification for physics calculations")
    print("- Performance monitoring for game loops")
    print("- Information flow tracking for scoring")
    print("- Temporal property verification for game events")
    print("- Error handling with graceful degradation")
    print("~" * 80)
    print()

    try:
        with ax.verification_context("main_game"):
            game = VerifiedSpaceShooter()
            game.run()

    except ax.VerificationError as e:
        print(f"Verification Error: {e}")
        print("The game encountered a verification failure.")
        print("This demonstrates axiomatik catching runtime issues!")

    except Exception as e:
        print(f"Unexpected Error: {e}")
        print("An unexpected error occurred.")

    finally:
        print("\nThank you for trying Verified Space Shooter!")
        print("Check the final report above to see axiomatik verification statistics.")


if __name__ == "__main__":
    main()