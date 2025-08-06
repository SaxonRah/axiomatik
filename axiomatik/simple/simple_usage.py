#!/usr/bin/env python3
"""
Simple Axiomatik Complete Usage Examples

Real-world examples showing how to use Simple Axiomatik in practical scenarios.
These examples demonstrate patterns you can adapt for your own projects.
"""

import simple_axiomatik as ax
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Union
from datetime import datetime

# Set mode for examples
ax.set_mode("dev")

print("Simple Axiomatik Complete Usage Examples")
print("~" * 80)
print("Real-world patterns you can adapt for your projects")
print()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# EXAMPLE 1: WEB API INPUT VALIDATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print("EXAMPLE 1: Web API Input Validation")
print("~" * 80)
print("Validate API requests with clear error messages")
print()


@ax.enable_for_dataclass
@dataclass
class CreateUserRequest:
    """API request for creating a user with automatic validation"""
    username: ax.NonEmpty[str]
    email: ax.NonEmpty[str]
    age: ax.Range[int, 13, 120]  # Must be teenager or older
    password: ax.NonEmpty[str]
    full_name: ax.NonEmpty[str]

    def __post_init__(self):
        # Additional business logic validation
        ax.require("@" in self.email, "Email must contain @ symbol")
        ax.require(len(self.password) >= 8, "Password must be at least 8 characters")
        ax.require(len(self.username) >= 3, "Username must be at least 3 characters")
        ax.require(" " in self.full_name.strip(), "Full name must contain first and last name")


@ax.enable_for_dataclass
@dataclass
class UpdateUserRequest:
    """API request for updating user information"""
    user_id: ax.PositiveInt
    email: Optional[ax.NonEmpty[str]] = None
    full_name: Optional[ax.NonEmpty[str]] = None
    age: Optional[ax.Range[int, 13, 120]] = None


@ax.checked
def create_user(request: CreateUserRequest) -> Dict[str, any]:
    """Create a new user with comprehensive validation"""

    # Business logic validation beyond types
    ax.require(not _username_exists(request.username), "Username already taken")
    ax.require(_is_valid_email_domain(request.email), "Email domain not allowed")

    # Simulate user creation
    user_id = abs(hash(request.username)) % 10000

    result = {
        "user_id": user_id,
        "username": request.username,
        "email": request.email,
        "full_name": request.full_name,
        "status": "created",
        "created_at": datetime.now().isoformat()
    }

    ax.ensure("user_id" in result, "User ID must be generated")
    ax.ensure(result["user_id"] > 0, "User ID must be positive")

    return result


@ax.checked
def update_user(request: UpdateUserRequest) -> Dict[str, any]:
    """Update user information with validation"""

    ax.require(_user_exists(request.user_id), "User not found")

    # Simulate update logic
    updates = {}
    if request.email:
        ax.require(_is_valid_email_domain(request.email), "Email domain not allowed")
        updates["email"] = request.email
    if request.full_name:
        ax.require(" " in request.full_name.strip(), "Full name must contain first and last name")
        updates["full_name"] = request.full_name
    if request.age:
        updates["age"] = request.age

    ax.require(len(updates) > 0, "At least one field must be updated")

    return {
        "user_id": request.user_id,
        "updates": updates,
        "status": "updated",
        "updated_at": datetime.now().isoformat()
    }


# Helper functions for business logic
def _username_exists(username: str) -> bool:
    """Check if username already exists (simulation)"""
    existing_users = ["admin", "root", "test", "user"]
    return username.lower() in existing_users


def _user_exists(user_id: int) -> bool:
    """Check if user exists (simulation)"""
    return 1 <= user_id <= 9999


def _is_valid_email_domain(email: str) -> bool:
    """Check if email domain is allowed (simulation)"""
    blocked_domains = ["tempmail.com", "10minutemail.com"]
    domain = email.split("@")[-1].lower()
    return domain not in blocked_domains


# Test the API validation
print("Testing API validation:")
try:
    # Valid request
    request = CreateUserRequest(
        username="alice123",
        email="alice@company.com",
        age=25,
        password="secure123password",
        full_name="Alice Johnson"
    )
    result = create_user(request)
    print(f"  +++ - User created: {result['username']} (ID: {result['user_id']})")

    # Valid update
    update_req = UpdateUserRequest(user_id=1234, email="alice.new@company.com")
    update_result = update_user(update_req)
    print(f"  +++ - User updated: {update_result['status']}")

except ax.VerificationError as e:
    print(f"  XXX - Validation failed: {str(e).split('Message: ')[-1]}")

print()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# EXAMPLE 2: FINANCIAL CALCULATIONS WITH PRECISION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print("EXAMPLE 2: Financial Calculations")
print("~" * 80)
print("Precise financial calculations with validation")
print()

from decimal import Decimal, ROUND_HALF_UP


@ax.verify(track_performance=True)
def calculate_compound_interest(
        principal: ax.Positive[float],
        annual_rate: ax.Range[float, 0.0, 0.5],  # Max 50% APR
        years: ax.Range[int, 1, 100],
        compound_frequency: ax.Range[int, 1, 365] = 12  # Monthly by default
) -> Dict[str, float]:
    """Calculate compound interest with comprehensive validation"""

    ax.require(principal > 0, "Principal must be positive")
    ax.require(0 <= annual_rate <= 0.5, "Annual rate must be between 0% and 50%")
    ax.require(1 <= years <= 100, "Years must be between 1 and 100")
    ax.require(compound_frequency in [1, 4, 12, 52, 365],
               "Compound frequency must be 1 (yearly), 4 (quarterly), 12 (monthly), 52 (weekly), or 365 (daily)")

    # Use Decimal for precise financial calculations
    p = Decimal(str(principal))
    r = Decimal(str(annual_rate))
    n = Decimal(str(compound_frequency))
    t = Decimal(str(years))

    # Compound interest formula: A = P(1 + r/n)^(nt)
    amount = p * ((1 + r / n) ** (n * t))
    interest_earned = amount - p

    # Convert back to float for return
    final_amount = float(amount.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
    total_interest = float(interest_earned.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))

    result = {
        'principal': principal,
        'final_amount': final_amount,
        'interest_earned': total_interest,
        'annual_rate': annual_rate,
        'years': years,
        'compound_frequency': compound_frequency,
        'effective_rate': (final_amount / principal) ** (1 / years) - 1
    }

    # Verify results make financial sense
    ax.ensure(result['final_amount'] >= result['principal'],
              "Final amount should be at least the principal")
    ax.ensure(result['interest_earned'] >= 0,
              "Interest earned cannot be negative")
    ax.ensure(result['effective_rate'] >= 0,
              "Effective rate cannot be negative")

    return result


@ax.checked
def calculate_loan_payment(
        loan_amount: ax.Positive[float],
        annual_rate: ax.Range[float, 0.0, 0.5],  # Max 50% APR
        years: ax.Range[int, 1, 50]  # 1-50 years
) -> Dict[str, float]:
    """Calculate monthly loan payment using standard formula"""

    if annual_rate == 0:
        # Interest-free loan
        monthly_payment = loan_amount / (years * 12)
    else:
        monthly_rate = annual_rate / 12
        num_payments = years * 12

        # Standard loan payment formula
        monthly_payment = (loan_amount * monthly_rate * (1 + monthly_rate) ** num_payments) / \
                          ((1 + monthly_rate) ** num_payments - 1)

    total_paid = monthly_payment * years * 12
    total_interest = total_paid - loan_amount

    result = {
        'loan_amount': loan_amount,
        'monthly_payment': round(monthly_payment, 2),
        'total_paid': round(total_paid, 2),
        'total_interest': round(total_interest, 2),
        'annual_rate': annual_rate,
        'years': years
    }

    return result


@ax.verify
def calculate_portfolio_value(holdings: ax.NonEmpty[List[Dict[str, Union[str, float]]]]) -> Dict[str, float]:
    """Calculate total portfolio value with validation"""

    ax.require(len(holdings) > 0, "Portfolio must have at least one holding")

    total_value = 0.0
    asset_count = 0

    for holding in holdings:
        ax.require("symbol" in holding, "Each holding must have a symbol")
        ax.require("shares" in holding, "Each holding must specify shares")
        ax.require("price" in holding, "Each holding must have a current price")

        symbol = holding["symbol"]
        shares = holding["shares"]
        price = holding["price"]

        ax.require(isinstance(symbol, str) and len(symbol) > 0, "Symbol must be non-empty string")
        ax.require(isinstance(shares, (int, float)) and shares > 0, "Shares must be positive number")
        ax.require(isinstance(price, (int, float)) and price > 0, "Price must be positive number")

        holding_value = shares * price
        total_value += holding_value
        asset_count += 1

    ax.ensure(total_value > 0, "Portfolio value must be positive")
    ax.ensure(asset_count == len(holdings), "All holdings must be processed")

    return {
        'total_value': round(total_value, 2),
        'asset_count': asset_count,
        'average_holding_value': round(total_value / asset_count, 2)
    }


# Test financial calculations
print("Testing financial calculations:")

# Compound interest
investment = calculate_compound_interest(10000.0, 0.07, 10, 12)
print(f"  +++ - $10k at 7% compounded monthly for 10 years: ${investment['final_amount']:,.2f}")
print(f"    Interest earned: ${investment['interest_earned']:,.2f}")

# Loan payment
loan = calculate_loan_payment(250000.0, 0.04, 30)
print(f"  +++ - $250k loan at 4% for 30 years: ${loan['monthly_payment']:,.2f}/month")

# Portfolio value
portfolio = [
    {"symbol": "AAPL", "shares": 100, "price": 150.0},
    {"symbol": "GOOGL", "shares": 50, "price": 2500.0},
    {"symbol": "MSFT", "shares": 75, "price": 300.0}
]
portfolio_value = calculate_portfolio_value(portfolio)
print(f"  +++ - Portfolio value: ${portfolio_value['total_value']:,.2f} ({portfolio_value['asset_count']} assets)")

print()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# EXAMPLE 3: DATA PROCESSING PIPELINE WITH STATE MANAGEMENT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print("EXAMPLE 3: Data Processing Pipeline")
print("~" * 80)
print("Stateful data processing with verified transitions")
print()


@ax.stateful(initial="initialized")
class DataProcessor:
    """Data processor with verified state transitions"""

    def __init__(self, processor_id: str, batch_size: ax.PositiveInt = 100):
        self.processor_id = processor_id
        self.batch_size = batch_size
        self.data = []
        self.processed_data = []
        self.error_count = 0
        self.start_time = None
        self.processing_stats = {
            'total_processed': 0,
            'total_errors': 0,
            'batches_completed': 0
        }

    @ax.state("initialized", "loading")
    def start_loading(self, data_source: ax.NonEmpty[str]):
        """Begin loading data from source"""
        ax.require(len(data_source) > 0, "Data source must be specified")

        print(f"  Starting to load data from {data_source}")
        self.start_time = time.time()

        # Simulate loading data
        self.data = list(range(1, 251))  # 250 items
        print(f"  Loaded {len(self.data)} items")

    @ax.state("loading", "ready")
    def finish_loading(self):
        """Complete data loading and prepare for processing"""
        ax.require(len(self.data) > 0, "No data was loaded")

        print(f"  Data loading complete: {len(self.data)} items ready")
        ax.ensure(len(self.data) > 0, "Data must be available after loading")

    @ax.state("ready", "processing")
    def start_processing(self):
        """Begin processing loaded data"""
        ax.require(len(self.data) > 0, "No data available to process")

        print(f"  Starting to process {len(self.data)} items in batches of {self.batch_size}")

    @ax.verify
    def process_batch(self) -> Dict[str, int]:
        """Process a single batch of data"""
        ax.require(len(self.data) > 0, "No data remaining to process")

        # Take a batch
        batch = self.data[:self.batch_size]
        self.data = self.data[self.batch_size:]

        processed_batch = []
        batch_errors = 0

        for item in batch:
            try:
                # Simulate processing with occasional errors
                if item % 37 == 0:  # Simulate error condition
                    raise ValueError(f"Processing error for item {item}")

                processed_item = item * 2 + 1
                processed_batch.append(processed_item)

            except ValueError:
                batch_errors += 1
                self.error_count += 1

        # Update statistics
        self.processed_data.extend(processed_batch)
        self.processing_stats['total_processed'] += len(processed_batch)
        self.processing_stats['total_errors'] += batch_errors
        self.processing_stats['batches_completed'] += 1

        result = {
            'items_processed': len(processed_batch),
            'errors': batch_errors,
            'remaining': len(self.data)
        }

        ax.ensure(result['items_processed'] >= 0, "Processed count cannot be negative")
        ax.ensure(result['errors'] >= 0, "Error count cannot be negative")

        return result

    @ax.state("processing", "completed")
    def finish_processing(self):
        """Complete processing and generate final results"""
        ax.require(len(self.data) == 0, "All data must be processed before finishing")

        end_time = time.time()
        processing_time = end_time - self.start_time if self.start_time else 0

        self.processing_stats['processing_time'] = processing_time
        self.processing_stats['success_rate'] = (
                (self.processing_stats['total_processed'] /
                 max(1, self.processing_stats['total_processed'] + self.processing_stats['total_errors'])) * 100
        )

        print(f"  Processing completed: {self.processing_stats['total_processed']} items processed")
        print(f"  Success rate: {self.processing_stats['success_rate']:.1f}%")

        ax.ensure(self.processing_stats['total_processed'] > 0, "Must have processed some items")

    @ax.state(["completed", "processing"], "ready")
    def reset(self):
        """Reset processor for new data"""
        self.data = []
        self.processed_data = []
        self.error_count = 0
        self.start_time = None
        self.processing_stats = {
            'total_processed': 0,
            'total_errors': 0,
            'batches_completed': 0
        }
        print(f"  Processor {self.processor_id} reset")

    @ax.verify
    def get_results(self) -> Dict[str, any]:
        """Get processing results (available after completion)"""
        ax.require(len(self.processed_data) > 0, "No processed data available")

        return {
            'processor_id': self.processor_id,
            'processed_items': len(self.processed_data),
            'statistics': self.processing_stats.copy(),
            'sample_results': self.processed_data[:5]  # First 5 results
        }


# Test the data processing pipeline
print("Testing data processing pipeline:")

try:
    processor = DataProcessor("PROC-001", batch_size=50)

    # Full processing pipeline
    processor.start_loading("database://prod/customer_data")
    processor.finish_loading()
    processor.start_processing()

    # Process all batches
    while True:
        batch_result = processor.process_batch()
        print(f"    Batch: {batch_result['items_processed']} processed, "
              f"{batch_result['errors']} errors, {batch_result['remaining']} remaining")

        if batch_result['remaining'] == 0:
            break

    processor.finish_processing()

    # Get final results
    results = processor.get_results()
    print(f"  +++ - Processing complete: {results['processed_items']} items")
    print(f"    Success rate: {results['statistics']['success_rate']:.1f}%")

except ax.VerificationError as e:
    print(f"  XXX - Processing failed: {str(e).split('Message: ')[-1]}")

print()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# EXAMPLE 4: GAME STATE MANAGEMENT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print("EXAMPLE 4: Game State Management")
print("~" * 80)
print("Turn-based game with verified state transitions")
print()


@ax.enable_for_dataclass
@dataclass
class Player:
    """Player with validated attributes"""
    name: ax.NonEmpty[str]
    health: ax.Range[int, 0, 100] = 100
    score: ax.Range[int, 0, 999999] = 0
    level: ax.PositiveInt = 1

    def is_alive(self) -> bool:
        return self.health > 0


@ax.stateful(initial="waiting")
class GameSession:
    """Turn-based game session with state verification"""

    def __init__(self, game_id: str, max_players: ax.Range[int, 2, 8] = 4):
        self.game_id = game_id
        self.max_players = max_players
        self.players = []
        self.current_turn = 0
        self.round_number = 0
        self.game_log = []

    @ax.state("waiting", "lobby")
    def open_lobby(self):
        """Open game lobby for players to join"""
        print(f"  Game {self.game_id} lobby opened (max {self.max_players} players)")
        self._log_event("lobby_opened")

    @ax.verify
    def add_player(self, player: Player) -> bool:
        """Add player to lobby"""
        ax.require(len(self.players) < self.max_players, "Lobby is full")
        ax.require(player.name not in [p.name for p in self.players], "Player name already taken")
        ax.require(player.is_alive(), "Player must be alive to join")

        self.players.append(player)
        self._log_event(f"player_joined", {"player": player.name, "count": len(self.players)})

        ax.ensure(player in self.players, "Player was not added")
        print(f"    {player.name} joined ({len(self.players)}/{self.max_players})")
        return True

    @ax.state("lobby", "playing")
    def start_game(self):
        """Start the game"""
        ax.require(len(self.players) >= 2, "Need at least 2 players to start")
        ax.require(all(p.is_alive() for p in self.players), "All players must be alive")

        self.round_number = 1
        self.current_turn = 0

        print(f"  Game started with {len(self.players)} players")
        self._log_event("game_started", {"players": [p.name for p in self.players]})

    @ax.verify
    def play_turn(self, action: ax.NonEmpty[str]) -> Dict[str, any]:
        """Execute a player's turn"""
        ax.require(0 <= self.current_turn < len(self.players), "Invalid turn index")

        current_player = self.players[self.current_turn]
        ax.require(current_player.is_alive(), "Current player must be alive")

        # Simulate turn actions
        turn_result = self._process_action(current_player, action)

        # Advance to next player
        self.current_turn = (self.current_turn + 1) % len(self.players)

        # Check if round completed
        if self.current_turn == 0:
            self.round_number += 1
            self._log_event("round_completed", {"round": self.round_number - 1})

        self._log_event("turn_played", {
            "player": current_player.name,
            "action": action,
            "result": turn_result
        })

        ax.ensure("player" in turn_result, "Turn result must include player")
        ax.ensure("action_success" in turn_result, "Turn result must indicate success")

        return turn_result

    @ax.verify
    def _process_action(self, player: Player, action: str) -> Dict[str, any]:
        """Process a player action"""
        action = action.lower().strip()

        if action == "attack":
            # Simulate attack
            damage = 10
            # Find target (simplified - attack next player)
            target_idx = (self.players.index(player) + 1) % len(self.players)
            target = self.players[target_idx]

            old_health = target.health
            target.health = max(0, target.health - damage)

            player.score += damage

            return {
                "player": player.name,
                "action": "attack",
                "target": target.name,
                "damage": damage,
                "target_health": target.health,
                "action_success": True
            }

        elif action == "heal":
            # Heal self
            heal_amount = 15
            old_health = player.health
            player.health = min(100, player.health + heal_amount)
            actual_heal = player.health - old_health

            return {
                "player": player.name,
                "action": "heal",
                "heal_amount": actual_heal,
                "new_health": player.health,
                "action_success": True
            }

        elif action == "defend":
            # Defensive action
            return {
                "player": player.name,
                "action": "defend",
                "defense_bonus": 5,
                "action_success": True
            }

        else:
            return {
                "player": player.name,
                "action": action,
                "action_success": False,
                "error": "Unknown action"
            }

    @ax.verify
    def check_game_over(self) -> Optional[Dict[str, any]]:
        """Check if game should end"""
        alive_players = [p for p in self.players if p.is_alive()]

        if len(alive_players) <= 1:
            # Game over - one or no players left
            winner = alive_players[0] if alive_players else None
            return {
                "game_over": True,
                "winner": winner.name if winner else "No winner",
                "final_scores": [(p.name, p.score) for p in self.players],
                "rounds_played": self.round_number
            }

        if self.round_number >= 20:  # Max 20 rounds
            # Game over - time limit
            winner = max(self.players, key=lambda p: p.score)
            return {
                "game_over": True,
                "winner": winner.name,
                "reason": "Time limit reached",
                "final_scores": [(p.name, p.score) for p in self.players],
                "rounds_played": self.round_number
            }

        return None

    @ax.state("playing", "finished")
    def end_game(self, reason: ax.NonEmpty[str] = "Game completed"):
        """End the current game"""
        game_over_result = self.check_game_over()
        ax.require(game_over_result is not None, "Game cannot end - conditions not met")

        self._log_event("game_ended", {
            "reason": reason,
            "winner": game_over_result.get("winner"),
            "rounds": self.round_number
        })

        print(f"  Game ended: {game_over_result.get('winner')} wins!")
        return game_over_result

    @ax.state(["finished", "lobby"], "waiting")
    def reset_game(self):
        """Reset game for new session"""
        self.players = []
        self.current_turn = 0
        self.round_number = 0
        self.game_log = []

        print(f"  Game {self.game_id} reset")

    def _log_event(self, event_type: str, data: Dict = None):
        """Log game events"""
        self.game_log.append({
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            "data": data or {}
        })

    @ax.verify
    def get_game_status(self) -> Dict[str, any]:
        """Get current game status"""
        alive_players = [p for p in self.players if p.is_alive()]

        return {
            "game_id": self.game_id,
            "round": self.round_number,
            "current_player": self.players[self.current_turn].name if self.players else None,
            "players_alive": len(alive_players),
            "total_players": len(self.players),
            "player_status": [
                {
                    "name": p.name,
                    "health": p.health,
                    "score": p.score,
                    "alive": p.is_alive()
                } for p in self.players
            ]
        }


# Test the game session
print("Testing game session:")

try:
    game = GameSession("GAME-001", max_players=3)

    # Set up game
    game.open_lobby()

    players = [
        Player("Alice", health=100, score=0),
        Player("Bob", health=100, score=0),
        Player("Charlie", health=100, score=0)
    ]

    for player in players:
        game.add_player(player)

    game.start_game()

    # Play several turns
    actions = ["attack", "heal", "defend", "attack", "attack", "heal"]
    for i, action in enumerate(actions):
        turn_result = game.play_turn(action)
        status = game.get_game_status()

        print(f"    Turn {i + 1}: {turn_result['player']} used {turn_result['action']}")
        print(f"      Players alive: {status['players_alive']}")

        # Check if game should end
        game_over = game.check_game_over()
        if game_over:
            result = game.end_game("Victory condition met")
            break

    if not game_over:
        print("    Game continuing...")

    print(f"  +++ - Game session completed successfully")

except ax.VerificationError as e:
    print(f"  XXX - Game error: {str(e).split('Message: ')[-1]}")

print()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# EXAMPLE 5: SCIENTIFIC DATA ANALYSIS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print("EXAMPLE 5: Scientific Data Analysis")
print("~" * 80)
print("Statistical analysis with verification")
print()


@ax.enable_for_dataclass
@dataclass
class DataPoint:
    """Scientific data point with validation"""
    timestamp: float
    value: float
    measurement_id: ax.NonEmpty[str]
    quality_score: ax.Range[float, 0.0, 1.0]  # 0-1 quality rating

    def __post_init__(self):
        ax.require(self.timestamp > 0, "Timestamp must be positive")
        ax.require(abs(self.value) < 1e10, "Value seems unreasonably large")


@ax.checked
def calculate_statistics(data: ax.NonEmpty[List[DataPoint]]) -> Dict[str, float]:
    """Calculate comprehensive statistics for scientific data"""

    values = [dp.value for dp in data]
    quality_scores = [dp.quality_score for dp in data]

    # Basic statistics
    n = len(values)
    mean = sum(values) / n

    # Calculate variance and standard deviation
    variance = sum((x - mean) ** 2 for x in values) / n
    std_dev = variance ** 0.5

    # Calculate median
    sorted_values = sorted(values)
    if n % 2 == 0:
        median = (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
    else:
        median = sorted_values[n // 2]

    # Quality-weighted statistics
    total_weight = sum(quality_scores)
    weighted_mean = sum(v * q for v, q in zip(values, quality_scores)) / total_weight

    # Range and quartiles
    q1 = sorted_values[n // 4]
    q3 = sorted_values[3 * n // 4]
    iqr = q3 - q1

    result = {
        'count': n,
        'mean': mean,
        'median': median,
        'std_dev': std_dev,
        'variance': variance,
        'min': min(values),
        'max': max(values),
        'range': max(values) - min(values),
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'weighted_mean': weighted_mean,
        'avg_quality': sum(quality_scores) / len(quality_scores)
    }

    return result


@ax.verify
def detect_outliers(data: ax.NonEmpty[List[DataPoint]],
                    method: str = "iqr",
                    threshold: ax.Positive[float] = 1.5) -> Dict[str, any]:
    """Detect outliers in scientific data"""

    ax.require(method in ["iqr", "zscore"], "Method must be 'iqr' or 'zscore'")
    ax.require(threshold > 0, "Threshold must be positive")

    values = [dp.value for dp in data]
    outliers = []

    if method == "iqr":
        # IQR method
        sorted_values = sorted(values)
        n = len(sorted_values)
        q1 = sorted_values[n // 4]
        q3 = sorted_values[3 * n // 4]
        iqr = q3 - q1

        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        for i, dp in enumerate(data):
            if dp.value < lower_bound or dp.value > upper_bound:
                outliers.append({
                    'index': i,
                    'value': dp.value,
                    'measurement_id': dp.measurement_id,
                    'deviation': min(abs(dp.value - lower_bound), abs(dp.value - upper_bound))
                })

    elif method == "zscore":
        # Z-score method
        mean = sum(values) / len(values)
        std_dev = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5

        ax.require(std_dev > 0, "Standard deviation must be positive for z-score method")

        for i, dp in enumerate(data):
            z_score = abs(dp.value - mean) / std_dev
            if z_score > threshold:
                outliers.append({
                    'index': i,
                    'value': dp.value,
                    'measurement_id': dp.measurement_id,
                    'z_score': z_score
                })

    result = {
        'method': method,
        'threshold': threshold,
        'outliers_found': len(outliers),
        'outlier_percentage': (len(outliers) / len(data)) * 100,
        'outliers': outliers
    }

    ax.ensure(result['outliers_found'] >= 0, "Outlier count cannot be negative")
    ax.ensure(0 <= result['outlier_percentage'] <= 100, "Percentage must be 0-100")

    return result


@ax.verify
def perform_regression_analysis(x_data: ax.NonEmpty[List[float]],
                                y_data: ax.NonEmpty[List[float]]) -> Dict[str, float]:
    """Perform simple linear regression with verification"""

    ax.require(len(x_data) == len(y_data), "X and Y data must have same length")
    ax.require(len(x_data) >= 3, "Need at least 3 data points for regression")

    n = len(x_data)

    # Calculate means
    x_mean = sum(x_data) / n
    y_mean = sum(y_data) / n

    # Calculate slope and intercept
    numerator = sum((x_data[i] - x_mean) * (y_data[i] - y_mean) for i in range(n))
    denominator = sum((x_data[i] - x_mean) ** 2 for i in range(n))

    ax.require(denominator != 0, "X values cannot all be the same")

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    # Calculate correlation coefficient
    x_std = (sum((x - x_mean) ** 2 for x in x_data) / n) ** 0.5
    y_std = (sum((y - y_mean) ** 2 for y in y_data) / n) ** 0.5

    if x_std > 0 and y_std > 0:
        correlation = numerator / (n * x_std * y_std)
    else:
        correlation = 0

    # Calculate R-squared
    r_squared = correlation ** 2

    result = {
        'slope': slope,
        'intercept': intercept,
        'correlation': correlation,
        'r_squared': r_squared,
        'n_points': n
    }

    ax.ensure(-1 <= result['correlation'] <= 1, "Correlation must be between -1 and 1")
    ax.ensure(0 <= result['r_squared'] <= 1, "R-squared must be between 0 and 1")

    return result


# Test scientific analysis
print("Testing scientific data analysis:")

# Generate sample data
import math, random

sample_data = []
for i in range(50):
    timestamp = time.time() + i * 3600  # Hourly measurements
    # Sine wave with noise
    value = 10 * math.sin(i * 0.2) + random.uniform(-1, 1) if 'random' in dir() else 10 * math.sin(i * 0.2)
    quality = 0.8 + 0.2 * (i % 3) / 2  # Varying quality

    sample_data.append(DataPoint(
        timestamp=timestamp,
        value=value,
        measurement_id=f"MEAS_{i:03d}",
        quality_score=quality
    ))

try:
    # Calculate statistics
    stats = calculate_statistics(sample_data)
    print(f"  +++ - Statistics: mean={stats['mean']:.2f}, std={stats['std_dev']:.2f}")
    print(f"    Range: {stats['min']:.2f} to {stats['max']:.2f}")

    # Detect outliers
    outliers = detect_outliers(sample_data, method="iqr", threshold=2.0)
    print(f"  +++ - Outlier detection: {outliers['outliers_found']} outliers found ({outliers['outlier_percentage']:.1f}%)")

    # Regression analysis
    x_values = [i for i in range(len(sample_data))]
    y_values = [dp.value for dp in sample_data]
    regression = perform_regression_analysis(x_values, y_values)
    print(f"  +++ - Regression: slope={regression['slope']:.3f}, R²={regression['r_squared']:.3f}")

except ax.VerificationError as e:
    print(f"  XXX - Analysis failed: {str(e).split('Message: ')[-1]}")

print()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FINAL SUMMARY AND PERFORMANCE REPORT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print("SUMMARY: Complete Usage Examples")
print("~" * 80)

summary = """
+++ - Example 1: Web API validation with dataclasses and business logic
+++ - Example 2: Financial calculations with precision and comprehensive checks  
+++ - Example 3: Stateful data processing pipeline with error handling
+++ - Example 4: Game state management with turn-based logic
+++ - Example 5: Scientific data analysis with statistical validation

Key Patterns Demonstrated:
- @ax.checked for automatic type validation
- @ax.verify for custom business logic
- @ax.stateful for protocol verification
- @ax.enable_for_dataclass for field validation
- Performance tracking with track_performance=True
- Error handling with helpful VerificationError messages
- Context managers for temporary mode changes

Production Considerations:
- Use ax.set_mode('prod') in production for minimal overhead
- Enable performance tracking selectively 
- Combine type hints with business logic validation
- Use dataclasses for structured input validation
- Implement state machines for complex workflows
"""

print(summary)

# Generate comprehensive reports
print("Performance Report:")
print(ax.performance_report())

print("\nSystem Status:")
print(ax.report())

print("\nThese examples show real-world patterns you can adapt for:")
print("~ Web APIs and microservices")
print("~ Financial and scientific calculations")
print("~ Data processing pipelines")
print("~ Game development and state machines")
print("~ Any Python application requiring reliability")

print(f"\nRemember: Start simple with @ax.verify, then add @ax.checked as needed!")
print(f"Current mode: {ax.get_mode()} - Set ax.mode='prod' for production deployment")
