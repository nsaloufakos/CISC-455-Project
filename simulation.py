
import random

DIRECTIONS = ("N", "S", "E", "W")
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
MOVEMENT_TIME = {"left": 4.0, "straight": 2.0, "right": 3.0}
CLEARANCE_TIME = 2.0
MAX_CYCLES = 100
PENALTY = 100000.0
EPS = 1e-9
TURNS = ("left", "straight", "right")
DEFAULT_TURN_WEIGHTS = (20, 60, 20)  # left, straight, right

def generate_variants(num_variants: int = 50, num_cars: int = 30, min_cars: int = 1, max_cars: int = 8, seed: int = 42, arrival_window: float = 120.0,
                      direction_weights: dict | None = None, turn_weights: dict | None = None) -> list[dict]:
    """Make num_variants random scenarios of num_cars cars each.

    direction_weights: relative demand per approach, e.g. {"N": 1, "S": 3, "E": 1, "W": 1}.
        An approach left out gets weight 1, so the default is equal demand.
    turn_weights: (left, straight, right) weights per approach, e.g. {"N": (50, 35, 15)}.
        An approach left out gets DEFAULT_TURN_WEIGHTS (20/60/20).
    """
    rng = random.Random(seed)
    lanes = [(direction, lane) for direction in DIRECTIONS for lane in range(2)]
    if not (len(lanes) * min_cars <= num_cars <= len(lanes) * max_cars):
        raise ValueError("Lane bounds cannot accommodate the requested number of cars")
    if arrival_window < 0:
        raise ValueError("Arrival window cannot be negative")

    direction_weights = direction_weights or {}
    turn_weights = turn_weights or {}
    unknown = (set(direction_weights) | set(turn_weights)) - set(DIRECTIONS)
    if unknown:
        raise ValueError(f"Unknown approaches in weights: {sorted(unknown)}")
    demand = [direction_weights.get(direction, 1) for direction in DIRECTIONS]
    turn_mix = {direction: tuple(turn_weights.get(direction, DEFAULT_TURN_WEIGHTS)) for direction in DIRECTIONS}
    if any(w < 0 for w in demand) or sum(demand) <= 0:
        raise ValueError("Direction weights must be non-negative with a positive total")
    for direction, weights in turn_mix.items():
        if len(weights) != 3 or any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError(f"Turn weights for {direction} must be three non-negative numbers with a positive total")

    variants = []
    for _ in range(num_variants):
        for attempt in range(100000):
            intersection = {key: [] for key in lanes}
            for _ in range(num_cars):
                direction = rng.choices(DIRECTIONS, weights=demand)[0]
                turn = rng.choices(TURNS, weights=turn_mix[direction])[0]
                valid_lanes = [0] if turn == "left" else [1] if turn == "right" else [0, 1]
                lane = rng.choice(valid_lanes)
                arrival = rng.uniform(0, arrival_window)
                intersection[(direction, lane)].append({"turn": turn, "arrival": arrival})

            if all(min_cars <= len(queue) <= max_cars for queue in intersection.values()):
                for queue in intersection.values():
                    queue.sort(key = lambda car: car["arrival"])
                variants.append(intersection)
                break
        else:
            raise RuntimeError("Could not generate a valid intersection variant")

    return variants

def _signal(advanced: float, green: float, t: float) -> tuple:
    """Signal shown to one direction at local time t: ("advanced" | "regular" | None, end time)."""
    if t < advanced - EPS:
        return "advanced", advanced
    if t < advanced + green - EPS:
        return "regular", advanced + green
    return None, None

def _conflicts(direction: str, turn: str, moving: list) -> bool:
    """Only a left turn and an oncoming straight or right turn cross paths.

    Opposite straights, opposite rights, opposing lefts, and the two lanes of
    one approach can all move at the same time.
    """
    for other_direction, other_turn, _ in moving:
        if other_direction == OPPOSITE[direction] and (turn == "left") != (other_turn == "left"):
            return True
    return False

def simulate_traffic(intersection: dict, individual: list[float]) -> dict:
    """Run the fixed-time signal plan in individual over one traffic scenario.

    Each lane sends one car at a time, and any cars whose paths do not cross
    can move together. At each moment cars start in priority order: protected
    lefts, then straights and rights, then permissive lefts (earlier arrivals
    first within each group). A permissive left therefore waits while oncoming
    straight or right traffic is moving, but never for oncoming left-turners.
    A car only starts if it can finish before its own signal ends.
    """
    times = {direction: (float(individual[2 * i]), float(individual[2 * i + 1])) for i, direction in enumerate(DIRECTIONS)}
    if any(a < 0 or g <= 0 for a, g in times.values()):
        raise ValueError("Invalid traffic-light durations")

    queues = {key: [dict(car) for car in value] for key, value in intersection.items()}
    remaining = sum(len(queue) for queue in queues.values())
    if remaining == 0:
        return {"completion_time": 0.0, "average_wait": 0.0, "max_wait": 0.0}

    elapsed = 0.0
    finish = 0.0
    waits = []
    axes = (("N", "S"), ("E", "W"))

    for _ in range(MAX_CYCLES):
        for axis in axes:
            axis_duration = max(sum(times[direction]) for direction in axis)
            signal_changes = [t for direction in axis for t in (times[direction][0], sum(times[direction]))]
            lane_free_at = {(direction, lane): 0.0 for direction in axis for lane in range(2)}
            moving = []  # (direction, turn, end time) of cars crossing right now
            now = 0.0

            while True:
                current_time = elapsed + now
                moving = [m for m in moving if m[2] > now + EPS]

                ready = []
                for direction in axis:
                    signal, signal_end = _signal(*times[direction], now)
                    if signal is None:
                        continue
                    for lane in range(2):
                        queue = queues[(direction, lane)]
                        if not queue or lane_free_at[(direction, lane)] > now + EPS or queue[0]["arrival"] > current_time + EPS:
                            continue
                        turn = queue[0]["turn"]
                        if signal == "advanced" and turn != "left":
                            continue
                        if now + MOVEMENT_TIME[turn] > signal_end + EPS:
                            continue
                        priority = 0 if signal == "advanced" else 2 if turn == "left" else 1
                        ready.append((priority, queue[0]["arrival"], direction, lane, turn))

                for _, _, direction, lane, turn in sorted(ready):
                    if _conflicts(direction, turn, moving):
                        continue
                    car = queues[(direction, lane)].pop(0)
                    end = now + MOVEMENT_TIME[turn]
                    moving.append((direction, turn, end))
                    lane_free_at[(direction, lane)] = end
                    waits.append(max(0.0, current_time - car["arrival"]))
                    finish = max(finish, elapsed + end)
                    remaining -= 1

                if remaining == 0:
                    return {"completion_time": finish, "average_wait": sum(waits) / len(waits), "max_wait": max(waits)}

                # Jump to the next moment anything can change: a car finishes,
                # a signal changes, or a car reaches the front of an empty lane.
                arrivals = [queues[(direction, lane)][0]["arrival"] - elapsed for direction in axis for lane in range(2) if queues[(direction, lane)]]
                events = [t for t in list(lane_free_at.values()) + signal_changes + arrivals if now + EPS < t < axis_duration - EPS]
                if not events:
                    break
                now = min(events)

            elapsed += axis_duration + CLEARANCE_TIME

    penalty = PENALTY + elapsed + remaining * PENALTY
    return {"completion_time": penalty, "average_wait": penalty, "max_wait": penalty}
