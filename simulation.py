
import random

DIRECTIONS = ("N", "S", "E", "W")
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
MOVEMENT_TIME = {"left": 4.0, "straight": 2.0, "right": 3.0}
CLEARANCE_TIME = 2.0
MAX_CYCLES = 100
PENALTY = 100000.0

def generate_variants(num_variants: int = 50, num_cars: int = 30, min_cars: int = 1, max_cars: int = 8, seed: int = 42, arrival_window: float = 120.0) -> list[dict]:
    rng = random.Random(seed)
    lanes = [(direction, lane) for direction in DIRECTIONS for lane in range(2)]
    if not (len(lanes) * min_cars <= num_cars <= len(lanes) * max_cars):
        raise ValueError("Lane bounds cannot accommodate the requested number of cars")
    if arrival_window < 0:
        raise ValueError("Arrival window cannot be negative")

    variants = []
    for _ in range(num_variants):
        for attempt in range(100000):
            intersection = {key: [] for key in lanes}
            for _ in range(num_cars):
                turn = rng.choices(["left", "straight", "right"], weights=[20, 60, 20])[0]
                direction = rng.choice(DIRECTIONS)
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

def simulate_traffic(intersection: dict, individual: list[float]) -> dict:
    times = {direction: (float(individual[2 * i]), float(individual[2 * i + 1])) for i, direction in enumerate(DIRECTIONS)}
    if any(a < 0 or g <= 0 for a, g in times.values()):
        raise ValueError("Invalid traffic-light durations")

    queues = {key: [dict(car) for car in value] for key, value in intersection.items()}
    remaining = sum(len(queue) for queue in queues.values())
    if remaining == 0:
        return {"completion_time": 0.0, "average_wait": 0.0, "max_wait": 0.0}

    elapsed = 0.0
    waits = []
    axes = (("N", "S"), ("E", "W"))

    for _ in range(MAX_CYCLES):
        for axis in axes:
            axis_duration = max(sum(times[direction]) for direction in axis)
            phase_times = sorted(set([0.0, axis_duration] + [times[d][0] for d in axis] + [sum(times[d]) for d in axis]))
            local_time = 0.0

            for phase_end in phase_times[1:]:
                while local_time < phase_end - 1e-9:
                    current_time = elapsed + local_time
                    candidates = []
                    for direction in axis:
                        advanced, green = times[direction]

                        if local_time < advanced - 1e-9:
                            signal = "advanced"
                            signal_end = advanced
                        elif local_time < advanced + green - 1e-9:
                            signal = "regular"
                            signal_end = advanced + green
                        else:
                            continue

                        for lane in range(2):
                            queue = queues[(direction, lane)]
                            if not queue or queue[0]["arrival"] > current_time + 1e-9:
                                continue
                            turn = queue[0]["turn"]

                            if signal == "advanced" and turn != "left":
                                continue
                            if signal == "regular" and turn == "left":
                                if any(queues[(OPPOSITE[direction], l)] and queues[(OPPOSITE[direction], l)][0]["arrival"] <= current_time + 1e-9 for l in range(2)):
                                    continue

                            candidates.append((direction, lane, turn, signal_end))

                    if not candidates:
                        arrivals = [queue[0]["arrival"] - elapsed for direction in axis for lane in range(2)
                                    for queue in [queues[(direction, lane)]] if queue and queue[0]["arrival"] > current_time + 1e-9]
                        local_time = min(phase_end, min(arrivals)) if arrivals else phase_end
                        continue

                    direction, lane, turn, signal_end = candidates[0]
                    duration = MOVEMENT_TIME[turn]
                    if local_time + duration > signal_end + 1e-9:
                        local_time = phase_end
                        continue

                    car = queues[(direction, lane)].pop(0)
                    waits.append(max(0.0, current_time - car["arrival"]))
                    remaining -= 1
                    local_time += duration
                    if remaining == 0:
                        return {"completion_time": elapsed + local_time,
                                "average_wait": sum(waits) / len(waits), "max_wait": max(waits)}

                local_time = max(local_time, phase_end)

            elapsed += axis_duration + CLEARANCE_TIME

    penalty = PENALTY + elapsed + remaining * PENALTY
    return {"completion_time": penalty, "average_wait": penalty, "max_wait": penalty}