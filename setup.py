
import json
from pathlib import Path
from simulation import generate_variants

NUM_VARIANTS = 50
NUM_CARS = 100
MIN_CARS = 1
MAX_CARS = 80
VARIANT_SEED = 42
ARRIVAL_WINDOW = 240.0
# Uneven traffic: S carries 3x the cars of each other approach, and N has many left turns.
DIRECTION_WEIGHTS = {"N": 1, "S": 3, "E": 1, "W": 1}
TURN_WEIGHTS = {"N": [50, 35, 15], "S": [5, 80, 15], "E": [20, 65, 15], "W": [20, 65, 15]}  # [left, straight, right]
VARIANT_FILE = Path(__file__).with_name("intersection_variants.json")

# score = average wait (typical driver) + 0.2 * max wait (fairness to the unluckiest car).
# completion_time is left out: among good plans it moves with max_wait (correlation 0.89),
# so it counted the worst-off cars twice. 0.2 gives both remaining terms about equal influence.
COMPLETION_WEIGHT = 0.0
AVERAGE_WAIT_WEIGHT = 1.0
MAX_WAIT_WEIGHT = 0.2

MU = 20
LAMD = 100
GENERATIONS = 100
SIGMA_INIT = 5.0
SIGMA_BOUND = 0.01
LR = 0.1
TOURNAMENT_SIZE = 3

def load_variants():
    settings = {"num_variants": NUM_VARIANTS, "num_cars": NUM_CARS, "min_cars": MIN_CARS, "max_cars": MAX_CARS, "seed": VARIANT_SEED, "arrival_window": ARRIVAL_WINDOW,
                "direction_weights": DIRECTION_WEIGHTS, "turn_weights": TURN_WEIGHTS}

    if VARIANT_FILE.exists():
        with VARIANT_FILE.open("r") as file:
            saved = json.load(file)

        if saved.get("settings") != settings:
            raise ValueError("Saved variants do not match setup.py. Rename or delete the JSON file to generate new variants.")
        
        data = saved["variants"]

    else:
        variants = generate_variants(NUM_VARIANTS, NUM_CARS, MIN_CARS, MAX_CARS, VARIANT_SEED, ARRIVAL_WINDOW, DIRECTION_WEIGHTS, TURN_WEIGHTS)
        data = [{f"{direction},{lane}": queue for (direction, lane), queue in variant.items()} for variant in variants]
        with VARIANT_FILE.open("w") as file:
            json.dump({"settings": settings, "variants": data}, file, indent=2)

    return [{(key.split(",")[0], int(key.split(",")[1])): queue for key, queue in variant.items()} for variant in data]

if __name__ == "__main__":
    variants = load_variants()
    print(f"Generated/loaded {len(variants)} fixed traffic scenarios")
    print(f"Saved in: {VARIANT_FILE}")