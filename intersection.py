"""Intersection state model for the CISC 455 traffic-light EA project.

A pure data/state model: four approaches, each with a few lanes, each lane
holding a queue of cars. There is no traffic-light logic, arrival process or
EA in here; those build on top of this module.

Approaches (direction of travel):
    E: eastbound  (cars coming from the west)
    W: westbound  (cars coming from the east)
    N: northbound (cars coming from the south)
    S: southbound (cars coming from the north)
"""

from __future__ import annotations

import math
import random
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

APPROACHES: Tuple[str, ...] = ("E", "W", "N", "S")

LaneKey = Tuple[str, int]  # (approach, lane index)


@dataclass
class Car:
    """A queued car.

    position is the distance from the stop line in queue slots:
    0 is the front car, 1 is directly behind it, and so on.
    """

    id: int
    approach: str
    lane: int
    position: int


class Intersection:
    """Car queues for every lane of a single four-way intersection."""

    def __init__(
        self,
        num_cars: int = 30,
        lanes_per_approach: int = 2,
        capacity_fraction: float = 1.0,
        lane_capacity: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Create an empty intersection; call populate_random() or populate_from().

        lane_capacity defaults to ceil(num_cars * capacity_fraction). An
        explicit lane_capacity takes precedence over capacity_fraction.
        """
        if num_cars < 0:
            raise ValueError(f"num_cars must be >= 0, got {num_cars}")
        if lanes_per_approach < 1:
            raise ValueError(f"lanes_per_approach must be >= 1, got {lanes_per_approach}")

        if lane_capacity is None:
            if capacity_fraction <= 0:
                raise ValueError(f"capacity_fraction must be > 0, got {capacity_fraction}")
            # Fraction(str(x)) keeps the decimal exact, so 30 * 0.1 gives 3
            # rather than ceil(3.0000000000000004) = 4.
            lane_capacity = math.ceil(num_cars * Fraction(str(capacity_fraction)))
        elif lane_capacity < 0:
            raise ValueError(f"lane_capacity must be >= 0, got {lane_capacity}")

        self.num_cars = num_cars
        self.lanes_per_approach = lanes_per_approach
        self.capacity_fraction = capacity_fraction
        self.lane_capacity: int = lane_capacity

        total_capacity = self.lane_capacity * self.num_lanes
        if total_capacity < num_cars:
            raise ValueError(
                f"total capacity {total_capacity} ({self.num_lanes} lanes x "
                f"{self.lane_capacity} per lane) is less than num_cars={num_cars}"
            )

        self._rng = random.Random(seed)
        self._queues: Dict[LaneKey, List[Car]] = {key: [] for key in self.lanes}
        self._cars: Dict[int, Car] = {}

    # ----- layout ---------------------------------------------------------

    @property
    def lanes(self) -> List[LaneKey]:
        """All (approach, lane) keys in display order."""
        return [(a, l) for a in APPROACHES for l in range(self.lanes_per_approach)]

    @property
    def num_lanes(self) -> int:
        """Total number of lanes across all approaches."""
        return len(APPROACHES) * self.lanes_per_approach

    def _check_lane(self, approach: str, lane: int) -> LaneKey:
        """Return (approach, lane) if it names a real lane, else raise ValueError."""
        if approach not in APPROACHES or not isinstance(lane, int) or not (
            0 <= lane < self.lanes_per_approach
        ):
            raise ValueError(
                f"no such lane ({approach!r}, {lane!r}); approaches are "
                f"{APPROACHES}, lanes 0..{self.lanes_per_approach - 1}"
            )
        return (approach, lane)

    def _place(self, placements: Sequence[Tuple[int, LaneKey]]) -> None:
        """Rebuild all queues. Cars join the back of their lane in the order given."""
        self._queues = {key: [] for key in self.lanes}
        self._cars = {}
        for car_id, (approach, lane) in placements:
            queue = self._queues[(approach, lane)]
            car = Car(id=car_id, approach=approach, lane=lane, position=len(queue))
            queue.append(car)
            self._cars[car_id] = car

    def populate_random(self) -> None:
        """Place every car in a random lane that still has room.

        Cars join in a shuffled order, so queue order is random too.
        """
        order = list(range(self.num_cars))
        self._rng.shuffle(order)
        counts: Dict[LaneKey, int] = dict.fromkeys(self.lanes, 0)
        placements: List[Tuple[int, LaneKey]] = []
        for car_id in order:
            # Never empty: __init__ guarantees total capacity >= num_cars.
            open_lanes = [k for k in self.lanes if counts[k] < self.lane_capacity]
            key = self._rng.choice(open_lanes)
            counts[key] += 1
            placements.append((car_id, key))
        self._place(placements)

    def populate_from(self, assignment: Sequence[Tuple[str, int]]) -> None:
        """Build the layout from assignment[car_id] = (approach, lane).

        Cars join their lane in id order, so the lowest id in each lane is at
        the front. The current state is left untouched if validation fails.
        """
        if len(assignment) != self.num_cars:
            raise ValueError(
                f"assignment has {len(assignment)} entries, expected num_cars={self.num_cars}"
            )
        keys: List[LaneKey] = []
        for car_id, entry in enumerate(assignment):
            try:
                approach, lane = entry
            except (TypeError, ValueError):
                raise ValueError(
                    f"assignment[{car_id}] must be an (approach, lane) pair, got {entry!r}"
                ) from None
            keys.append(self._check_lane(approach, lane))

        overfull = {k: n for k, n in Counter(keys).items() if n > self.lane_capacity}
        if overfull:
            raise ValueError(
                f"layout overfills lane(s) {overfull}; lane_capacity is {self.lane_capacity}"
            )
        self._place(list(enumerate(keys)))

    # ----- queries --------------------------------------------------------

    def get_car_location(self, car_id: int) -> Tuple[str, int, int]:
        """Return (approach, lane, position) for a car still in the intersection."""
        try:
            car = self._cars[car_id]
        except KeyError:
            raise KeyError(f"car {car_id} is not in the intersection") from None
        return (car.approach, car.lane, car.position)

    def cars_in_lane(self, approach: str, lane: int) -> List[Car]:
        """Cars in one lane, front of the queue first. Treat the cars as read-only."""
        return list(self._queues[self._check_lane(approach, lane)])

    def queue_length(self, approach: str, lane: int) -> int:
        """Number of cars queued in one lane."""
        return len(self._queues[self._check_lane(approach, lane)])

    def queue_lengths(self) -> Dict[LaneKey, int]:
        """Queue length for every lane, keyed by (approach, lane)."""
        return {key: len(queue) for key, queue in self._queues.items()}

    # ----- mutation -------------------------------------------------------

    def remove_front_car(self, approach: str, lane: int) -> Car:
        """Pop the car at the stop line and move everyone behind it up one slot.

        Raises IndexError if the lane is empty. The removed car leaves the
        intersection, so get_car_location() no longer finds it.
        """
        queue = self._queues[self._check_lane(approach, lane)]
        if not queue:
            raise IndexError(f"lane ({approach!r}, {lane}) is empty")
        car = queue.pop(0)
        del self._cars[car.id]
        for behind in queue:
            behind.position -= 1
        return car

    def copy(self) -> Intersection:
        """Independent deep copy, including the random generator's state."""
        return deepcopy(self)

    def __str__(self) -> str:
        lines = [
            f"Intersection: {len(self._cars)}/{self.num_cars} cars, "
            f"lane_capacity={self.lane_capacity} (front of queue on the left)"
        ]
        for approach, lane in self.lanes:
            cells = "".join(f"[{car.id}]" for car in self._queues[(approach, lane)])
            lines.append(f"{approach} lane {lane}: {cells or '(empty)'}")
        return "\n".join(lines)


if __name__ == "__main__":
    demo = Intersection(seed=42)
    demo.populate_random()
    print(f"lane_capacity = {demo.lane_capacity}")
    print(demo)
