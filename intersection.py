"""Intersection state model for the CISC 455 traffic-light EA project.

Four approaches, each with a few lanes, each lane holding a queue of cars.
Pure data/state: no traffic-light logic, arrivals or EA in here.

Approaches (direction of travel):
    E: eastbound  (cars coming from the west)
    W: westbound  (cars coming from the east)
    N: northbound (cars coming from the south)
    S: southbound (cars coming from the north)
"""

import math
import random
from copy import deepcopy
from dataclasses import dataclass

APPROACHES = ("E", "W", "N", "S")
LaneKey = tuple[str, int]  # (approach, lane index)


@dataclass
class Car:
    """A queued car. position is slots from the stop line (0 = front)."""

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
        lane_capacity: int | None = None,
        seed: int | None = None,
    ) -> None:
        """Create an empty intersection; call populate_random() or populate_from().

        lane_capacity defaults to ceil(num_cars * capacity_fraction); an
        explicit lane_capacity takes precedence.
        """
        self.num_cars = num_cars
        self.lanes_per_approach = lanes_per_approach
        if lane_capacity is None:
            # round() removes float noise: 30 * 0.1 is 3.0000000000000004.
            lane_capacity = math.ceil(round(num_cars * capacity_fraction, 9))
        self.lane_capacity = lane_capacity

        if lane_capacity * self.num_lanes < num_cars:
            raise ValueError(
                f"total capacity {lane_capacity * self.num_lanes} "
                f"({self.num_lanes} lanes x {lane_capacity}) is less than "
                f"num_cars={num_cars}"
            )

        self._rng = random.Random(seed)
        self._queues: dict[LaneKey, list[Car]] = {key: [] for key in self.lanes}

    @property
    def lanes(self) -> list[LaneKey]:
        """All (approach, lane) keys."""
        return [(a, l) for a in APPROACHES for l in range(self.lanes_per_approach)]

    @property
    def num_lanes(self) -> int:
        """Total number of lanes."""
        return len(APPROACHES) * self.lanes_per_approach

    def _queue(self, approach: str, lane: int) -> list[Car]:
        """The queue for one lane; raises ValueError for an unknown lane."""
        if (approach, lane) not in self._queues:
            raise ValueError(f"no such lane ({approach!r}, {lane!r})")
        return self._queues[(approach, lane)]

    def _place(self, keys: list[LaneKey]) -> None:
        """Rebuild all queues; car i joins the back of lane keys[i]."""
        self._queues = {key: [] for key in self.lanes}
        for car_id, (approach, lane) in enumerate(keys):
            queue = self._queues[(approach, lane)]
            queue.append(Car(car_id, approach, lane, position=len(queue)))

    def populate_random(self) -> None:
        """Put every car in a random lane that still has room."""
        counts = dict.fromkeys(self.lanes, 0)
        keys: list[LaneKey] = []
        for _ in range(self.num_cars):
            open_lanes = [k for k in self.lanes if counts[k] < self.lane_capacity]
            key = self._rng.choice(open_lanes)
            counts[key] += 1
            keys.append(key)
        self._rng.shuffle(keys)  # so car ids aren't tied to the order lanes fill
        self._place(keys)

    def populate_from(self, assignment: list[tuple[str, int]]) -> None:
        """Build the layout from assignment[car_id] = (approach, lane).

        Within a lane, lower car ids are closer to the front. Raises
        ValueError, leaving the current state unchanged, if the layout is
        invalid or overfills a lane.
        """
        if len(assignment) != self.num_cars:
            raise ValueError(f"expected {self.num_cars} entries, got {len(assignment)}")
        keys = [tuple(entry) for entry in assignment]
        for key in keys:
            if key not in self._queues:
                raise ValueError(f"no such lane {key!r}")
        for key in set(keys):
            if keys.count(key) > self.lane_capacity:
                raise ValueError(
                    f"lane {key} overfilled: {keys.count(key)} cars, "
                    f"lane_capacity is {self.lane_capacity}"
                )
        self._place(keys)

    def get_car_location(self, car_id: int) -> tuple[str, int, int]:
        """Return (approach, lane, position) for a car in the intersection."""
        for (approach, lane), queue in self._queues.items():
            for car in queue:
                if car.id == car_id:
                    return (approach, lane, car.position)
        raise KeyError(f"car {car_id} is not in the intersection")

    def cars_in_lane(self, approach: str, lane: int) -> list[Car]:
        """Cars in one lane, front of the queue first."""
        return list(self._queue(approach, lane))

    def queue_length(self, approach: str, lane: int) -> int:
        """Number of cars in one lane."""
        return len(self._queue(approach, lane))

    def queue_lengths(self) -> dict[LaneKey, int]:
        """Queue length for every lane."""
        return {key: len(queue) for key, queue in self._queues.items()}

    def remove_front_car(self, approach: str, lane: int) -> Car:
        """Pop the front car, move the rest up one slot, and return the car."""
        queue = self._queue(approach, lane)
        if not queue:
            raise IndexError(f"lane ({approach!r}, {lane}) is empty")
        car = queue.pop(0)
        for behind in queue:
            behind.position -= 1
        return car

    def copy(self) -> "Intersection":
        """Independent deep copy (including the random generator's state)."""
        return deepcopy(self)

    def __str__(self) -> str:
        lines = [f"lane_capacity={self.lane_capacity} (front of queue on the left)"]
        for (approach, lane), queue in self._queues.items():
            cells = "".join(f"[{car.id}]" for car in queue) or "(empty)"
            lines.append(f"{approach} lane {lane}: {cells}")
        return "\n".join(lines)


if __name__ == "__main__":
    demo = Intersection(seed=42)
    demo.populate_random()
    print(f"lane_capacity = {demo.lane_capacity}")
    print(demo)
