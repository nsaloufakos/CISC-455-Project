"""Unit tests for intersection.py. Run with: python3 -m unittest -v"""

import random
import unittest
from typing import Dict, List

from intersection import APPROACHES, Intersection, LaneKey

def layout(inter: Intersection) -> Dict[LaneKey, List[int]]:
    """Car ids per lane, front first, for comparing two intersections."""
    return {key: [c.id for c in inter.cars_in_lane(*key)] for key in inter.lanes}


class IntersectionTestCase(unittest.TestCase):
    def assertValidState(self, inter: Intersection, expected_cars: int) -> None:
        """Check every invariant of the model in one place."""
        seen_ids = set()
        for approach, lane in inter.lanes:
            cars = inter.cars_in_lane(approach, lane)
            positions = [c.position for c in cars]
            # Unique and contiguous from 0 (no duplicates, no gaps).
            self.assertEqual(positions, list(range(len(cars))), (approach, lane))
            self.assertLessEqual(len(cars), inter.lane_capacity)
            for car in cars:
                self.assertEqual((car.approach, car.lane), (approach, lane))
                self.assertEqual(
                    inter.get_car_location(car.id), (approach, lane, car.position)
                )
                self.assertNotIn(car.id, seen_ids)
                seen_ids.add(car.id)
        self.assertEqual(len(seen_ids), expected_cars)
        self.assertEqual(sum(inter.queue_lengths().values()), expected_cars)


class TestPlacement(IntersectionTestCase):
    def test_all_cars_placed_random(self):
        for seed in range(50):
            inter = Intersection(seed=seed)
            inter.populate_random()
            self.assertValidState(inter, 30)
            for car_id in range(30):
                inter.get_car_location(car_id)  # raises KeyError if missing

    def test_all_cars_placed_from_assignment(self):
        inter = Intersection()
        assignment = [(APPROACHES[i % 4], i % 2) for i in range(30)]
        inter.populate_from(assignment)
        self.assertValidState(inter, 30)
        for car_id, (approach, lane) in enumerate(assignment):
            self.assertEqual(inter.get_car_location(car_id)[:2], (approach, lane))

    def test_no_duplicate_positions_in_a_lane(self):
        for seed in range(50):
            inter = Intersection(seed=seed)
            inter.populate_random()
            for key in inter.lanes:
                positions = [c.position for c in inter.cars_in_lane(*key)]
                self.assertEqual(len(positions), len(set(positions)))

    def test_random_respects_tight_capacity(self):
        # ceil(30 * 0.125) = 4 per lane, 32 slots for 30 cars.
        for seed in range(200):
            inter = Intersection(capacity_fraction=0.125, seed=seed)
            inter.populate_random()
            self.assertValidState(inter, 30)
            self.assertLessEqual(max(inter.queue_lengths().values()), 4)

    def test_lopsided_layout_allowed_by_default(self):
        inter = Intersection()
        inter.populate_from([("E", 0)] * 30)
        self.assertEqual(inter.queue_length("E", 0), 30)
        self.assertValidState(inter, 30)

    def test_populate_replaces_previous_state(self):
        inter = Intersection(seed=1)
        inter.populate_random()
        inter.populate_from([("S", 1)] * 30)
        self.assertEqual(inter.queue_lengths()[("S", 1)], 30)
        self.assertValidState(inter, 30)

    def test_queue_lengths_covers_all_lanes(self):
        inter = Intersection(seed=3)
        inter.populate_random()
        lengths = inter.queue_lengths()
        self.assertEqual(set(lengths), {(a, l) for a in APPROACHES for l in (0, 1)})
        for key, n in lengths.items():
            self.assertEqual(inter.queue_length(*key), n)

    def test_bad_assignment_rejected(self):
        inter = Intersection()
        with self.assertRaises(ValueError):
            inter.populate_from([("E", 0)] * 29)  # wrong length
        with self.assertRaises(ValueError):
            inter.populate_from([("X", 0)] + [("E", 0)] * 29)  # bad approach
        with self.assertRaises(ValueError):
            inter.populate_from([("E", 2)] + [("E", 0)] * 29)  # bad lane

    def test_str_diagram(self):
        inter = Intersection(num_cars=3)
        inter.populate_from([("N", 0), ("N", 0), ("W", 1)])
        text = str(inter)
        self.assertIn("N lane 0: [0][1]", text)
        self.assertIn("W lane 1: [2]", text)
        self.assertIn("E lane 0: (empty)", text)


class TestRemoveFrontCar(IntersectionTestCase):
    def test_contiguous_after_each_removal(self):
        inter = Intersection(seed=7)
        inter.populate_from([("N", 1)] * 10 + [("E", 0)] * 20)
        remaining = 30
        while inter.queue_length("E", 0):
            front = inter.cars_in_lane("E", 0)[0]
            second = inter.cars_in_lane("E", 0)[1:2]
            removed = inter.remove_front_car("E", 0)
            remaining -= 1
            self.assertIs(removed, front)
            self.assertValidState(inter, remaining)
            with self.assertRaises(KeyError):
                inter.get_car_location(removed.id)
            if second:
                self.assertEqual(inter.get_car_location(second[0].id), ("E", 0, 0))
        self.assertEqual(inter.queue_length("N", 1), 10)  # other lanes untouched

    def test_random_layout_drain(self):
        inter = Intersection(seed=11)
        inter.populate_random()
        remaining = 30
        for key in inter.lanes:
            while inter.queue_length(*key):
                inter.remove_front_car(*key)
                remaining -= 1
                self.assertValidState(inter, remaining)
        self.assertEqual(remaining, 0)

    def test_empty_lane_raises(self):
        inter = Intersection()
        inter.populate_from([("E", 0)] * 30)
        with self.assertRaises(IndexError):
            inter.remove_front_car("W", 0)


class TestReproducibility(unittest.TestCase):
    def test_same_seed_same_layout(self):
        a, b = Intersection(seed=42), Intersection(seed=42)
        a.populate_random()
        b.populate_random()
        self.assertEqual(layout(a), layout(b))
        self.assertEqual(str(a), str(b))

    def test_different_seeds_differ(self):
        a, b = Intersection(seed=1), Intersection(seed=2)
        a.populate_random()
        b.populate_random()
        self.assertNotEqual(layout(a), layout(b))

    def test_global_random_untouched(self):
        random.seed(123)
        before = random.getstate()
        inter = Intersection(seed=5)
        inter.populate_random()
        inter.copy().populate_random()
        self.assertEqual(random.getstate(), before)

    def test_copy_is_independent(self):
        original = Intersection(seed=9)
        original.populate_random()
        snapshot = layout(original)
        clone = original.copy()
        self.assertEqual(layout(clone), snapshot)
        key = max(clone.lanes, key=lambda k: clone.queue_length(*k))
        clone.remove_front_car(*key)
        self.assertEqual(layout(original), snapshot)
        self.assertNotEqual(layout(clone), snapshot)

    def test_copy_continues_same_random_stream(self):
        a = Intersection(seed=4)
        a.populate_random()
        b = a.copy()
        a.populate_random()
        b.populate_random()
        self.assertEqual(layout(a), layout(b))


class TestCapacity(unittest.TestCase):
    def test_default_capacity_equals_num_cars(self):
        self.assertEqual(Intersection().lane_capacity, 30)
        self.assertEqual(Intersection(num_cars=10).lane_capacity, 10)
        self.assertEqual(Intersection(num_cars=100).lane_capacity, 100)

    def test_capacity_scales_with_fraction(self):
        cases = [
            (30, 0.5, 15),
            (31, 0.5, 16),   # ceil(15.5)
            (30, 0.25, 8),   # ceil(7.5)
            (30, 0.125, 4),  # ceil(3.75)
            (40, 0.25, 10),
        ]
        for num_cars, fraction, expected in cases:
            with self.subTest(num_cars=num_cars, fraction=fraction):
                inter = Intersection(num_cars=num_cars, capacity_fraction=fraction)
                self.assertEqual(inter.lane_capacity, expected)

    def test_fraction_has_no_float_rounding_error(self):
        # 30 * 0.1 is 3.0000000000000004 as a float; capacity must be 3, not 4.
        inter = Intersection(num_cars=30, capacity_fraction=0.1, lanes_per_approach=3)
        self.assertEqual(inter.lane_capacity, 3)

    def test_explicit_override_takes_precedence(self):
        inter = Intersection(capacity_fraction=0.5, lane_capacity=5)
        self.assertEqual(inter.lane_capacity, 5)
        self.assertEqual(Intersection(lane_capacity=12).lane_capacity, 12)

    def test_total_capacity_too_small(self):
        with self.assertRaises(ValueError):
            Intersection(num_cars=30, lane_capacity=3)  # 8 * 3 = 24 < 30
        with self.assertRaises(ValueError):
            Intersection(num_cars=30, capacity_fraction=0.1)  # 8 * 3 = 24 < 30
        Intersection(num_cars=30, lane_capacity=4)  # 32 >= 30 is fine

    def test_populate_from_overfilled_lane(self):
        inter = Intersection(capacity_fraction=0.5)  # 15 per lane
        inter.populate_from([("E", 0)] * 15 + [("W", 1)] * 15)  # exactly full: ok
        before = layout(inter)
        with self.assertRaises(ValueError):
            inter.populate_from([("E", 0)] * 16 + [("W", 1)] * 14)
        self.assertEqual(layout(inter), before)  # failed call changed nothing

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            Intersection(capacity_fraction=0)
        with self.assertRaises(ValueError):
            Intersection(lanes_per_approach=0)
        with self.assertRaises(ValueError):
            Intersection(num_cars=-1)


if __name__ == "__main__":
    unittest.main()
