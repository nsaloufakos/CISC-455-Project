
import unittest
from copy import deepcopy
from simulation import generate_variants, simulate_traffic
from algorithm import evaluate_fitness, initalize_population, mutate, crossover, BOUNDS
from setup import load_variants, COMPLETION_WEIGHT, AVERAGE_WAIT_WEIGHT, MAX_WAIT_WEIGHT

class TestTrafficSimulation(unittest.TestCase):
    def setUp(self):
        self.chromosome = [5.0, 20.0] * 4 + [5.0] * 8
        self.empty = {(d, l): [] for d in ("N", "S", "E", "W") for l in range(2)}

    def test_fifty_fixed_variants_with_constraints(self):
        variants = generate_variants(seed=17)
        self.assertEqual(len(variants), 50)
        for variant in variants:
            self.assertEqual(sum(map(len, variant.values())), 30)
            self.assertTrue(all(1 <= len(q) <= 8 for q in variant.values()))
            self.assertTrue(all(car["turn"] in ("left", "straight") for (d, l), q in variant.items() if l == 0 for car in q))
            self.assertTrue(all(car["turn"] in ("straight", "right") for (d, l), q in variant.items() if l == 1 for car in q))
            self.assertTrue(all(0 <= car["arrival"] <= 120 for q in variant.values() for car in q))
            self.assertTrue(all(q[i]["arrival"] <= q[i + 1]["arrival"] for q in variant.values() for i in range(len(q) - 1)))
        self.assertEqual(variants, generate_variants(seed=17))

    def test_saved_variants_are_reused(self):
        self.assertEqual(load_variants(), load_variants())

    def test_input_unchanged(self):
        variant = generate_variants(num_variants=1)[0]
        before = deepcopy(variant)
        self.assertGreater(simulate_traffic(variant, self.chromosome)["completion_time"], 0)
        self.assertEqual(variant, before)

    def test_empty_intersection(self):
        self.assertEqual(simulate_traffic(self.empty, self.chromosome)["completion_time"], 0)

    def test_single_straight_crossing(self):
        self.empty[("N", 0)] = [{"turn": "straight", "arrival": 0.0}]
        self.assertEqual(simulate_traffic(self.empty, self.chromosome)["completion_time"], 7.0)
        self.assertEqual(simulate_traffic(self.empty, self.chromosome)["average_wait"], 5.0)

    def test_dynamic_arrival(self):
        self.empty[("N", 0)] = [{"turn": "straight", "arrival": 10.0}]
        result = simulate_traffic(self.empty, self.chromosome)
        self.assertEqual(result["completion_time"], 12.0)
        self.assertEqual(result["average_wait"], 0.0)

    def test_opposite_straights_move_together(self):
        self.empty[("N", 0)] = [{"turn": "straight", "arrival": 0.0}]
        self.empty[("S", 0)] = [{"turn": "straight", "arrival": 0.0}]
        self.assertEqual(simulate_traffic(self.empty, self.chromosome)["completion_time"], 7.0)

    def test_both_lanes_of_one_approach_move_together(self):
        self.empty[("N", 0)] = [{"turn": "straight", "arrival": 0.0}]
        self.empty[("N", 1)] = [{"turn": "right", "arrival": 0.0}]
        self.assertEqual(simulate_traffic(self.empty, self.chromosome)["completion_time"], 8.0)

    def test_one_car_at_a_time_per_lane(self):
        self.empty[("N", 0)] = [{"turn": "straight", "arrival": 0.0}] * 2
        self.assertEqual(simulate_traffic(self.empty, self.chromosome)["completion_time"], 9.0)

    def test_opposing_lefts_do_not_deadlock(self):
        self.empty[("N", 0)] = [{"turn": "left", "arrival": 0.0}]
        self.empty[("S", 0)] = [{"turn": "left", "arrival": 0.0}]
        times = [0.0, 20.0] * 4 + [5.0] * 8
        self.assertEqual(simulate_traffic(self.empty, times)["completion_time"], 4.0)

    def test_protected_left_goes_before_oncoming_straight(self):
        self.empty[("N", 0)] = [{"turn": "left", "arrival": 0.0}]
        self.empty[("S", 0)] = [{"turn": "straight", "arrival": 0.0}]
        times = [5.0, 20.0, 0.0, 25.0] + [5.0, 20.0] * 2 + [5.0] * 8
        result = simulate_traffic(self.empty, times)
        self.assertEqual(result["completion_time"], 6.0)  # left 0-4, then straight 4-6
        self.assertEqual(result["max_wait"], 4.0)

    def test_car_that_cannot_finish_does_not_block_others(self):
        # N's green ends at 5, so its right turn (3s) arriving at 3 must wait for
        # the next cycle; S's straight arriving at the same moment still goes.
        self.empty[("N", 1)] = [{"turn": "right", "arrival": 3.0}]
        self.empty[("S", 0)] = [{"turn": "straight", "arrival": 3.0}]
        times = [0.0, 5.0, 0.0, 20.0, 0.0, 10.0, 0.0, 10.0] + [5.0] * 8
        result = simulate_traffic(self.empty, times)
        self.assertEqual(result["completion_time"], 37.0)  # next N green at 20+2+10+2 = 34
        self.assertEqual(result["average_wait"], 15.5)  # S waits 0, N waits 31

    def test_advanced_left_turn(self):
        self.empty[("N", 0)] = [{"turn": "left", "arrival": 0.0}]
        self.assertEqual(simulate_traffic(self.empty, self.chromosome)["completion_time"], 4.0)

    def test_left_waits_for_opposite_queue(self):
        self.empty[("N", 0)] = [{"turn": "left", "arrival": 0.0}]
        self.empty[("S", 0)] = [{"turn": "straight", "arrival": 0.0}]
        times = [0.0, 20.0] * 4 + [5.0] * 8
        self.assertEqual(simulate_traffic(self.empty, times)["completion_time"], 6.0)

    def test_short_green_does_not_allow_unfinished_car(self):
        self.empty[("N", 0)] = [{"turn": "straight", "arrival": 0.0}] * 3
        times = [0.0, 5.0] * 4 + [5.0] * 8
        self.assertEqual(simulate_traffic(self.empty, times)["completion_time"], 16.0)

    def test_mutation_and_crossover_shapes(self):
        parent1, parent2 = initalize_population(2, BOUNDS, 5.0)
        child = crossover(parent1, parent2)
        child = mutate(child, BOUNDS, 0.1, 0.01)
        self.assertEqual(len(child), 16)
        self.assertTrue(all(low <= child[i] <= high for i, (low, high) in enumerate(BOUNDS)))
        self.assertTrue(all(s >= 0.01 for s in child[8:]))

    def test_fitness_mean_and_reproducibility(self):
        variants = generate_variants(num_variants=3, seed=8)
        scores = evaluate_fitness([self.chromosome], variants)
        expected = sum(COMPLETION_WEIGHT * (r := simulate_traffic(v, self.chromosome))["completion_time"]
                       + AVERAGE_WAIT_WEIGHT * r["average_wait"] + MAX_WAIT_WEIGHT * r["max_wait"] for v in variants) / 3
        self.assertAlmostEqual(scores[0], expected)
        self.assertEqual(scores, evaluate_fitness([self.chromosome], variants))

if __name__ == "__main__":
    unittest.main()