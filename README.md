# CISC-455-Project

Evolving traffic-light timings for a four-way intersection with an evolutionary algorithm (EA), plus a second "outer" EA that tunes the first EA's hyperparameters.

## Application Problem

This evolutionary algorithm can be utilized to optimize the timing of a four-way traffic light intersection, in order to minimize the average time cars will spend waiting. 

The genotype of each light will represent the length of their advance green signal, forward green signal (which also accounts for right-hand turns), and red light signal. The genotypes will then be paired with constraints to ensure all traffic laws are obeyed. For instance, light scheduling will define that a North road cannot have a green light at the same time as an East or West road. Left turns may occur during either the advance green, or under regular green lights if there is no oncoming traffic.

The fitness of each candidate solution will be evaluated over N-tuned scenarios, where fitness is determined by the average time taken for the intersection to empty over all scenarios. This is done to ensure every candidate solution has an equal fitness evaluation, while ensuring enough diversity and no favouritism.

## Justification for an EA

An evolutionary algorithm is suitable for this problem because it involves the optimization of multiple interacting variables under constraints. Each individual's genotype contains 12 genes, representing the advanced green light duration, forward green light duration, and red light duration for all four traffic lights. The resulting phenotype is an intersection’s signal schedule. This problem may also support many locally optimal solutions, as some light setups may be more optimal for specific intersection configurations, leaving a lot of opportunity for an EA to find many optimal solutions without needing to exploit a single region. For example, one locally optimal solution which is relatively optimal for 70% of the intersections it is evaluated on may have the same fitness as a solution which is exceedingly optimal for the other 30%. In turn, this  directly implies vastly different genotypes can achieve the same high fitness levels.

Additionally, the fitness of a genotype depends on how its genes work together. A timing value that performs well in one genotype may perform poorly in another if other genes in that genotype have different durations. For example, if a genotype has a Northbound green light set to be active for 15 seconds, but the Southbound green light is set to be active for only 10, it would cause the cause moving towards the Southbound light to miss 5 seconds of potential usable green light time. This illustrates why a gene's effectiveness depends on the combination of all timing values, and why genes are unable to be optimized independently of each other.

## Files

- simulation.py: Generates random traffic scenarios and simulates one light-timing plan on one scenario 
- setup.py: Scenario settings and fitness weights; creates or loads `intersection_variants.json` 
- algorithm.py: Inner EA: a (μ + λ) evolution strategy with self-adaptive mutation 
- outer_algorithm.py: Outer EA: tunes the inner EA's hyperparameters and saves the best to `outer_results.json` 
- test_simulation.py: Unit tests for the simulator, scenario generation and EA operators 
- intersection_variants.json: The saved set of 50 traffic scenarios, so every run uses the same data 
- outer_results.json: Best hyperparameters found by the outer EA; `algorithm.py` reads this file if it exists

## 1. The intersection model: simulation.py

### Layout

- Four approaches: `N`, `S`, `E`, `W`, each with two lanes.
  - Lane `0`: left turns or straight.
  - Lane `1`: straight or right turns.
- A scenario is a dict keyed by `(direction, lane)`. Each value is a queue of cars sorted by arrival time:
  `{("N", 0): [{"turn": "left", "arrival": 18.3}, ...], ...}`

### Generating scenarios: `generate_variants`

Each car gets:
- a random direction, weighted by `direction_weights` (equal by default),
- a random turn, weighted by that direction's `turn_weights` (20% left, 60% straight, 20% right by default),
- a valid lane for that turn,
- an arrival time between 0 and `arrival_window` seconds.

A scenario is accepted only if every lane has between `min_cars` and `max_cars` cars. Otherwise the generator redraws it. A fixed seed makes the scenarios reproducible.

The saved scenarios use the settings in `setup.py`: 200 cars arriving over 240 s, with uneven traffic so that the timing genes have real trade-offs:

| Approach | Share of cars | Left / straight / right |
|---|---|---|
| N | 1/6 | 50 / 35 / 15 |
| S | 3/6 | 5 / 80 / 15 |
| E | 1/6 | 20 / 65 / 15 |
| W | 1/6 | 20 / 65 / 15 |

So N has many left turns facing a heavy oncoming stream from S.

### Simulating a timing plan: `simulate_traffic(intersection, individual)`

`individual` holds the timing plan. Its first 8 numbers are the genes, two per direction in the order N, S, E, W:
- [N_advanced, N_green, S_advanced, S_green, E_advanced, E_green, W_advanced, W_green, ...]

The signal cycle alternates between two axes, N/S and then E/W:

1. **Advanced green** (length `advanced`): only left turns from that direction may go. These are protected lefts.
2. **Regular green** (length `green`): straight and right turns may go. A left turn may also go, but it waits while an oncoming car is going straight or turning right. It does not wait for oncoming left-turners, so two opposing lefts can go together. This models a permissive left.
3. An axis phase lasts as long as the longer of its two directions (`advanced + green`). Then there is a `CLEARANCE_TIME` of 2 s (all red) before the other axis starts.

Movement rules:
- Each lane sends one car at a time, but cars whose paths don't cross move at the same time: opposite straights and rights, both lanes of one approach, and opposing lefts. Only a left turn and an oncoming straight or right turn conflict.
- When several cars can start at once, they start in this order: protected lefts, then straights and rights, then permissive lefts, earliest arrival first within each group.
- A car takes 2 s to go straight, 3 s to turn right, or 4 s to turn left.
- A car starts only if it has already arrived and can finish before its signal ends. A car that can't finish waits, but it no longer stops other cars from going.
- The cycle repeats until every car has cleared, up to `MAX_CYCLES` (100). If cars remain after that, the plan gets a large `PENALTY` score.

The function returns `completion_time`, `average_wait` and `max_wait`. It does not modify the scenario it is given.

## 2. Fixed scenarios and fitness settings: setup.py

`load_variants()` makes sure every run is tested on the same 50 scenarios:
- If `intersection_variants.json` exists and its saved `settings` match the constants in `setup.py`, it loads that file.
- If the file is missing, it generates the scenarios and saves them.
- If the settings differ, it raises an error. To make new scenarios, delete or rename the JSON file.

JSON keys can't be tuples, so the file stores lane keys as strings like `"N,0"`. `load_variants()` converts them back into tuples.

`setup.py` also sets the fitness weights. The EA minimises this score:
- score = 1.0 * completion_time + 1.0 * average_wait + 0.5 * max_wait

## 3. The inner EA: algorithm.py

This is a (μ + λ) evolution strategy with self-adaptive step sizes.

**Chromosome:** 16 floats. The first 8 are timing genes. The last 8 are each gene's mutation step size σ.
- Advanced-green genes are bounded to [0, 15] s.
- Regular-green genes are bounded to [5, 60] s.
- See `BOUNDS`.

**Fitness (`evaluate_fitness`):** the mean score across all 50 scenarios. Lower is better.

**One generation:**
1. **Parent selection:** tournament selection (`tournament_size`) fills a mating pool of `mating_pool_size` parents.
2. **Crossover:** two parents are drawn from the pool.
   - Timing genes use uniform crossover: each gene is copied from one parent or the other.
   - σ genes are blended: `alpha * p1 + (1 - alpha) * p2`.
3. **Mutation:** each σ is updated as `σ' = σ · exp(lr · N(0,1))`, with a floor of `sigma_bound`. Each timing gene then gets `N(0, σ')` noise and is clipped to its bounds.
4. Steps 2–3 repeat to produce `lamd` (λ) children.
5. **Survivor selection:** parents and children are pooled, and the best `mu` (μ) survive. This is (μ + λ) selection, so the best solution is never lost.

`run_algorithm(...)` runs this loop and returns `(best_solution, best_fitness)`. You can pass in scenarios and a seed, which is how the outer EA uses it.

When you run `algorithm.py` directly, it:
- uses the hyperparameters in `outer_results.json` if that file exists, or the built-in defaults otherwise,
- runs for 100 generations,
- prints the best timings, their σ values and the fitness.

## 4. The outer EA: outer_algorithm.py

This is a meta-EA. Each of its individuals is a set of gene hyperparameters for the inner EA:
- `mu`: Range = 10–60; Type = int 
- `lamd`: Range = 20–150; Type = int 
- `lr`: Range = 0.01–0.5; Type = float
- `tournament_size`: Range = 2–10; Type = int, capped at `mu`
- `crossover_alpha`: Range = 0–1; Type = float
- `mating_pool_size`: Range = 20–300; Type = int, rounded to an even number
- `sigma_init`: Range = 0.5–15; Type = float
- `sigma_bound`: Range = 0.001–1; Type = float

**Evaluating one configuration:**
1. The configuration gets a fixed budget of `INNER_EVALUATION_BUDGET` (300) fitness evaluations, so `generations = (300 - mu) // lamd`. This keeps the comparison fair: a large population can't win just by doing more work.
2. The inner EA runs `INNER_REPEATS` (2) times with fixed seeds.
3. The configuration's score is the mean of the best fitness from each run.
4. Scores are cached, so a repeated configuration isn't re-run.

**Search loop:**
- Population of 6, with 6 children per generation, for 4 generations.
- Parents are chosen by tournament, then combined with uniform crossover.
- Each child gene has a 30% chance of Gaussian mutation, with a step of 10% of that gene's range.
- Survivors are chosen by (μ + λ) selection.

The best configuration is then written to `outer_results.json`, and `algorithm.py` then picks it up automatically.

## Tests: test_simulation.py

The tests check:
- **Scenario generation:** 50 scenarios, 30 cars each, 1–8 cars per lane, valid lane/turn pairs, sorted arrival times, and reproducible results from a seed.
- **Traffic weights:** direction and turn weights are respected, invalid weights are rejected, and the saved scenarios match the uneven settings in `setup.py`.
- **Simulator behaviour on hand-built cases:**
  - an empty intersection,
  - a single car going straight,
  - a car arriving later,
  - opposite straights moving together, both lanes of one approach moving together, and one car at a time within a lane,
  - a protected left on the advanced green, and a protected left going before an oncoming straight,
  - a permissive left waiting for oncoming traffic, and opposing lefts not deadlocking,
  - a car that can't finish not blocking others,
  - a green too short for a car to finish.
- **EA operators:** chromosome length and bounds after crossover and mutation, the σ floor, and that fitness is the deterministic mean over scenarios.
