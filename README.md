# SugarScape-Style Multi-Agent Simulation with Dual-Objective Emotional Switching

This repository provides a modular Python simulation inspired by SugarScape where decentralized agents adapt behavior under a global emotional switching signal:

- Growth objective proxy: favor actions aligned with maximizing aggregate gains
- Survival objective proxy: favor safe actions that protect the system floor (`min A_i`)

The switching signal is computed each timestep from system floor pressure:

- `A_min = min_i A_i`
- `S = 1 - normalized(A_min)`
- `lambda = 1 / (1 + exp(k * (S - theta)))`

Agents use weighted action scoring:

- `score = lambda * U_growth + (1 - lambda) * U_survival`

## Features

- 2D grid with regenerating sugar resources
- Agents with metabolism, vision, movement, harvesting, and death
- Dynamic floor-sensitive switching (or fixed lambda for baselines)
- Internal agent model:
  - bounded low-dimensional preference vector
  - spatial + outcome memory with decay / fixed-size behavior
  - stress-dependent learning updates
- Built-in experiments:
  1. Baseline pure growth (`lambda=1`)
  2. Baseline pure survival (`lambda=0`)
  3. Dynamic emotional switching
  4. Floor intervention (target bottom quantile)
  5. Stress test with resource shock
- Metrics tracked per timestep:
  - floor resource
  - average resource
  - inequality (Gini)
  - lambda
  - alive agents
  - growth vs survival mode ratios

## Run

```bash
python sugarscape_sim.py --steps 250 --seed 42 --output outputs
```

## Outputs

The script saves plots in the output directory:

- `floor_vs_time.png`
- `lambda_vs_time.png`
- `agent_distribution.png`

It also prints summary stats for each experiment.

## Notes

- The simulation is intentionally compact and interpretable.
- Parameters are centralized in `SimulationConfig` for easy tuning.
- The architecture is modular through `Environment`, `Agent`, and `Simulation` classes.
