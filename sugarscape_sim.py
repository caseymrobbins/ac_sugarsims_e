import argparse
import dataclasses
import importlib.util
import math
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


Coord = Tuple[int, int]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = int(clamp(q, 0.0, 1.0) * (len(arr) - 1))
    return arr[idx]


def toroidal_distance(a: Coord, b: Coord, width: int, height: int) -> int:
    dx = min((a[0] - b[0]) % width, (b[0] - a[0]) % width)
    dy = min((a[1] - b[1]) % height, (b[1] - a[1]) % height)
    return dx + dy


def gini(values: Sequence[float]) -> float:
    arr = [max(0.0, v) for v in values]
    if not arr:
        return 0.0
    s = sum(arr)
    if s <= 0:
        return 0.0
    arr.sort()
    n = len(arr)
    cum = 0.0
    for i, v in enumerate(arr, 1):
        cum += i * v
    return (2 * cum) / (n * s) - (n + 1) / n


@dataclass
class Environment:
    width: int
    height: int
    max_capacity: float
    regen_rate: float
    sugar: List[List[float]] = field(init=False)

    def __post_init__(self) -> None:
        self.sugar = [
            [random.uniform(0.3 * self.max_capacity, self.max_capacity) for _ in range(self.width)]
            for _ in range(self.height)
        ]

    def regenerate(self) -> None:
        for y in range(self.height):
            for x in range(self.width):
                self.sugar[y][x] = min(self.max_capacity, self.sugar[y][x] + self.regen_rate)

    def harvest(self, coord: Coord) -> float:
        x, y = coord
        amt = self.sugar[y][x]
        self.sugar[y][x] = 0.0
        return amt

    def apply_shock(self, severity: float = 0.5) -> None:
        factor = max(0.0, 1.0 - severity)
        for y in range(self.height):
            for x in range(self.width):
                self.sugar[y][x] *= factor


@dataclass
class Agent:
    agent_id: int
    x: int
    y: int
    resource: float
    metabolism: float
    vision: int
    memory_size: int = 25
    learning_rate: float = 0.08

    alive: bool = True
    last_mode: str = "growth"
    preference: Dict[str, float] = field(default_factory=dict)
    spatial_memory: Dict[Coord, float] = field(default_factory=dict)
    outcome_memory: deque = field(default_factory=lambda: deque(maxlen=25))

    def __post_init__(self) -> None:
        if not self.preference:
            self.preference = {
                "resource": random.uniform(0.3, 0.9),
                "risk": random.uniform(0.2, 0.8),
                "social": random.uniform(0.1, 0.6),
                "horizon": random.uniform(0.2, 0.8),
            }
        self._normalize_preferences()

    @property
    def coord(self) -> Coord:
        return (self.x, self.y)

    def _normalize_preferences(self) -> None:
        for k in list(self.preference.keys()):
            self.preference[k] = clamp(self.preference[k], 0.0, 1.0)

    def _visible_cells(self, env: Environment) -> List[Coord]:
        cells = []
        for dx in range(-self.vision, self.vision + 1):
            for dy in range(-self.vision, self.vision + 1):
                if abs(dx) + abs(dy) <= self.vision:
                    x = (self.x + dx) % env.width
                    y = (self.y + dy) % env.height
                    cells.append((x, y))
        return cells

    def _novelty(self, coord: Coord) -> float:
        return 1.0 if coord not in self.spatial_memory else 0.2

    def _certainty(self, coord: Coord, env: Environment) -> float:
        observed = env.sugar[coord[1]][coord[0]]
        mem = self.spatial_memory.get(coord, observed)
        return max(0.0, 1.0 - abs(observed - mem) / max(1e-6, observed + 1e-6))

    def choose_action(self, env: Environment, lambda_growth: float, agent_positions: Dict[Coord, int]) -> Coord:
        best_score = -1e18
        best = self.coord
        for c in self._visible_cells(env):
            d = toroidal_distance(self.coord, c, env.width, env.height)
            gain = env.sugar[c[1]][c[0]]
            novelty = self._novelty(c)
            crowd_penalty = 1.0 if c in agent_positions and c != self.coord else 0.0
            certainty = self._certainty(c, env)
            risk = d / max(1, self.vision) + 0.8 * crowd_penalty

            u_growth = (
                self.preference["resource"] * gain
                + self.preference["risk"] * novelty
                + self.preference["social"] * (1.0 - crowd_penalty)
                + self.preference["horizon"] * (gain / (1 + d))
            )
            u_survival = 1.8 * certainty + gain / (1 + d) - 1.2 * risk
            score = lambda_growth * u_growth + (1.0 - lambda_growth) * u_survival

            if score > best_score:
                best_score = score
                best = c

        self.last_mode = "growth" if random.random() < lambda_growth else "survival"
        return best

    def apply_learning(self, reward: float, pressure: float) -> None:
        stress_scale = max(0.15, 1.0 - pressure)
        lr = self.learning_rate * stress_scale
        self.preference["resource"] += lr * reward
        self.preference["risk"] += lr * (0.5 * reward - pressure * 0.3)
        self.preference["social"] += lr * (0.2 * reward)
        self.preference["horizon"] += lr * (reward - pressure * 0.4)

        if pressure > 0.6:
            self.preference["risk"] *= 0.85
            self.preference["horizon"] *= 0.90
        self._normalize_preferences()

    def step(self, env: Environment, new_coord: Coord, pressure: float) -> None:
        if not self.alive:
            return
        move_cost = 0.05 * toroidal_distance(self.coord, new_coord, env.width, env.height)
        self.x, self.y = new_coord
        harvested = env.harvest(self.coord)

        before = self.resource
        self.resource += harvested
        self.resource -= self.metabolism + move_cost
        reward = (self.resource - before) / max(1.0, before)

        self.outcome_memory.append((new_coord, reward))
        self.spatial_memory[new_coord] = harvested

        if len(self.spatial_memory) > self.memory_size:
            worst = min(self.spatial_memory.keys(), key=lambda c: self.spatial_memory[c])
            del self.spatial_memory[worst]

        for c in list(self.spatial_memory.keys()):
            self.spatial_memory[c] *= 0.98
            if self.spatial_memory[c] < 0.05:
                del self.spatial_memory[c]

        self.apply_learning(reward, pressure)
        if self.resource <= 0:
            self.alive = False


@dataclass
class SimulationConfig:
    width: int = 30
    height: int = 30
    n_agents: int = 80
    steps: int = 250
    max_capacity: float = 8.0
    regen_rate: float = 0.9
    initial_resource: Tuple[float, float] = (8.0, 14.0)
    metabolism_range: Tuple[float, float] = (0.7, 1.4)
    vision_range: Tuple[int, int] = (1, 5)
    k_switch: float = 10.0
    theta: float = 0.45
    dynamic_switching: bool = True
    fixed_lambda: Optional[float] = None
    floor_intervention: bool = False
    intervention_amount: float = 3.0
    intervention_quantile: float = 0.1
    shock_step: Optional[int] = None
    shock_severity: float = 0.5
    seed: int = 42


class Simulation:
    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        random.seed(cfg.seed)
        self.env = Environment(cfg.width, cfg.height, cfg.max_capacity, cfg.regen_rate)
        self.agents: List[Agent] = []
        self.time = 0
        self.history: Dict[str, List[float]] = {
            "floor": [], "avg": [], "gini": [], "lambda": [], "alive": [], "growth_ratio": [], "survival_ratio": []
        }
        self._spawn_agents()

    def _spawn_agents(self) -> None:
        for i in range(self.cfg.n_agents):
            self.agents.append(
                Agent(
                    agent_id=i,
                    x=random.randrange(self.cfg.width),
                    y=random.randrange(self.cfg.height),
                    resource=random.uniform(*self.cfg.initial_resource),
                    metabolism=random.uniform(*self.cfg.metabolism_range),
                    vision=random.randint(*self.cfg.vision_range),
                )
            )

    def _alive_agents(self) -> List[Agent]:
        return [a for a in self.agents if a.alive]

    def _compute_lambda(self, floor: float) -> Tuple[float, float]:
        norm_floor = clamp(floor / self.cfg.initial_resource[1], 0.0, 1.0)
        pressure = 1.0 - norm_floor
        lam = sigmoid(-self.cfg.k_switch * (pressure - self.cfg.theta)) if self.cfg.dynamic_switching else (1.0 if self.cfg.fixed_lambda is None else self.cfg.fixed_lambda)
        return clamp(lam, 0.0, 1.0), pressure

    def _apply_floor_intervention(self) -> None:
        alive = self._alive_agents()
        cutoff = quantile([a.resource for a in alive], self.cfg.intervention_quantile)
        for a in alive:
            if a.resource <= cutoff:
                a.resource += self.cfg.intervention_amount

    def step(self) -> None:
        self.time += 1
        alive = self._alive_agents()
        if not alive:
            return

        lam, pressure = self._compute_lambda(min(a.resource for a in alive))

        if self.cfg.shock_step is not None and self.time == self.cfg.shock_step:
            self.env.apply_shock(self.cfg.shock_severity)
        if self.cfg.floor_intervention:
            self._apply_floor_intervention()

        positions = {a.coord: a.agent_id for a in alive}
        chosen = {a.agent_id: a.choose_action(self.env, lam, positions) for a in alive}

        growth_count = 0
        survival_count = 0
        for a in alive:
            a.step(self.env, chosen[a.agent_id], pressure)
            if a.last_mode == "growth":
                growth_count += 1
            else:
                survival_count += 1

        self.env.regenerate()
        alive2 = self._alive_agents()
        resources = [a.resource for a in alive2]

        self.history["floor"].append(min(resources) if resources else 0.0)
        self.history["avg"].append(mean(resources))
        self.history["gini"].append(gini(resources))
        self.history["lambda"].append(lam)
        self.history["alive"].append(float(len(alive2)))
        total = max(1, growth_count + survival_count)
        self.history["growth_ratio"].append(growth_count / total)
        self.history["survival_ratio"].append(survival_count / total)

    def run(self) -> Dict[str, List[float]]:
        for _ in range(self.cfg.steps):
            if not self._alive_agents():
                break
            self.step()
        return self.history


def write_csv(histories: Dict[str, Dict[str, List[float]]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    header = "experiment,timestep,floor,avg,gini,lambda,alive,growth_ratio,survival_ratio\n"
    lines = [header]
    for name, h in histories.items():
        n = len(h["floor"])
        for t in range(n):
            lines.append(
                f"{name},{t},{h['floor'][t]:.6f},{h['avg'][t]:.6f},{h['gini'][t]:.6f},{h['lambda'][t]:.6f},{h['alive'][t]:.0f},{h['growth_ratio'][t]:.6f},{h['survival_ratio'][t]:.6f}\n"
            )
    (out_dir / "metrics.csv").write_text("".join(lines))


def make_plots(histories: Dict[str, Dict[str, List[float]]], out_dir: Path) -> bool:
    has_matplotlib = importlib.util.find_spec("matplotlib") is not None
    if not has_matplotlib:
        return False

    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 5))
    for name, h in histories.items():
        plt.plot(h["floor"], label=name)
    plt.title("Floor vs time")
    plt.xlabel("t")
    plt.ylabel("min resource")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "floor_vs_time.png", dpi=150)
    plt.close()

    plt.figure(figsize=(11, 5))
    for name, h in histories.items():
        plt.plot(h["lambda"], label=name)
    plt.title("Lambda vs time")
    plt.xlabel("t")
    plt.ylabel("lambda")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "lambda_vs_time.png", dpi=150)
    plt.close()

    plt.figure(figsize=(11, 5))
    widths = []
    labels = []
    for name, h in histories.items():
        vals = [v for v in h["avg"] if v > 0]
        widths.append(sum(vals) / len(vals) if vals else 0.0)
        labels.append(name)
    plt.bar(labels, widths)
    plt.title("Agent distribution proxy (mean avg resources)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "agent_distribution.png", dpi=150)
    plt.close()
    return True


def summarize(name: str, history: Dict[str, List[float]]) -> str:
    if not history["alive"]:
        return f"{name}: no surviving trajectory"
    i = -1
    return (
        f"{name}: alive_end={history['alive'][i]:.0f}, floor_end={history['floor'][i]:.2f}, avg_end={history['avg'][i]:.2f}, "
        f"gini_end={history['gini'][i]:.2f}, growth_ratio_end={history['growth_ratio'][i]:.2f}, lambda_end={history['lambda'][i]:.2f}"
    )


def run_experiments(args: argparse.Namespace) -> Dict[str, Dict[str, List[float]]]:
    base = SimulationConfig(steps=args.steps, seed=args.seed)
    return {
        "baseline_growth": Simulation(dataclasses.replace(base, dynamic_switching=False, fixed_lambda=1.0)).run(),
        "baseline_survival": Simulation(dataclasses.replace(base, dynamic_switching=False, fixed_lambda=0.0)).run(),
        "dynamic_switching": Simulation(dataclasses.replace(base, dynamic_switching=True)).run(),
        "floor_intervention": Simulation(dataclasses.replace(base, dynamic_switching=True, floor_intervention=True)).run(),
        "stress_test": Simulation(dataclasses.replace(base, dynamic_switching=True, shock_step=max(10, args.steps // 2), shock_severity=0.65)).run(),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("outputs"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    histories = run_experiments(args)
    write_csv(histories, args.output)
    plotted = make_plots(histories, args.output)

    print("Experiment summaries")
    print("=" * 80)
    for name, hist in histories.items():
        print(summarize(name, hist))
    print(f"CSV written to: {args.output / 'metrics.csv'}")
    if plotted:
        print(f"Plots written to: {args.output}")
    else:
        print("matplotlib not available; skipped PNG plot generation (CSV metrics still generated).")


if __name__ == "__main__":
    main()
