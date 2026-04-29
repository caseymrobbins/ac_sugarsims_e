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


def safe_log(value: float, eps: float = 1e-6) -> float:
    return math.log(max(value, eps))


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
    money: float
    goods: float
    goods_quality: float
    metabolism: float
    vision: int
    memory_size: int = 25
    learning_rate: float = 0.08

    alive: bool = True
    last_mode: str = "growth"
    preference: Dict[str, float] = field(default_factory=dict)
    spatial_memory: Dict[Coord, float] = field(default_factory=dict)
    outcome_memory: deque = field(default_factory=lambda: deque(maxlen=25))
    trust: Dict[str, float] = field(default_factory=dict)
    knowledge: float = 0.5
    debt: float = 0.0
    objective_lambda: float = field(default_factory=lambda: random.uniform(0.35, 0.85))
    campaign_bias: float = field(default_factory=lambda: random.uniform(-0.15, 0.15))

    def __post_init__(self) -> None:
        if not self.preference:
            self.preference = {
                "resource": random.uniform(0.3, 0.9),
                "money": random.uniform(0.2, 0.8),
                "goods": random.uniform(0.2, 0.8),
                "quality": random.uniform(0.2, 0.8),
                "risk": random.uniform(0.2, 0.8),
                "social": random.uniform(0.1, 0.6),
                "horizon": random.uniform(0.2, 0.8),
            }
        if not self.trust:
            self.trust = {
                "neighbors": random.uniform(0.2, 0.9),
                "news": random.uniform(0.1, 0.9),
                "leader": random.uniform(0.1, 0.9),
                "bank": random.uniform(0.2, 0.9),
                "market": random.uniform(0.2, 0.9),
            }
        self._normalize_preferences()

    @property
    def coord(self) -> Coord:
        return (self.x, self.y)

    def _normalize_preferences(self) -> None:
        for k in list(self.preference.keys()):
            self.preference[k] = clamp(self.preference[k], 0.0, 1.0)
        for k in list(self.trust.keys()):
            self.trust[k] = clamp(self.trust[k], 0.0, 1.0)
        self.knowledge = clamp(self.knowledge, 0.0, 1.0)
        self.objective_lambda = clamp(self.objective_lambda, 0.0, 1.0)

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

            quality_estimate = (self.goods_quality + 0.1 * self.knowledge) * (0.8 + 0.2 * self.trust["market"])
            u_growth = (
                self.preference["resource"] * gain
                + self.preference["money"] * self.money
                + self.preference["goods"] * self.goods
                + self.preference["quality"] * quality_estimate
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

    def objective_value(self, lambda_switch: float, eps: float = 1e-6) -> float:
        vital_min = min(self.resource, self.money, self.goods, self.goods_quality)
        if vital_min <= 0:
            needs_utility = -1e12
        else:
            needs_utility = math.log(vital_min)

        resource_term = max(self.resource + eps, eps)
        money_term = max(self.money + eps, eps)
        goods_term = max(self.goods + eps, eps)
        quality_term = max(self.goods_quality + eps, eps)
        sum_logs = (
            math.log(resource_term)
            + math.log(money_term)
            + math.log(goods_term)
            + math.log(quality_term)
        )
        return lambda_switch * needs_utility + (1.0 - lambda_switch) * sum_logs

    def survival_check(self) -> None:
        if self.resource <= 0 or self.money <= -15.0 or self.goods <= 0:
            self.alive = False

    def apply_learning(self, reward: float, pressure: float) -> None:
        stress_scale = max(0.15, 1.0 - pressure)
        lr = self.learning_rate * stress_scale
        self.preference["resource"] += lr * reward
        self.preference["money"] += lr * (0.4 * reward - 0.15 * pressure)
        self.preference["goods"] += lr * (0.35 * reward)
        self.preference["quality"] += lr * (0.3 * reward + 0.1 * self.knowledge)
        self.preference["risk"] += lr * (0.5 * reward - pressure * 0.3)
        self.preference["social"] += lr * (0.2 * reward)
        self.preference["horizon"] += lr * (reward - pressure * 0.4)
        self.trust["bank"] += lr * (0.05 - pressure * 0.08)
        self.trust["market"] += lr * (0.03 + reward * 0.06)
        self.knowledge += lr * (0.1 + 0.3 * self.trust["news"])

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
        self.money += harvested * (0.06 + 0.12 * self.trust["market"])
        self.goods += harvested * (0.03 + 0.06 * self.preference["goods"])
        self.goods_quality = clamp(
            0.95 * self.goods_quality + 0.05 * (0.5 * self.knowledge + 0.5 * self.trust["news"]),
            0.01,
            3.0,
        )
        self.resource -= self.metabolism + move_cost
        self.goods = max(0.0, self.goods - 0.25 * self.metabolism)
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

        self.apply_learning(reward + 0.08 * self.objective_value(self.objective_lambda), pressure)
        self.survival_check()


@dataclass
class Loan:
    borrower_id: int
    principal: float
    interest_rate: float
    term_remaining: int


@dataclass
class Bank:
    reserves: float = 500.0
    loans: List[Loan] = field(default_factory=list)

    def maybe_issue_loan(self, agent: Agent) -> None:
        credit_score = 0.5 * agent.trust["bank"] + 0.5 * agent.knowledge - 0.03 * max(0.0, agent.debt)
        if credit_score > 0.45 and agent.money < 5.0 and self.reserves > 10.0:
            principal = min(12.0, self.reserves * 0.05)
            self.reserves -= principal
            agent.money += principal
            agent.debt += principal
            self.loans.append(Loan(agent.agent_id, principal, interest_rate=0.025, term_remaining=12))

    def process_loans(self, agents_by_id: Dict[int, Agent]) -> None:
        survivors: List[Loan] = []
        for loan in self.loans:
            borrower = agents_by_id.get(loan.borrower_id)
            if borrower is None or not borrower.alive:
                continue
            due = (loan.principal * (1.0 + loan.interest_rate)) / max(1, loan.term_remaining)
            paid = min(due, max(0.0, borrower.money))
            borrower.money -= paid
            borrower.debt = max(0.0, borrower.debt - paid)
            self.reserves += paid
            loan.term_remaining -= 1
            if loan.term_remaining > 0 and borrower.debt > 0:
                survivors.append(loan)
        self.loans = survivors


@dataclass
class Market:
    base_price: float = 1.0
    quality_premium: float = 0.35

    def trade(self, agent: Agent) -> None:
        if not agent.alive:
            return
        price = self.base_price + self.quality_premium * agent.goods_quality
        trade_qty = min(agent.goods, 0.5 + 0.5 * agent.trust["market"])
        revenue = trade_qty * price
        agent.goods -= trade_qty
        agent.money += revenue
        buy_back = min(agent.money / max(0.3, price), 0.35 * (1.0 - agent.trust["market"]))
        agent.money -= buy_back * price
        agent.goods += buy_back


@dataclass
class LeaderCandidate:
    name: str
    objective_style: str
    campaign_cycle: int = 40
    platform_weight: float = 1.0

    def objective(self, values: Sequence[float], lambda_switch: float, eps: float = 1e-6) -> float:
        if self.objective_style == "sum":
            return sum(values)
        if self.objective_style == "sum_log":
            return sum(safe_log(v + eps, eps) for v in values)
        vital_min = min(values)
        needs = math.log(vital_min) if vital_min > 0 else -1e12
        sum_logs = sum(safe_log(v + eps, eps) for v in values)
        return lambda_switch * needs + (1.0 - lambda_switch) * sum_logs


@dataclass
class Firm:
    name: str
    objective_style: str
    cash: float = 100.0
    investor_ids: List[int] = field(default_factory=list)
    is_news_outlet: bool = False

    def objective(self, shareholders: float, employees: float, environment: float, company: float, eps: float = 1e-6) -> float:
        if self.objective_style == "sum":
            return shareholders + employees + environment + company
        vital = min(shareholders, employees, environment, company)
        return safe_log(vital, eps)

    def publish_signal(self) -> float:
        return clamp(0.35 + 0.4 * random.random() + 0.15 * (1 if self.is_news_outlet else 0), 0.0, 1.0)


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
    election_interval: int = 20
    campaign_interval: int = 40
    seed: int = 42


class Simulation:
    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        random.seed(cfg.seed)
        self.env = Environment(cfg.width, cfg.height, cfg.max_capacity, cfg.regen_rate)
        self.agents: List[Agent] = []
        self.bank = Bank()
        self.market = Market()
        self.candidates = [
            LeaderCandidate("Candidate Sum", "sum"),
            LeaderCandidate("Candidate SumLog", "sum_log"),
            LeaderCandidate("Candidate NeedsSwitch", "logmin_switch"),
        ]
        self.current_leader: LeaderCandidate = self.candidates[0]
        self.firms: List[Firm] = [
            Firm("Goods Cooperative", "sum"),
            Firm("Stakeholder Trust", "logmin"),
            Firm("Civic News", "sum", is_news_outlet=True),
        ]
        self.time = 0
        self.history: Dict[str, List[float]] = {
            "floor": [], "avg": [], "gini": [], "lambda": [], "alive": [], "growth_ratio": [], "survival_ratio": [],
            "avg_money": [], "avg_goods": [], "avg_trust_news": [], "avg_trust_leader": []
        }
        self._spawn_agents()

    def snapshot(self) -> Dict[str, object]:
        return {
            "time": self.time,
            "sugar": [[v for v in row] for row in self.env.sugar],
            "agents": [
                {"x": a.x, "y": a.y, "resource": a.resource, "alive": a.alive}
                for a in self.agents
                if a.alive
            ],
        }

    def _spawn_agents(self) -> None:
        for i in range(self.cfg.n_agents):
            self.agents.append(
                Agent(
                    agent_id=i,
                    x=random.randrange(self.cfg.width),
                    y=random.randrange(self.cfg.height),
                    resource=random.uniform(*self.cfg.initial_resource),
                    money=random.uniform(4.0, 18.0),
                    goods=random.uniform(2.0, 9.0),
                    goods_quality=random.uniform(0.3, 1.2),
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

    def _word_of_mouth_update(self, alive: List[Agent]) -> None:
        by_coord = {a.coord: a for a in alive}
        for agent in alive:
            neighbors = []
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                c = ((agent.x + dx) % self.cfg.width, (agent.y + dy) % self.cfg.height)
                other = by_coord.get(c)
                if other is not None:
                    neighbors.append(other)
            if not neighbors:
                continue
            social_pull = mean([n.trust["leader"] for n in neighbors])
            market_pull = mean([n.trust["market"] for n in neighbors])
            agent.trust["leader"] = clamp(
                0.9 * agent.trust["leader"] + 0.1 * (agent.trust["neighbors"] * social_pull), 0.0, 1.0
            )
            agent.trust["market"] = clamp(
                0.9 * agent.trust["market"] + 0.1 * (agent.trust["neighbors"] * market_pull), 0.0, 1.0
            )

    def _run_news_cycle(self, alive: List[Agent]) -> None:
        news_outlets = [f for f in self.firms if f.is_news_outlet]
        if not news_outlets or not alive:
            return
        signal = mean([n.publish_signal() for n in news_outlets])
        for agent in alive:
            trust_news = agent.trust["news"]
            trust_leader = agent.trust["leader"]
            agent.knowledge = clamp(agent.knowledge + 0.04 * trust_news * signal, 0.0, 1.0)
            agent.trust["leader"] = clamp(trust_leader + 0.03 * trust_news * (signal - 0.5), 0.0, 1.0)

    def _campaign_cycle(self, alive: List[Agent], lambda_switch: float) -> None:
        if self.time % self.cfg.campaign_interval != 0:
            return
        for c in self.candidates:
            scores = []
            for a in alive:
                basket = [a.resource, a.money, a.goods, a.goods_quality]
                scores.append(c.objective(basket, lambda_switch))
            c.platform_weight = 0.6 * c.platform_weight + 0.4 * (mean(scores) if scores else 0.0)

    def _election_cycle(self, alive: List[Agent], lambda_switch: float) -> None:
        if self.time % self.cfg.election_interval != 0 or not alive:
            return
        votes = {c.name: 0.0 for c in self.candidates}
        for a in alive:
            utilities = []
            basket = [a.resource, a.money, a.goods, a.goods_quality]
            for c in self.candidates:
                score = c.objective(basket, lambda_switch) + a.campaign_bias + 0.2 * c.platform_weight
                score *= 0.5 + 0.5 * a.trust["leader"]
                utilities.append((score, c.name))
            winner = max(utilities, key=lambda x: x[0])[1]
            votes[winner] += 1.0 + 0.1 * a.knowledge
        lead_name = max(votes.items(), key=lambda kv: kv[1])[0]
        self.current_leader = next(c for c in self.candidates if c.name == lead_name)
        for a in alive:
            a.trust["leader"] = clamp(a.trust["leader"] + 0.02, 0.0, 1.0)

    def _firm_cycle(self, alive: List[Agent]) -> None:
        if not alive:
            return
        env_score = mean([self.env.sugar[y][x] / max(1e-6, self.cfg.max_capacity) for y in range(self.cfg.height) for x in range(self.cfg.width)])
        for firm in self.firms:
            if not firm.investor_ids:
                firm.investor_ids = random.sample([a.agent_id for a in alive], k=min(8, len(alive)))
            shareholders = mean([alive[i % len(alive)].money for i in range(len(firm.investor_ids))]) if firm.investor_ids else 1.0
            employees = mean([a.resource for a in alive])
            company = firm.cash
            firm_score = firm.objective(shareholders + 1e-6, employees + 1e-6, env_score + 1e-6, company + 1e-6)
            firm.cash += 0.25 * firm_score
            for aid in firm.investor_ids[: min(4, len(firm.investor_ids))]:
                for agent in alive:
                    if agent.agent_id == aid:
                        dividend = max(0.0, 0.015 * firm.cash)
                        agent.money += dividend
                        firm.cash -= dividend
                        break

    def step(self) -> None:
        self.time += 1
        alive = self._alive_agents()
        if not alive:
            return
        for a in alive:
            a.survival_check()
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
            self.bank.maybe_issue_loan(a)
            self.market.trade(a)
            a.step(self.env, chosen[a.agent_id], pressure)
            if a.last_mode == "growth":
                growth_count += 1
            else:
                survival_count += 1
        alive = self._alive_agents()
        self.bank.process_loans({a.agent_id: a for a in alive})
        self._firm_cycle(alive)
        self._run_news_cycle(alive)
        self._word_of_mouth_update(alive)
        self._campaign_cycle(alive, lam)
        self._election_cycle(alive, lam)

        self.env.regenerate()
        alive2 = self._alive_agents()
        resources = [a.resource for a in alive2]
        monies = [a.money for a in alive2]
        goods = [a.goods for a in alive2]
        trust_news = [a.trust["news"] for a in alive2]
        trust_leader = [a.trust["leader"] for a in alive2]

        self.history["floor"].append(min(resources) if resources else 0.0)
        self.history["avg"].append(mean(resources))
        self.history["gini"].append(gini(resources))
        self.history["lambda"].append(lam)
        self.history["alive"].append(float(len(alive2)))
        self.history["avg_money"].append(mean(monies))
        self.history["avg_goods"].append(mean(goods))
        self.history["avg_trust_news"].append(mean(trust_news))
        self.history["avg_trust_leader"].append(mean(trust_leader))
        total = max(1, growth_count + survival_count)
        self.history["growth_ratio"].append(growth_count / total)
        self.history["survival_ratio"].append(survival_count / total)

    def run(self) -> Dict[str, List[float]]:
        for _ in range(self.cfg.steps):
            if not self._alive_agents():
                break
            self.step()
        return self.history

    def run_with_snapshots(self, every: int = 1) -> Tuple[Dict[str, List[float]], List[Dict[str, object]]]:
        frames: List[Dict[str, object]] = [self.snapshot()]
        for _ in range(self.cfg.steps):
            if not self._alive_agents():
                break
            self.step()
            if self.time % max(1, every) == 0:
                frames.append(self.snapshot())
        return self.history, frames


def write_csv(histories: Dict[str, Dict[str, List[float]]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    header = "experiment,timestep,floor,avg,gini,lambda,alive,growth_ratio,survival_ratio,avg_money,avg_goods,avg_trust_news,avg_trust_leader\n"
    lines = [header]
    for name, h in histories.items():
        n = len(h["floor"])
        for t in range(n):
            lines.append(
                f"{name},{t},{h['floor'][t]:.6f},{h['avg'][t]:.6f},{h['gini'][t]:.6f},{h['lambda'][t]:.6f},"
                f"{h['alive'][t]:.0f},{h['growth_ratio'][t]:.6f},{h['survival_ratio'][t]:.6f},{h['avg_money'][t]:.6f},"
                f"{h['avg_goods'][t]:.6f},{h['avg_trust_news'][t]:.6f},{h['avg_trust_leader'][t]:.6f}\n"
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
        f"gini_end={history['gini'][i]:.2f}, growth_ratio_end={history['growth_ratio'][i]:.2f}, lambda_end={history['lambda'][i]:.2f}, "
        f"money_end={history['avg_money'][i]:.2f}, goods_end={history['avg_goods'][i]:.2f}"
    )


def animate_snapshots(
    snapshots: List[Dict[str, object]],
    output_path: Path,
    title: str = "Sugarscape simulation",
    fps: int = 10,
) -> bool:
    if not snapshots:
        return False
    has_matplotlib = importlib.util.find_spec("matplotlib") is not None
    if not has_matplotlib:
        return False
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    sugar0 = snapshots[0]["sugar"]
    height = len(sugar0)
    width = len(sugar0[0]) if sugar0 else 0
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    fig.suptitle(title)
    image = ax.imshow(sugar0, cmap="YlOrBr", vmin=0.0, vmax=max(max(row) for row in sugar0))
    scat = ax.scatter([], [], s=14, c="cyan", edgecolors="black", linewidths=0.2)
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", color="white", fontsize=10, bbox={"facecolor": "black", "alpha": 0.35})
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    def update(i: int):
        frame = snapshots[i]
        sugar = frame["sugar"]
        image.set_data(sugar)
        alive_agents = frame["agents"]
        points = [[a["x"], a["y"]] for a in alive_agents]
        scat.set_offsets(points if points else [])
        txt.set_text(f"t={frame['time']}  alive={len(alive_agents)}")
        return image, scat, txt

    anim = FuncAnimation(fig, update, frames=len(snapshots), interval=1000 / max(1, fps), blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ext = output_path.suffix.lower()
    if ext == ".gif":
        anim.save(output_path, writer="pillow", fps=fps)
    else:
        try:
            anim.save(output_path, writer="ffmpeg", fps=fps)
        except Exception:
            fallback = output_path.with_suffix(".gif")
            anim.save(fallback, writer="pillow", fps=fps)
    plt.close(fig)
    return True


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
