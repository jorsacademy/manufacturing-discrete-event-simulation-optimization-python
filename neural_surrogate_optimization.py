from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn

from manufacturing_des_optimization import LineDesign, ManufacturingLineDES, generate_scenario


@dataclass(frozen=True)
class SurrogateSearchConfig:
    buffer_min: int = 1
    buffer_max: int = 20
    technicians: tuple[int, ...] = (1, 2, 3)
    initial_designs: int = 36
    active_rounds: int = 4
    batch_designs: int = 10
    selection_replications: int = 3
    validation_replications: int = 24
    hidden_dim: int = 64
    epochs: int = 250
    learning_rate: float = 2e-3
    seed: int = 2027
    warmup_hours: float = 8.0
    measurement_hours: float = 40.0

    def validate(self) -> None:
        if self.buffer_min < 1 or self.buffer_max < self.buffer_min:
            raise ValueError("invalid buffer bounds")
        if not self.technicians or min(self.technicians) < 1:
            raise ValueError("technicians must be positive")
        if self.initial_designs < 2 or self.batch_designs < 1 or self.active_rounds < 0:
            raise ValueError("invalid search budget")
        if self.selection_replications < 1 or self.validation_replications < 2:
            raise ValueError("invalid replication counts")


class ProfitSurrogate(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def expanded_candidate_designs(config: SurrogateSearchConfig) -> list[LineDesign]:
    config.validate()
    return [
        LineDesign(b1, b2, tech)
        for b1 in range(config.buffer_min, config.buffer_max + 1)
        for b2 in range(config.buffer_min, config.buffer_max + 1)
        for tech in config.technicians
    ]


def design_features(designs: Iterable[LineDesign], config: SurrogateSearchConfig) -> np.ndarray:
    designs = list(designs)
    span = max(config.buffer_max - config.buffer_min, 1)
    max_tech = max(config.technicians)
    return np.asarray(
        [
            [
                (d.buffer_machining_assembly - config.buffer_min) / span,
                (d.buffer_assembly_test - config.buffer_min) / span,
                d.repair_technicians / max_tech,
            ]
            for d in designs
        ],
        dtype=np.float32,
    )


def evaluate_design_mean_profit(
    design: LineDesign,
    seeds: Iterable[int],
    *,
    warmup_hours: float,
    measurement_hours: float,
) -> float:
    profits = []
    horizon = warmup_hours + measurement_hours
    for seed in seeds:
        scenario = generate_scenario(int(seed), horizon_hours=horizon)
        result = ManufacturingLineDES(
            design,
            scenario,
            warmup_hours=warmup_hours,
            measurement_hours=measurement_hours,
        ).run()
        profits.append(result.profit_per_hour)
    return float(np.mean(profits))


def fit_surrogate(
    x: np.ndarray,
    y: np.ndarray,
    config: SurrogateSearchConfig,
) -> tuple[ProfitSurrogate, float, float]:
    torch.manual_seed(config.seed)
    model = ProfitSurrogate(config.hidden_dim)
    x_t = torch.as_tensor(x, dtype=torch.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    y_mean = float(np.mean(y_arr))
    y_std = float(np.std(y_arr) + 1e-6)
    y_t = torch.as_tensor((y_arr - y_mean) / y_std, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    for _ in range(config.epochs):
        pred = model(x_t)
        loss = torch.mean((pred - y_t) ** 2)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    model.eval()
    return model, y_mean, y_std


@torch.no_grad()
def predict_surrogate(model: ProfitSurrogate, x: np.ndarray, y_mean: float, y_std: float) -> np.ndarray:
    pred = model(torch.as_tensor(x, dtype=torch.float32)).cpu().numpy()
    return pred * y_std + y_mean


@dataclass(frozen=True)
class SearchResult:
    selected_design: LineDesign
    selected_validation_profit: float
    random_baseline_design: LineDesign
    random_baseline_validation_profit: float
    evaluated_designs: int
    simulation_calls_per_method: int


def _validate_design(design: LineDesign, config: SurrogateSearchConfig, seed_base: int) -> float:
    seeds = range(seed_base, seed_base + config.validation_replications)
    return evaluate_design_mean_profit(
        design,
        seeds,
        warmup_hours=config.warmup_hours,
        measurement_hours=config.measurement_hours,
    )


def surrogate_assisted_search(config: SurrogateSearchConfig = SurrogateSearchConfig()) -> SearchResult:
    """Active neural-surrogate search with a simulation-budget-matched random baseline."""
    config.validate()
    rng = np.random.default_rng(config.seed)
    candidates = expanded_candidate_designs(config)
    n_budget = min(
        len(candidates),
        config.initial_designs + config.active_rounds * config.batch_designs,
    )
    if n_budget < 2:
        raise ValueError("candidate set too small")

    selection_seeds = list(range(config.seed + 10_000, config.seed + 10_000 + config.selection_replications))

    initial_idx = rng.choice(len(candidates), size=min(config.initial_designs, len(candidates)), replace=False)
    evaluated: dict[int, float] = {}
    for idx in initial_idx:
        evaluated[int(idx)] = evaluate_design_mean_profit(
            candidates[int(idx)],
            selection_seeds,
            warmup_hours=config.warmup_hours,
            measurement_hours=config.measurement_hours,
        )

    for _ in range(config.active_rounds):
        if len(evaluated) >= n_budget:
            break
        idxs = np.asarray(sorted(evaluated), dtype=int)
        x = design_features([candidates[i] for i in idxs], config)
        y = np.asarray([evaluated[int(i)] for i in idxs], dtype=np.float32)
        model, y_mean, y_std = fit_surrogate(x, y, config)

        remaining = np.asarray([i for i in range(len(candidates)) if i not in evaluated], dtype=int)
        if len(remaining) == 0:
            break
        pred = predict_surrogate(model, design_features([candidates[i] for i in remaining], config), y_mean, y_std)
        take = min(config.batch_designs, n_budget - len(evaluated), len(remaining))
        chosen = remaining[np.argsort(pred)[-take:]]
        for idx in chosen:
            evaluated[int(idx)] = evaluate_design_mean_profit(
                candidates[int(idx)],
                selection_seeds,
                warmup_hours=config.warmup_hours,
                measurement_hours=config.measurement_hours,
            )

    best_idx = max(evaluated, key=evaluated.get)
    selected = candidates[best_idx]

    random_idx = rng.choice(len(candidates), size=len(evaluated), replace=False)
    random_scores = {
        int(idx): evaluate_design_mean_profit(
            candidates[int(idx)],
            selection_seeds,
            warmup_hours=config.warmup_hours,
            measurement_hours=config.measurement_hours,
        )
        for idx in random_idx
    }
    random_best = candidates[max(random_scores, key=random_scores.get)]

    validation_seed = config.seed + 200_000
    selected_validation = _validate_design(selected, config, validation_seed)
    random_validation = _validate_design(random_best, config, validation_seed)

    calls = len(evaluated) * config.selection_replications + config.validation_replications
    return SearchResult(
        selected_design=selected,
        selected_validation_profit=selected_validation,
        random_baseline_design=random_best,
        random_baseline_validation_profit=random_validation,
        evaluated_designs=len(evaluated),
        simulation_calls_per_method=calls,
    )


def main() -> None:
    result = surrogate_assisted_search()
    print("Neural surrogate-assisted DES optimization")
    print("selected:", result.selected_design, result.selected_validation_profit)
    print("random:  ", result.random_baseline_design, result.random_baseline_validation_profit)
    print("evaluated designs:", result.evaluated_designs)
    print("simulation calls per method:", result.simulation_calls_per_method)


if __name__ == "__main__":
    main()
