import random
from dataclasses import dataclass

import torch

from kvcompact.engine import Engine, SessionCache


@dataclass(frozen=True)
class Score:
    session: str
    method: str
    slots: int
    nll: torch.Tensor

    @property
    def mean_nll(self) -> float:
        return float(self.nll.mean())


def excess_by_distance(score: Score, reference: Score, buckets: list[int]) -> dict[int, float]:
    """Mean excess NLL over `reference` for continuation tokens up to each bucket's distance from the cut."""
    excess = (score.nll - reference.nll).flatten()
    return {b: float(excess[:b].mean()) for b in buckets if b <= excess.numel()}


def paired_bootstrap(a: dict[str, float], b: dict[str, float], samples: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of a - b over shared sessions with a 95% interval, resampling sessions."""
    sessions = sorted(a.keys() & b.keys())
    diffs = [a[s] - b[s] for s in sessions]
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(samples))
    return sum(diffs) / len(diffs), means[int(0.025 * samples)], means[int(0.975 * samples)]


def score(engine: Engine, name: str, method: str, compacted: SessionCache, continuation: torch.Tensor, budget: int) -> Score:
    if method != "full" and compacted.size > budget:
        raise ValueError(f"{method} used {compacted.size} slots, over the budget of {budget}")
    return Score(name, method, compacted.size, engine.nll(compacted, continuation))
