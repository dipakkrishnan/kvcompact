import torch

from kvcompact.eval import Score, excess_by_distance, paired_bootstrap


def test_paired_bootstrap_brackets_the_mean():
    a = {f"s{i}": 1.0 + 0.1 * i for i in range(10)}
    b = {f"s{i}": 1.0 for i in range(10)}
    mean, low, high = paired_bootstrap(a, b, samples=2_000)
    assert abs(mean - 0.45) < 1e-9 and low < mean < high and low > 0


def test_excess_by_distance_uses_the_first_tokens():
    reference = Score("s", "full", 10, torch.zeros(1, 4))
    score = Score("s", "recent", 2, torch.tensor([[1.0, 1.0, 0.0, 0.0]]))
    assert excess_by_distance(score, reference, [2, 4, 8]) == {2: 1.0, 4: 0.5}
