"""Step 0: cache surgery must be exact before any experiment result means anything."""

import pytest
import torch

from kvcompact.engine import Engine

pytestmark = pytest.mark.model
TEXT = (
    "The build failed because the config loader read the wrong environment file. We pointed it at the staging "
    "file, reran the migration, and confirmed the service answered again. Next we need to rotate the API key and "
    "rerun the integration test before telling the team the deploy is back."
)
ATOL = 1e-3  # float32 summation-order noise on logits of ~10-20


@pytest.fixture(scope="module")
def engine() -> Engine:
    return Engine("Qwen/Qwen3-0.6B")


@pytest.fixture(scope="module")
def ids(engine) -> torch.Tensor:
    ids = engine.tokenizer(TEXT, return_tensors="pt")["input_ids"]
    assert ids.shape[1] > 40
    return ids


def split(ids, at=-16):
    return ids[:, :at], ids[:, at:]


def dropped(n: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    return order[n // 2 :].sort().values, order[: n // 2].sort().values


def test_forward_matches_a_single_pass(engine, ids):
    prefix, rest = split(ids)
    expected = engine.model(input_ids=ids).logits[:, prefix.shape[1] :]
    logits, _ = engine.forward(engine.prefill(prefix), rest)
    torch.testing.assert_close(logits, expected, atol=ATOL, rtol=0)


def test_noop_edit_is_exact(engine, ids):
    prefix, rest = split(ids)
    session = engine.prefill(prefix)
    untouched, _ = engine.forward(session, rest)
    edited, _ = engine.forward(session.keep(torch.arange(session.size)), rest)
    torch.testing.assert_close(edited, untouched, atol=0, rtol=0)


def test_forward_does_not_modify_the_session(engine, ids):
    prefix, rest = split(ids)
    session = engine.prefill(prefix)
    before = session.cache.layers[0].keys.clone()
    engine.forward(session, rest)
    assert session.size == prefix.shape[1]
    assert torch.equal(session.cache.layers[0].keys, before)


def test_pruned_cache_matches_masked_full_cache(engine, ids):
    prefix, rest = split(ids)
    session = engine.prefill(prefix)
    kept, hidden = dropped(session.size)
    pruned, _ = engine.forward(session.keep(kept), rest)
    masked, _ = engine.forward(session, rest, hidden=hidden)
    torch.testing.assert_close(pruned, masked, atol=ATOL, rtol=0)


def test_positions_stay_logical_after_pruning(engine, ids):
    prefix, rest = split(ids)
    session = engine.prefill(prefix)
    kept, _ = dropped(session.size)
    pruned = session.keep(kept)
    assert pruned.size == kept.numel()
    assert pruned.next_position == prefix.shape[1]
    assert torch.equal(pruned.positions, kept)
    _, grown = engine.forward(pruned, rest)
    assert grown.positions[-1].item() == ids.shape[1] - 1


def test_new_tokens_stay_causal_across_appends(engine, ids):
    prefix, rest = split(ids)
    session = engine.prefill(prefix)
    kept, hidden = dropped(session.size)
    _, step = engine.forward(session.keep(kept), rest[:, :8])
    second, _ = engine.forward(step, rest[:, 8:])
    masked, _ = engine.forward(session, rest, hidden=hidden)
    torch.testing.assert_close(second, masked[:, 8:], atol=ATOL, rtol=0)
