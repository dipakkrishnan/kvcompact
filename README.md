# kvcompact

When a session hits its context limit, does a compactor that edits the model's KV cache keep more of what the
rest of the session needs than one that writes a summary, at the same memory budget? See `docs/plan.md`.

## Yours (the fun parts)

| Stub | Spec |
|---|---|
| `SessionCache.keep` (`src/kvcompact/engine.py`) | `tests/test_engine.py` |
| `Engine.forward`: logical positions + your own 4D causal mask | `tests/test_engine.py` |
| `SnapKV.compact` (`src/kvcompact/methods.py`) | arXiv:2404.14469 |
| `KVAgent.compact`: the kernel compactor | `docs/plan.md` |

The six engine tests fail until `keep` and `forward` exist. They are step 0: nothing downstream means anything
until they pass.

## Done (the plumbing)

- `sessions.py`: Claude Code JSONL to Qwen chat messages, turn starts, cut at the limit
- `methods.py`: `Full`, `Recent`, `SummaryTail` (summary plus verbatim recent turns, re-prefilled)
- `eval.py`: continuation NLL, excess NLL by distance, paired bootstrap over sessions, budget check
- `replay.py`: `uv run kvcompact-replay ~/.claude/projects/<project>/<session>.jsonl --limit 32768 --budget 4096`

## Run

```bash
uv sync
uv run pytest            # tests marked `model` download Qwen3-0.6B (~1.5 GB) on first run
uv run pytest -m "not model"
```
