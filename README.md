# kvcompact

An open-weight model runs a coding session in Pi. When its context fills up, does the agent do better if a
frontier advisor edits its KV cache instead of writing a summary? See `docs/plan.md`.

## Yours

| Stub | Where | Spec |
|---|---|---|
| `SessionCache.keep`, `Engine.forward` | `src/kvcompact/engine.py` | `tests/test_engine.py` (step 0) |
| `expand`, `render`, `sync`, `parse` | `src/kvcompact/server.py` | docstrings |
| `KernelAdvisor.compact` | `src/kvcompact/advisor.py` | `docs/plan.md` |

## Done

- `server.py`: OpenAI-compatible `/v1/chat/completions` (streaming), `/compact`, `/health`, request log
- `.pi/extensions/kvcompact/`: registers the `kvcompact` provider and swaps compaction per `KVCOMPACT_MODE`
- `.pi/settings.json`: compaction fires at the limit, short verbatim tail
- `RecentAdvisor`: keep the last B slots, for testing the loop before the kernel advisor exists

## Run

```bash
uv sync && npm install
uv run pytest                                    # engine tests; download Qwen3-0.6B on first run

# server (small model for plumbing; the 30B on RunPod for real runs)
uv run kvcompact-serve --model Qwen/Qwen3-0.6B --dtype float32 --advisor recent --budget 1024

# agent: KVCOMPACT_MODE = kv | summary | none; KVCOMPACT_LIMIT sets where compaction fires
KVCOMPACT_MODE=kv KVCOMPACT_LIMIT=2048 npx pi --model kvcompact/qwen3-coder
npm run typecheck                                # the extension
```

The `summary` arm needs Anthropic credentials in Pi (`/login`, or `ANTHROPIC_API_KEY`). Pi prints
`[kvcompact] ...` to stderr on every compaction; a `FAILED` line means that run is invalid.
