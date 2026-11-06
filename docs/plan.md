# Advisor KV compaction for a live coding agent: plan v3

## Question
An open-weight model runs a real coding session in Pi. When its context fills up, a frontier advisor (Claude)
compacts it. Does the agent do better when the advisor edits the model's KV cache than when it writes a
summary, at the same context budget?

## Arms (same tasks, same subject, same advisor)
1. none: no compaction. The native window is far past the limit, so the agent just keeps going (the ceiling).
2. summary: Pi's own compaction prompt and cut rules, with the summary written by the advisor.
3. kv: the advisor gets a Python kernel over the live cache and returns the slots to keep. Qwen continues on
   the edited cache.

Metric: task solved (hidden tests). Also log steps, tokens, compactions per run and re-read/redone work.
Same seeds per task across arms. A pilot is ~10 tasks long enough to compact at least once.

## Fixed choices
- Subject: Qwen3-Coder-30B-A3B-Instruct, bf16, HF transformers, 1x H200 (RunPod). Plain softmax attention in
  every layer, so the KV cache really is the model's whole memory of the past (see "Subject model" below).
- Limit L = 32k tokens; budget B = 4k slots after compaction (resident K/V, everything included).
- Pi pinned at 0.70.2 (`package.json`); the advisor is the same Claude model in both compacting arms.

## How it fits together

```
Pi (agent loop, tools, compaction trigger)
 ├─ provider "kvcompact" ──/v1/chat/completions──> server.py: expand -> render -> sync -> generate -> parse
 └─ extension: session_before_compact
      kv:      POST /compact -> advisor.compact(live cache) -> Pi stores summary "[kv-compacted:<id>]"
      summary: pi's compact() with the advisor model
      none:    cancel (window opened to 256k instead)
```

- Trigger: Pi compacts when `contextTokens > contextWindow - reserveTokens`. The extension sets the window to
  L + 8192 and `.pi/settings.json` sets `reserveTokens` 8192, so compaction fires at L. `keepRecentTokens`
  is 2048, so Pi keeps a short verbatim tail in every arm.
- After a kv compaction Pi sends `[system, user("...<summary>[kv-compacted:ID]</summary>"), kept..., new...]`
  (verified against a capture server). The server swaps the sentinel back for the history it stands for, so
  the request is always "the full logical history", and only new tokens get prefilled.
- The server keeps the logical token history (`Live.ids`) across compactions. Every request is diffed
  against it. Engine positions are logical, so the cache can be sparse while the history stays exact.
- `prompt_tokens` reports resident slots, so after a kv compaction Pi sees a small context and doesn't
  re-trigger.

## Who writes what
- Done: `server.py` HTTP + OpenAI wire format + `/compact`; Pi extension (provider, three arms); settings.
- Yours: engine `keep` / `forward` (step 0); server `expand`, `render`, `sync`, `parse`; `KernelAdvisor`.
- `RecentAdvisor` (keep the last B slots) exists so the whole loop runs before `KernelAdvisor` does.

## Order
1. Engine tests pass on Qwen3-0.6B (Mac is fine).
2. Server stubs, then `--advisor recent` with Qwen3-0.6B and `KVCOMPACT_LIMIT=2048`: run Pi through a tiny
   task, watch it compact and keep going. All plumbing bugs show up here, cheaply.
3. Same on RunPod with the 30B at L = 32k, `--advisor recent`, one real task.
4. `KernelAdvisor`, then the pilot: 3 arms x ~10 tasks.

## Known traps
- Throwing from `session_before_compact` makes Pi fall back to its default summary (verified). The
  extension returns `{ cancel: true }` and prints to stderr instead; check stderr before trusting a kv run.
- Don't use the extension `ctx` after an `await` in that handler: in print mode it's stale and throws.
- Pi's system prompt alone is ~15k characters (~4k tokens). With B = 4k the advisor must spend most of the
  budget keeping it. Decide whether B counts the system prompt; if it does, raise B to 8k.
- Chat-template round trip: re-rendering our own generated turn may not reproduce the generated tokens.
  `sync` rolls back to the divergence point instead of failing, but only after the latest compaction.
- Eager attention materializes L x L scores per head; the server defaults to sdpa for the 30B.
- `prefill` is one pass; chunk it if 32k OOMs.

## Subject model
Stronger open coders exist (DeepSeek-V4-Pro, Kimi K2.6, MiniMax M2.5 report ~80% SWE-bench Verified), but
they are far too big for one GPU. Qwen3-Coder-Next (80B total, 3B active, reported ~70.6%) is the
interesting one, but it is a hybrid: 3 of every 4 layers are Gated DeltaNet with a fixed recurrent state and no
KV cache. Pruning the attention layers' cache would leave the dropped tokens' information in the recurrent
state, so "kv compaction" would no longer remove anything cleanly. It is also ~160 GB in bf16. Stay on
Qwen3-Coder-30B-A3B for v1; revisit for a full-attention model that fits.

## Deferred
Writing K/V (merging, synthetic entries), SnapKV-style heuristic baselines, multiple concurrent sessions,
token-by-token streaming.
