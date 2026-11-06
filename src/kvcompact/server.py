import asyncio
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from kvcompact.advisor import Advisor, KernelAdvisor, RecentAdvisor
from kvcompact.engine import Engine, SessionCache

SENTINEL = re.compile(r"\[kv-compacted:([0-9a-f]+)\]")


@dataclass
class Live:
    """The one agent session this server holds.

    `ids` is the LOGICAL token history: every token the session has ever contained, in order, including tokens
    the advisor has since dropped from the cache. Because the engine keeps logical positions, `ids` stays the
    reference that new requests are diffed against, before and after compaction.
    """

    session: SessionCache | None = None
    ids: list[int] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    compactions: dict[str, list[dict]] = field(default_factory=dict)


def expand(messages: list[dict], live: Live) -> list[dict]:
    """YOURS. Undo Pi's view of a KV compaction so the request describes the full logical history again.

    After a kv compaction Pi sends: [system, user("...<summary>[kv-compacted:ID]</summary>"), kept..., new...]
    where `kept` are the messages Pi kept verbatim (a suffix of `live.compactions[ID]`) and `new` arrived since.
    Return `live.compactions[ID] + new`. Find `new` by matching the longest prefix of the messages after the
    sentinel against a suffix of the stored list. Requests without a sentinel come back unchanged. Fail loudly
    (raise) on an unknown ID: silently re-prefilling would turn a kv run into a no-compaction run.
    """
    raise NotImplementedError


def render(engine: Engine, messages: list[dict], tools: list[dict] | None) -> list[int]:
    """YOURS. OpenAI chat messages + tools -> Qwen token ids, ending with the assistant generation prompt.

    Use `engine.tokenizer.apply_chat_template(..., tools=tools, add_generation_prompt=True)`. Gotchas:
    OpenAI sends `tool_calls[].function.arguments` as a JSON string but Qwen3-Coder's template iterates it as a
    mapping (json.loads it); `content` can be a list of {"type": "text"} parts; Pi may send no system message.
    """
    raise NotImplementedError


def sync(engine: Engine, live: Live, ids: list[int]) -> tuple[SessionCache, list[int]]:
    """YOURS. Return (cache to generate from, ids still to feed) so cache + those ids represent `ids` exactly.

    Normally `live.ids` is a prefix of `ids` and you return (live.session, ids[len(live.ids):]). Two exceptions:
    - First request: prefill everything but the last token, and return that last token as the remainder.
    - Divergence at position p < len(live.ids): usually the template re-rendering our own last assistant turn
      differently from what we generated (tool call formatting, whitespace). Roll back by keeping only slots
      whose logical position is < p (`session.keep`), then feed ids[p:]. If p falls before the latest
      compaction you cannot roll back (those tokens are gone): raise.
    Always leave at least one id to feed, since `generate` needs logits for the next token.
    """
    raise NotImplementedError


def parse(engine: Engine, generated: list[int]) -> dict:
    """YOURS. Generated ids -> an OpenAI assistant message: {"role", "content", "tool_calls"?}.

    Qwen3-Coder emits <tool_call><function=NAME><parameter=KEY>VALUE</parameter>...</function></tool_call>
    (check the chat template for the exact format). Each call becomes
    {"id": "call_<hex>", "type": "function", "function": {"name": NAME, "arguments": json.dumps(params)}}.
    Text outside tool calls is `content`. Parameter values are strings: json.loads them when they parse.
    """
    raise NotImplementedError


def compact(engine: Engine, advisor: Advisor, live: Live, budget: int) -> str:
    """Compact the live cache with the advisor and remember the history it stands for under a new id."""
    if live.session is None:
        raise HTTPException(409, "nothing to compact yet")
    compacted = advisor.compact(engine, live.session, live.ids, live.messages, budget)
    if compacted.size > budget:
        raise HTTPException(500, f"advisor returned {compacted.size} slots, over the budget of {budget}")
    cid = uuid.uuid4().hex[:12]
    live.compactions[cid] = list(live.messages)
    live.session = compacted
    return cid


def create_app(engine: Engine, advisor: Advisor, budget: int, max_new: int = 4096, log: Path | None = None) -> FastAPI:
    app = FastAPI()
    live = Live()
    lock = asyncio.Lock()

    def turn(body: dict) -> tuple[dict, int, int]:
        messages = expand(body["messages"], live)
        ids = render(engine, messages, body.get("tools"))
        session, rest = sync(engine, live, ids)
        generated, session = engine.generate(session, torch.tensor([rest]), min(body.get("max_tokens") or max_new, max_new))
        message = parse(engine, generated)
        live.session, live.ids = session, ids + generated
        live.messages = messages + [message]
        return message, session.size - len(generated), len(generated)

    @app.get("/health")
    async def health():
        return {"ok": True, "slots": live.session.size if live.session else 0, "logical": len(live.ids)}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": engine.name, "object": "model", "owned_by": "kvcompact"}]}

    @app.post("/compact")
    async def compact_route(request: Request):
        body = await request.json() if await request.body() else {}
        async with lock:
            cid = await asyncio.to_thread(compact, engine, advisor, live, body.get("budget", budget))
        return {"id": cid, "slots": live.session.size, "logical": len(live.ids)}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        if log:
            with log.open("a") as f:
                f.write(json.dumps(body) + "\n")
        async with lock:
            message, prompt_tokens, completion_tokens = await asyncio.to_thread(turn, body)
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                 "total_tokens": prompt_tokens + completion_tokens}
        finish = "tool_calls" if message.get("tool_calls") else "stop"
        base = {"id": f"chatcmpl-{uuid.uuid4().hex}", "created": int(time.time()), "model": body.get("model", engine.name)}
        if not body.get("stream"):
            return JSONResponse({**base, "object": "chat.completion", "usage": usage,
                                 "choices": [{"index": 0, "message": message, "finish_reason": finish}]})
        return StreamingResponse(stream(base, message, finish, usage), media_type="text/event-stream")

    return app


def stream(base: dict, message: dict, finish: str, usage: dict):
    """Replay a finished message as OpenAI SSE chunks. Generation is not streamed token by token; Pi doesn't care."""
    def chunk(delta: dict, finish_reason=None, **extra) -> str:
        payload = {**base, "object": "chat.completion.chunk", **extra,
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
        return f"data: {json.dumps(payload)}\n\n"

    yield chunk({"role": "assistant", "content": ""})
    if message.get("content"):
        yield chunk({"content": message["content"]})
    for i, call in enumerate(message.get("tool_calls", [])):
        yield chunk({"tool_calls": [{"index": i, **call}]})
    yield chunk({}, finish)
    yield f"data: {json.dumps({**base, 'object': 'chat.completion.chunk', 'choices': [], 'usage': usage})}\n\n"
    yield "data: [DONE]\n\n"


def main() -> None:
    import argparse

    import uvicorn

    args = argparse.ArgumentParser()
    args.add_argument("--model", default=os.environ.get("KVCOMPACT_MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct"))
    args.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args.add_argument("--attn", default="sdpa", help="eager materializes 32k x 32k scores per head; too big for the 30B")
    args.add_argument("--advisor", default="kernel", choices=["kernel", "recent"])
    args.add_argument("--budget", type=int, default=4096)
    args.add_argument("--port", type=int, default=8000)
    args.add_argument("--log", type=Path, default=Path("logs/requests.jsonl"))
    opts = args.parse_args()

    engine = Engine(opts.model, opts.device, getattr(torch, opts.dtype), opts.attn)
    advisor = {"kernel": KernelAdvisor, "recent": RecentAdvisor}[opts.advisor]()
    opts.log.parent.mkdir(parents=True, exist_ok=True)
    uvicorn.run(create_app(engine, advisor, opts.budget, log=opts.log), host="127.0.0.1", port=opts.port)


if __name__ == "__main__":
    main()
