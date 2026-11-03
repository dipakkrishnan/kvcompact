import json

import pytest
import torch

from kvcompact.sessions import Rendered, Transcript


def write(tmp_path, records):
    path = tmp_path / "s.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    return path


def turn(role, content, **extra):
    return {"type": role, "message": {"role": role, "content": content}, **extra}


def test_claude_code_transcript_becomes_chat_messages(tmp_path):
    path = write(tmp_path, [
        {"type": "queue-operation"},
        turn("user", "fix the relay"),
        turn("assistant", [{"type": "thinking", "thinking": ""}, {"type": "text", "text": "Looking."}]),
        turn("assistant", [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}]),
        turn("user", [{"type": "tool_result", "tool_use_id": "t1", "content": "README.md"}]),
        turn("user", "side quest", isSidechain=True),
        turn("user", [{"type": "image", "source": {}}]),
    ])
    messages = Transcript.from_claude_code(path).messages
    assert [m["role"] for m in messages] == ["user", "assistant", "tool"]
    assert messages[1]["content"] == "Looking."
    assert messages[1]["tool_calls"][0]["function"] == {"name": "Bash", "arguments": {"command": "ls"}}
    assert messages[2]["content"] == "README.md"


def test_cut_lands_on_a_turn_start():
    rendered = Rendered(torch.arange(100)[None], [0, 30, 55, 80])
    prefix, continuation = rendered.cut(limit=60, continuation=20)
    assert prefix.ids.shape[1] == 55
    assert prefix.turn_starts == [0, 30]
    assert continuation[0, 0].item() == 55


def test_cut_rejects_short_sessions():
    with pytest.raises(ValueError):
        Rendered(torch.arange(100)[None], [0, 30, 55, 80]).cut(limit=60, continuation=50)


@pytest.mark.model
def test_render_finds_turn_starts(tmp_path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    rendered = Transcript([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]).render(tokenizer)
    assert rendered.turn_starts[0] == 0 and len(rendered.turn_starts) == 2
