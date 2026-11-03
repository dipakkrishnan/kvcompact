"""Load agent transcripts and render them into the subject model's chat template."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class Rendered:
    ids: torch.Tensor
    turn_starts: list[int]

    def cut(self, limit: int, continuation: int) -> tuple["Rendered", torch.Tensor]:
        """Split at the last turn start at or before `limit`: (prefix, next `continuation` ids)."""
        boundary = max((s for s in self.turn_starts if 0 < s <= limit), default=None)
        if boundary is None or self.ids.shape[1] < boundary + continuation:
            raise ValueError(f"session too short to cut at {limit} with {continuation} continuation tokens")
        prefix = Rendered(self.ids[:, :boundary], [s for s in self.turn_starts if s < boundary])
        return prefix, self.ids[:, boundary : boundary + continuation]


@dataclass(frozen=True)
class Transcript:
    messages: list[dict]

    @classmethod
    def from_claude_code(cls, path: Path) -> "Transcript":
        """Main-thread user/assistant turns; tool calls become Qwen-style tool calls, results become tool turns.

        Images and redacted thinking are dropped. Consecutive records from the same role are merged.
        """
        messages: list[dict] = []
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record.get("type") not in ("user", "assistant") or record.get("isSidechain"):
                continue
            content = record["message"]["content"]
            blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
            for block in blocks:
                cls._append(messages, record["type"], block)
        return cls(messages)

    @staticmethod
    def _append(messages: list[dict], role: str, block: dict) -> None:
        kind = block.get("type")
        if kind == "text" and block["text"].strip():
            message = {"role": role, "content": block["text"]}
        elif kind == "tool_use":
            message = {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": block["name"], "arguments": block["input"]}}
            ]}
        elif kind == "tool_result":
            message = {"role": "tool", "content": Transcript._result_text(block.get("content"))}
        else:
            return
        last = messages[-1] if messages else None
        if last and last["role"] == message["role"] == "assistant":
            last["content"] = "\n".join(c for c in (last["content"], message["content"]) if c)
            last.setdefault("tool_calls", []).extend(message.get("tool_calls", []))
        elif last and last["role"] == message["role"] and role == "user" and kind == "text":
            last["content"] += "\n" + message["content"]
        else:
            messages.append(message)

    @staticmethod
    def _result_text(content) -> str:
        if isinstance(content, str):
            return content
        return "\n".join(b.get("text", "") for b in content or [] if b.get("type") == "text")

    def render(self, tokenizer) -> Rendered:
        ids = tokenizer.apply_chat_template(self.messages, tokenize=True, return_tensors="pt", return_dict=True)["input_ids"]
        start = tokenizer.convert_tokens_to_ids("<|im_start|>")
        return Rendered(ids, (ids[0] == start).nonzero().flatten().tolist())
