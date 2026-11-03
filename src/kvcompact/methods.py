"""Compaction methods. Each turns a full session cache into one that fits a budget of `budget` slots."""

from typing import Protocol

import torch

from kvcompact.engine import Engine, SessionCache
from kvcompact.sessions import Rendered


class Method(Protocol):
    name: str

    def compact(self, engine: Engine, session: SessionCache, prefix: Rendered, budget: int) -> SessionCache: ...


class Full:
    name = "full"

    def compact(self, engine, session, prefix, budget):
        return session


class Recent:
    name = "recent"

    def compact(self, engine, session, prefix, budget):
        return session.keep(torch.arange(session.size - budget, session.size))


class SummaryTail:
    """The compactor writes a summary; the new context is the summary turn followed by the most recent whole turns."""

    name = "summary+tail"
    prompt = (
        "Your context is about to be cleared. Write a summary of the session so far that lets you continue the work: "
        "the goal, decisions made, current state of files and commands, open problems, and next steps."
    )
    header = "Summary of the session so far:\n"
    overhead = 16

    def __init__(self, tail: int):
        self.tail = tail

    def compact(self, engine, session, prefix, budget):
        n = prefix.ids.shape[1]
        tail_start = min((s for s in prefix.turn_starts if n - s <= self.tail), default=n)
        tail = prefix.ids[:, tail_start:]
        room = budget - tail.shape[1] - self.overhead
        summary = engine.tokenizer.decode(engine.generate(session, self._turn(engine, self.prompt, True), max_new=room))
        head = self._turn(engine, self.header + summary, False)
        return engine.prefill(torch.cat([head, tail], dim=1))

    @staticmethod
    def _turn(engine, text: str, ask: bool) -> torch.Tensor:
        return engine.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=ask, enable_thinking=False,
            return_tensors="pt", return_dict=True
        )["input_ids"]


class SnapKV:
    """YOURS. SnapKV (arXiv:2404.14469): score slots by attention from an observation window at the end of the
    prefix, pool the scores, and keep the top `budget` slots with their original positions."""

    name = "snapkv"

    def compact(self, engine, session, prefix, budget):
        raise NotImplementedError


class KVAgent:
    """YOURS. A frontier compactor in a Python kernel over the session cache: inspect attention, probe with
    development questions, and submit a selection of at most `budget` slots."""

    name = "kv-agent"

    def compact(self, engine, session, prefix, budget):
        raise NotImplementedError
