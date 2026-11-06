from typing import Protocol

from kvcompact.engine import Engine, SessionCache


class Advisor(Protocol):
    """A frontier model that compacts the subject's cache. The server calls it when Pi decides to compact."""

    def compact(self, engine: Engine, session: SessionCache, ids: list[int], messages: list[dict], budget: int) -> SessionCache:
        """Return a SessionCache of at most `budget` slots derived from `session`.

        `ids` is the logical token history: slot i of `session` holds `ids[session.positions[i]]`, so
        `engine.tokenizer.decode` on a slice of it shows the advisor what each slot is. `messages` is the same
        history as chat messages, for turn structure. Must not modify `session`.
        """
        ...


class KernelAdvisor:
    """YOURS. Claude with a Python kernel tool over `engine`, `session`, `ids` and `messages`; it submits slots to
    keep, optionally after probing the candidate cache with `engine.generate`."""

    def compact(self, engine, session, ids, messages, budget):
        raise NotImplementedError


class RecentAdvisor:
    """Keeps the most recent `budget` slots. A stand-in so the server and Pi loop can be tested end to end before
    the kernel advisor exists; also the cheapest baseline."""

    def compact(self, engine, session, ids, messages, budget):
        import torch

        return session.keep(torch.arange(max(0, session.size - budget), session.size))
