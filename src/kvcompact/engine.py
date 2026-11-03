"""Model and KV cache with logical positions kept separate from physical cache slots."""

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


@dataclass(frozen=True)
class SessionCache:
    """A KV cache whose slots may be sparse: slot i holds the token at logical position `positions[i]`."""

    cache: DynamicCache
    positions: torch.Tensor
    next_position: int

    @property
    def size(self) -> int:
        return self.positions.numel()

    def keep(self, slots: torch.Tensor) -> "SessionCache":
        """YOURS. Return a new SessionCache holding only `slots` (physical indices) in every layer.

        Keys keep their original rotations, `positions` follows the kept slots, and `next_position` is unchanged.
        Must not modify `self`.
        """
        raise NotImplementedError

    def fork(self) -> "SessionCache":
        return self.keep(torch.arange(self.size))


class Engine:
    def __init__(self, model_id: str, device: str = "cpu", dtype: torch.dtype = torch.float32, attn: str = "eager"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, attn_implementation=attn).to(device)
        self.model.eval()
        self.device = device

    def prefill(self, ids: torch.Tensor) -> SessionCache:
        cache = DynamicCache(config=self.model.config)
        with torch.no_grad():
            self.model(input_ids=ids.to(self.device), past_key_values=cache, use_cache=True)
        n = ids.shape[1]
        return SessionCache(cache, torch.arange(n), n)

    def forward(
        self, session: SessionCache, ids: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, SessionCache]:
        """YOURS. Append `ids` after `session` and return (logits, grown session) without modifying `session`.

        New tokens take logical positions from `session.next_position`. `hidden` lists physical slots to mask out
        rather than remove: the masked full cache is the reference a pruned cache must match. Qwen3 returns an
        explicit 4D `attention_mask` unchanged (eager attention adds it to the scores), so the causal mask over
        [cache slots + new tokens] is yours to build.
        """
        raise NotImplementedError

    def nll(self, session: SessionCache, ids: torch.Tensor) -> torch.Tensor:
        """Per-token negative log-likelihood of ids[1:] after `session` and ids[0]."""
        logits, _ = self.forward(session, ids)
        logp = logits[:, :-1].float().log_softmax(-1)
        return -logp.gather(-1, ids[:, 1:, None].to(logp.device)).squeeze(-1)

    def generate(self, session: SessionCache, ids: torch.Tensor, max_new: int) -> list[int]:
        """Greedy decode after `session` + `ids`, stopping at end of turn."""
        stop = {self.tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}
        logits, session = self.forward(session, ids)
        out: list[int] = []
        for _ in range(max_new):
            token = int(logits[0, -1].argmax())
            if token in stop:
                break
            out.append(token)
            logits, session = self.forward(session, torch.tensor([[token]]))
        return out
