from typing import TYPE_CHECKING, List

import torch

from ..base.block_hooker import BlockHooker
from ..base.attention_storage import OnlineAttentionStorage   # ← add this import
from .daam_block import CrossAttentionDAAMBlock

if TYPE_CHECKING:
    from diffusers.models.attention import CrossAttention

__all__ = ["CrossAttentionHooker"]


class CrossAttentionHooker(BlockHooker):
    def __init__(
        self,
        module: "CrossAttention",
        name: str,
        store_unconditional_hidden_states: bool = True,
        store_conditional_hidden_states: bool = False,
    ):
        super().__init__(module=module, name=name)
        self._current_hidden_state: List["torch.tensor"] = []
        self.store_conditional_hidden_states = store_conditional_hidden_states
        self.store_unconditional_hidden_states = store_unconditional_hidden_states
        self.key_states = OnlineAttentionStorage(name=name)   # ← new

    def _hooked_forward(
        hk_self: "BlockHooker",
        _: "CrossAttention",
        hidden_states: "torch.Tensor",
        **kwargs,
    ):
        if hk_self.store_unconditional_hidden_states:
            hk_self._current_hidden_state.append(hidden_states[0].cpu())
        if hk_self.store_conditional_hidden_states:
            assert hidden_states.shape[0] > 1
            hk_self._current_hidden_state.append(hidden_states[1].cpu())

        return hk_self.monkey_super("forward", hidden_states, **kwargs)

    def store_hidden_states(self) -> None:
        if not self._current_hidden_state:
            return

        device = self.module.to_q.weight.device
        is_self_attn = "attn1" in self.name   # ← attn1 = self-attention

        queries = []
        keys = []

        for c in self._current_hidden_state:
            c_gpu = c.unsqueeze(0).to(device)

            query = self.module.to_q(c_gpu)
            query = self.module.head_to_batch_dim(query)
            queries.append(query.cpu())

            if is_self_attn:                          # ← only for attn1
                key = self.module.to_k(c_gpu)
                key = self.module.head_to_batch_dim(key)
                keys.append(key.cpu())

        self.hidden_states.store(torch.stack(queries))

        if is_self_attn and keys:                     # ← store keys for attn1
            self.key_states.store(torch.stack(keys))

        self._current_hidden_state = []

    def clear(self) -> None:
        super().clear()
        self.key_states.clear()                       # ← clear keys too

    def daam_block(self, **kwargs) -> "CrossAttentionDAAMBlock":
        return CrossAttentionDAAMBlock(
            to_k=self.module.to_k,
            hidden_states=self.hidden_states,
            scale=self.module.scale,
            heads=self.module.heads,
            name=self.name,
            **kwargs,
        )