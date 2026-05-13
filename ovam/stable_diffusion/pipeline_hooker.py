import math
import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING, Optional, Tuple

from ..base.store_hooker import StoreHiddenStatesHooker
from ..base.pipeline_hooker import PipelineHooker
from .block_hooker import CrossAttentionHooker
from .daam_module import StableDiffusionDAAM
from .locator import UNetCrossAttentionLocator

if TYPE_CHECKING:
    from diffusers import StableDiffusionPipeline
    from ..utils.attention_ops import ActivationTypeVar, AggregationTypeVar


class StableDiffusionHooker(PipelineHooker):

    def __init__(
        self,
        pipeline: "StableDiffusionPipeline",
        locate_middle_block: bool = False,
        block_hooker_kwargs: dict = {},
        locator_kwargs: dict = {},
    ):
        super().__init__(
            pipeline,
            locator=UNetCrossAttentionLocator(
                locate_middle_block=locate_middle_block,
                **locator_kwargs,
            ),
            block_hooker_class=CrossAttentionHooker,
            daam_module_class=StableDiffusionDAAM,
            block_hooker_kwargs=block_hooker_kwargs,
        )

    def _register_extra_hooks(self):
        self.register_hook(
            StoreHiddenStatesHooker(
                module=self.pipeline.image_processor,
                parent_trace=self,
                function_patched="postprocess",
            )
        )

    @property
    def cross_attention_hookers(self):
        return self.module[:-1]

    def get_self_attention_map(
        self,
        size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Compute self-attention map from stored attn1 hidden states.
        Uses only 64x64 blocks (highest resolution in SD1.5 UNet)
        as per the OVAM paper Section 3.4.

        Returns a (H, W) tensor — or upsampled to `size` if provided.
        """
        # only attn1 hookers have key_states stored
        self_attn_hookers = [
            h for h in self.cross_attention_hookers
            if "attn1" in h.name and len(h.key_states) > 0
        ]

        all_maps = []

        for hooker in self_attn_hookers:
            for queries_per_image, keys_per_image in zip(
                hooker.hidden_states, hooker.key_states
            ):
                # shape: (n_epochs, n_heads, seq_len, head_dim)
                seq_len = queries_per_image.shape[-2]
                h = w = int(math.sqrt(seq_len))

                # paper: highest-resolution blocks only (64x64)
                if h != 64:
                    continue

                epoch_maps = []
                for q, k in zip(queries_per_image, keys_per_image):
                    # q, k: (n_heads, seq_len, head_dim)
                    scale = q.shape[-1] ** -0.5
                    # (n_heads, seq_len, seq_len)
                    attn = torch.bmm(
                        q.float(), k.float().transpose(-1, -2)
                    ) * scale
                    attn = attn.softmax(dim=-1)
                    # average attention received per position across heads
                    # → (seq_len,) → (h, w)
                    spatial = attn.mean(dim=0).mean(dim=0).reshape(h, w)
                    epoch_maps.append(spatial)

                # average across timesteps
                all_maps.append(torch.stack(epoch_maps).mean(0))  # (h, w)

        if not all_maps:
            raise RuntimeError(
                "No self-attention maps found. "
                "Make sure to use locator_kwargs={'locate_attn1': True}."
            )

        # average across all 64x64 blocks → (h, w)
        result = torch.stack(all_maps).mean(0)

        if size is not None:
            result = F.interpolate(
                result.unsqueeze(0).unsqueeze(0).float(),
                size=size,
                mode="bilinear",
                align_corners=False,
            ).squeeze()

        return result

    def get_ovam_callable(
        self,
        heads_epochs_activation: "ActivationTypeVar" = "token_softmax",  # "linear" for linear daam,
        heads_epochs_aggregation: "AggregationTypeVar" = "sum",
        heads_activation: "ActivationTypeVar" = "linear",
        heads_aggregation: "AggregationTypeVar" = "sum",
        block_interpolation_mode: str = "bilinear",
        blocks_activation: "ActivationTypeVar" = "linear",
        heatmaps_activation: Optional["ActivationTypeVar"] = None,
        heatmaps_aggregation: "AggregationTypeVar" = "sum",
        expand_size: Optional[Tuple[int, int]] = None,
        expand_interpolation_mode: str = "bilinear",
        block_kwargs: dict = {},
        module_kwargs: dict = {},
    ) -> "StableDiffusionDAAM":
        """
        Buld a OVAM module with the current hidden states.
        This module can be evaluated to obtain the attention maps
        for any arbitrary text embedding (sentence) or token (word).

        Arguments
        ---------
        heads_epochs_activation: str or Callable, default="token_softmax"
            The activation function applied to the attentions of each attention
            head of each block of each epoch. By default the attention heads
            of the epoch are softmaxed in the token dimension. To execute the
            linear damm version set this parameter to "linear". Activation
            used in a tensor of shape (n_epochs, heads, n_tokens,
            latent_size / factor, latent_size / factor) where factor
            depends on the block.
        heads_epochs_aggregation: str or Callable, default="sum"
            The aggregation function applied to aggregate the attention
            heads across all epochs. By default the epochs are summed.
            Collapses the `n_epochs` dimension of a tensor with shape
            (n_epochs, heads, n_tokens, latent_size / factor, latent_size / factor)
        heads_activation : str or Callable, default="linear"
            The activation function to apply to each of the attention blocks
            after aggregate their epochs. By default the attention blocks are
            not activated. Recieves a tensor of shape (heads, n_tokens,
            latent_size / factor, latent_size / factor) where factor depends
            on the block.
        heads_aggregation : str or Callable, default="sum"
            Aggregation function applied to the attention heads of each block.
            By default the attention heads are summed. Collapses the `heads`
            dimension of a tensor with shape (heads, n_tokens, latent_size / factor,
            latent_size / factor).
        block_interpolation_mode : str, default="bilinear"
            The interpolation mode to use when expanding the attention maps
            of each block to normalize all sizes to the original latent size.
        blocks_activation : str or Callable, default="linear"
            The activation function to apply to each of the attention blocks
            after the aggregation of the attention heads. By default the
            attention blocks are not activated.
        """

        var_module_kwargs = {
            "heatmaps_activation": heatmaps_activation,
            "heatmaps_aggregation": heatmaps_aggregation,
            "expand_size": expand_size,
            "expand_interpolation_mode": expand_interpolation_mode,
            "block_interpolation_mode": block_interpolation_mode,
        }
        var_module_kwargs.update(module_kwargs)

        var_block_kwargs = {
            "heads_activation": heads_activation,
            "blocks_activation": blocks_activation,
            "heads_epochs_activation": heads_epochs_activation,
            "heads_aggregation": heads_aggregation,
            "heads_epochs_aggregation": heads_epochs_aggregation,
        }
        var_block_kwargs.update(block_kwargs)

        return super().daam(
            module_kwargs=var_module_kwargs,
            block_kwargs=var_block_kwargs,
        )
