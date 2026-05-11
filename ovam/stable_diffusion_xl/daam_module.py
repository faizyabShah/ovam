import torch
from ..stable_diffusion.daam_module import StableDiffusionDAAM
from ..utils.text_encoding import encode_text as _encode_text
from typing import Optional

class StableDiffusionXLDAAM(StableDiffusionDAAM):

    def __init__(self, blocks, pipeline, **kwargs):
        super().__init__(blocks, pipeline, **kwargs)
        self.tokenizer_2 = pipeline.tokenizer_2
        self.text_encoder_2 = pipeline.text_encoder_2

    def encode_text(
        self,
        text: str,
        context_sentence: Optional[str] = None,
        remove_special_tokens: bool = True,
        padding=False,
    ) -> torch.Tensor:

        emb_1 = _encode_text(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            text=text,
            context_sentence=context_sentence,
            remove_special_tokens=remove_special_tokens,
            padding=padding,
        )  # (n_tokens_1, 768)

        emb_2 = _encode_text(
            tokenizer=self.tokenizer_2,
            text_encoder=self.text_encoder_2,
            text=text,
            context_sentence=context_sentence,
            remove_special_tokens=remove_special_tokens,
            padding=padding,
        )  # (n_tokens_2, 1280)

        # Both encoders agree for simple prompts (no context_sentence).
        # With context_sentence, different BPE vocabs can yield n_tokens_1 ≠ n_tokens_2.
        # We truncate to the shorter to keep shapes compatible — a safe fallback
        # since both encoders cover the same semantic content.
        if emb_1.shape[0] != emb_2.shape[0]:
            min_len = min(emb_1.shape[0], emb_2.shape[0])
            emb_1 = emb_1[:min_len]
            emb_2 = emb_2[:min_len]

        return torch.cat([emb_1, emb_2], dim=-1)  # (n_tokens, 2048)