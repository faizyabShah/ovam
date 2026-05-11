from diffusers import StableDiffusionXLPipeline
from ..stable_diffusion.pipeline_hooker import StableDiffusionHooker
from ..stable_diffusion.block_hooker import CrossAttentionHooker
from ..stable_diffusion.locator import UNetCrossAttentionLocator
from .daam_module import StableDiffusionXLDAAM


class StableDiffusionXLHooker(StableDiffusionHooker):

    def __init__(
        self,
        pipeline: StableDiffusionXLPipeline,
        locate_middle_block: bool = False,
        block_hooker_kwargs: dict = {},
        locator_kwargs: dict = {},
    ):
        # We call PipelineHooker.__init__ directly to inject XL's daam_module_class,
        # bypassing StableDiffusionHooker which hardcodes StableDiffusionDAAM
        from ..base.pipeline_hooker import PipelineHooker
        PipelineHooker.__init__(
            self,
            pipeline,
            locator=UNetCrossAttentionLocator(
                locate_middle_block=locate_middle_block,
                **locator_kwargs,
            ),
            block_hooker_class=CrossAttentionHooker,
            daam_module_class=StableDiffusionXLDAAM,   # <-- only change
            block_hooker_kwargs=block_hooker_kwargs,
        )