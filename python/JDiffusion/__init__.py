import jittor as jt
from .models import (
    AutoencoderKL,
    UNet2DConditionModel,
    ControlNetModel)
from .schedulers import (
    PNDMScheduler,
    UniPCMultistepScheduler,

)
from .pipelines import (
    StableDiffusionInstructPix2PixPipeline,
    LatentConsistencyModelPipeline,
    LatentConsistencyModelImg2ImgPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionControlNetPipeline,
    AnimateDiffPipeline,
    StableDiffusionPipeline,

)
from .utils import *
# save memory
jt.cudnn.set_max_workspace_ratio(0.0)
