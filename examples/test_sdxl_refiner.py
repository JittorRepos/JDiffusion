
from typing import List
from JDiffusion import StableDiffusionXLImg2ImgPipeline,StableDiffusionXLPipeline
import PIL
from PIL import Image
import jittor as jt
import numpy as np
jt.flags.use_cuda = 1

prompt = "A majestic lion jumping from a big stone at night"
base = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", use_safetensors=True
)
base_image = base(
    prompt=prompt,
    num_inference_steps=40,
    denoising_end=0.8,
    output_type="latent",
).images
base_vae = base.vae
base_text_encoder_2 = base.text_encoder_2
# to save memory, we can delete the base pipeline
del base

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base_text_encoder_2,
    vae=base_vae,
    use_safetensors=True,
)

refine_image = refiner(
    prompt=prompt,
    num_inference_steps=40,
    denoising_start=0.8,
    image=base_image,
).images[0]



refine_image.save("./output/test_sdxl_refiner_refine.png")
