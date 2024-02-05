from diffusers import AutoPipelineForInpainting
from JDiffusion import StableDiffusionXLPipeline
from PIL import Image
import jittor as jt
jt.flags.use_cuda = 1
pipeline_text2image = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", use_safetensors=True
)


pipeline = AutoPipelineForInpainting.from_pipe(pipeline_text2image)

init_image = Image.open("./asset/sdxl-text2img.png").convert("RGB").resize((512, 512))
mask_image = Image.open("./asset/sdxl-inpaint-mask.png").convert("RGB").resize((512, 512))

prompt = "A deep sea diver floating"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, strength=0.85, guidance_scale=12.5).images[0]
image.save("./output/test_sdxl_inpainting.png")