from JDiffusion import StableDiffusionXLImg2ImgPipeline
from PIL import Image
import jittor as jt
jt.flags.use_cuda = 1
pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")

init_image = Image.open("./asset/sdxl-text2img.png").convert("RGB")
prompt = "a dog catching a frisbee in the jungle"
image = pipeline(prompt, image=init_image, strength=0.8, guidance_scale=10.5).images[0]
image.save("./output/test_sdxl_img2img.png")