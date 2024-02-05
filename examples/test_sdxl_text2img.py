from JDiffusion import StableDiffusionXLPipeline
import jittor as jt
jt.flags.use_cuda = 1

pipeline_text2image = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", use_safetensors=True
)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipeline_text2image(prompt=prompt).images[0]
image.save("./output/test_sdxl_text2img.jpg")