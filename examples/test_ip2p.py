from PIL import Image
from JDiffusion import StableDiffusionInstructPix2PixPipeline
import jittor as jt
jt.flags.use_cuda = 1
image = Image.open("./asset/mountain.png").convert("RGB").resize((512, 512))

pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", dtype=jt.float32)


prompt = "make the mountains snowy"
image = pipe(prompt=prompt, image=image).images[0]
image.save("./output/test_ip2p.jpg")