from PIL import Image
import cv2
import numpy as np
import jittor as jt
jt.flags.use_cuda = 1

image = Image.open("./asset/input_image_vermeer.png")
image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)


from JDiffusion import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", use_safetensors=True)
pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, use_safetensors=True)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)


output = pipe(
    "the mona lisa", image=canny_image
).images[0]
output.save("./output/test_controlnet.png")