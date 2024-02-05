from JDiffusion import StableDiffusionPipeline
import jittor as jt
jt.flags.use_cuda = 1


pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5',dtype=jt.float32,safety_checker=None)

text = ["a photo of cute cat"]

images = pipe(text*2, height=512, width=512,seed=[20,30]).images

images[0].save('./output/stable_diffusion_output_0.jpg')
images[1].save('./output/stable_diffusion_output_1.jpg')

