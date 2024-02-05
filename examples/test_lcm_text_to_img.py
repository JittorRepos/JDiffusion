from JDiffusion import LatentConsistencyModelPipeline
import jittor as jt
jt.flags.use_cuda = 1
pipe = LatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",safety_checker=None)

prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"

# Can be set to 1~50 steps. LCM support fast inference even <= 4 steps. Recommend: 1~8 steps.
num_inference_steps = 4
images = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=8.0).images
images[0].save("./output/test_lcm_text_to_img.png")