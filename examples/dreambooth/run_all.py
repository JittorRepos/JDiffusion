import json, os, tqdm
import jittor as jt

from JDiffusion.pipelines import StableDiffusionPipeline

max_num = 15
dataset_root = "the-path-to-dataset"

with jt.no_grad():
    for tempid in tqdm.tqdm(range(0, max_num)):
        taskid = "{:0>2d}".format(tempid)
        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1").to("cuda")
        pipe.load_lora_weights(f"style/style_{taskid}")

        # load json
        with open(f"{dataset_root}/{taskid}/prompt.json", "r") as file:
            prompts = json.load(file)

        for id, prompt in prompts.items():
            print(prompt)
            image = pipe(prompt + f" in style_{taskid}", num_inference_steps=25, width=512, height=512).images[0]
            os.makedirs(f"./output/{taskid}", exist_ok=True)
            image.save(f"./output/{taskid}/{prompt}.png")
