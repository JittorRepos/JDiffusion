import json, os, tqdm, torch

from diffusers import DiffusionPipeline

max_num = 43
dataset_root = "the-path-to-dataset"

with torch.no_grad():
    for taskid in tqdm.tqdm(range(0, max_num)):
        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1").to("cuda")
        pipe.load_lora_weights(f"style/style_{taskid}")

        # load json
        with open(f"{dataset_root}/{taskid}/prompt.json", "r") as file:
            prompts = json.load(file)

        for id, prompt in prompts.items():
            print(prompt)
            image = pipe(prompt + f" in style_{taskid}", num_inference_steps=25).images[0]
            os.makedirs(f"./output/{taskid}", exist_ok=True)
            image.save(f"./output/{taskid}/{prompt}.png")