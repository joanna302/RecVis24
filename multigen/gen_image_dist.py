import torch
import json
import argparse
from tqdm import tqdm
from random import sample
from diffusers import StableDiffusionPipeline
from accelerate import PartialState

parser = argparse.ArgumentParser()

parser.add_argument("--text_file_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--model_name", type=str, default="CompVis/stable-diffusion-v1-4")
parser.add_argument("--img_per_prompt", type=int, default=5)
parser.add_argument("--nb_prompts", type=int, default=200)
parser.add_argument("--prompt_json_name", type=str, default="prompts.json")

args = parser.parse_args()

model_name = args.model_name
pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)

distributed_state = PartialState()
pipe.to(distributed_state.device)
pipe.set_progress_bar_config(disable=True)
print("Model loaded.")

text_file_path = args.text_file_path

with open(text_file_path, "r") as f:
    text_data = json.load(f)

# select 200 random prompts 
prompts_selected = sample(text_data, args.nb_prompts)

# prepare the data
d = []
for index in range(len(prompts_selected)):
    texts = [prompts_selected[index]["text"] for _ in range(args.img_per_prompt)]
    for j in range(args.img_per_prompt):
        d.append((index, texts[j], j))
print("Data prepared.")

print("Start generating images.")
with distributed_state.split_between_processes(d) as data:
    for index, text, j in tqdm(data):
        img_id = "{}_{}".format(index, j)
        save_path = f"{args.output_dir}/{img_id}.png"
        image = pipe(prompt=[text]).images[0]
        image.save(save_path)

# save json with selected prompts 
with open(f"{args.output_dir}/{args.prompt_json_name}", "w") as f:
    json.dump(prompts_selected, f, indent=4)
