
import argparse
import os
import json 
import tqdm as tqdm 

from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

if __name__ == "__main__":

    device = 'cuda'

    # Load the model in half-precision
    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto")
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    # Prepare a batch of two prompts
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is the caption with many adjectives and nouns for the image ? "},
            ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    #prompt_2 = processor.apply_chat_template(conversation_2, add_generation_prompt=True)

    
    with open('input.json', 'r') as json_file:
        json_list = json.load(json_file)

    output_json_data = []
    for i in range(len(json_list)):
        dict_img = {}
        d = json_list[i]
        image_path = d['img_path']
        dict_img['img_path']=image_path
        dict_img['caption']=d['caption']
        image = Image.open(image_path)

        # We can simply feed images in the order they have to be used in the text prompt
        inputs = processor(images=image, text=prompt, padding=True, return_tensors="pt").to(model.device, torch.float16)
        #inputs_2 = processor(images=image, text=prompt_2, padding=True, return_tensors="pt").to(model.device, torch.float16)

        # Generate
        generate_ids = model.generate(**inputs, max_new_tokens=50)
        output_caption=processor.batch_decode(generate_ids, skip_special_tokens=True)

        # Generate
        #generate_ids = model.generate(**inputs_2, max_new_tokens=50)
        #output_caption_adj =processor.batch_decode(generate_ids, skip_special_tokens=True)

        x = output_caption[0].split('ASSISTANT:')
        dict_img['caption_llava']=x[-1]

        #x = output_caption[0].split('ASSISTANT:')
        #dict_img['caption_llava_adj']=x[-1]

        output_json_data.append(dict_img)

    with open('output_caption.json', 'w') as fw:
        for d in output_json_data:
            fw.write(json.dumps(d))
            fw.write("\n")


