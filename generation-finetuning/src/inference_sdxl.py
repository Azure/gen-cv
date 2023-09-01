from huggingface_hub.repocard import RepoCard
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
import torch
import os
import argparse
import mlflow



parser = argparse.ArgumentParser()

parser.add_argument("--model_path", type=str, help="path to model")

args = parser.parse_args()


# lora_model_id = <"lora-sdxl-dreambooth-id">
# card = RepoCard.load(lora_model_id)
# base_model_id = card.data.to_dict()["base_model"]

# Load the base pipeline and load the LoRA parameters into it. 
pipe = DiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.load_lora_weights(torch.load(f'{args.model_path}/pytorch_lora_weights.bin'))

# Load the refiner.
refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
)
refiner.to("cuda")

suffix = ", (high detailed skin:1.2), 8k uhd, dslr, advertising photography, high quality, film grain, real-world, unedited, photorealistic, Fujifilm XT3, natural"

# negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"

negative_prompt = '(semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck'

# prompts = [
# "product studio photography of a sks car speeding over snow in an alpine mountain",
# "product studio photography of a sks car speeding over a beach in a tropical island",
# "product studio photography of a sks car in a glamorous showroom with lots of spotlights and lighting",
# "product studio photography of a sks car in a ultra-modern urban setting, in some futuristic city",
# "A magical sks car, futuristic, stunning product studio photography, low-key lighting, bokeh, smoke effects",
# "product studio photography of a sks car on icy lake with reflections. bokeh, blue sky.",
# ]

# prompts = [
# "abstract lnl painting of a person on snow in an alpine mountain",
# "abstract lnl painting of a person on a beach in a tropical island",
# "abstract lnl painting of a woman in a ultra-modern urban setting, in some futuristic city",
# "A magical abstract lnl painting, futuristic, stunning product painting style, low-key lighting, bokeh, smoke effects",
# "abstract lnl painting of a person on icy lake with reflections. bokeh, blue sky.",
# ]

prompts = [
"in the style of ingbsx, realistic studio photo of two persons enjoying business lunch",
"in the style of ingbsx, realistic studio photo of 5 people in a business meeting enjoying their time",
"in the style of ingbsx, realistic studio photo of a CEO in a suit laughing on the phone",
"in the style of ingbsx, realistic studio photo of a business lobby with people socializing",
"in the style of ingbsx, realistic studio photo of a business professional smelling their coffee in a suit in a coffee shop",
]





num_samples = 5
guidance_scale = 7.5
num_inference_steps = 50
height = 512
width = 768

if not os.path.exists('outputs/final_images/'):
    os.mkdir('outputs/final_images/')

generator = torch.Generator("cuda").manual_seed(0)

prompt_count = 1

for prompt in prompts:
    os.mkdir(f'outputs/final_images/prompt-{prompt_count:02}')
    
    # Run inference.
    full_count = 5
    for count in range(full_count):
        
        image = pipe(prompt=prompt, negative_prompt=negative_prompt,output_type="latent", generator=generator).images[0]
        image = refiner(prompt=prompt, image=image[None, :], negative_prompt=negative_prompt, generator=generator).images[0]
        image.save(f"./outputs/final_images/prompt-{prompt_count:02}/img-{count:02}.png")
        count += 1
    
    prompt_count +=1




    