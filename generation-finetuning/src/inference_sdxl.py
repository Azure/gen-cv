from huggingface_hub.repocard import RepoCard
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
import torch
import os
import argparse
import mlflow



parser = argparse.ArgumentParser()

parser.add_argument("--model_path", type=str, help="path to model")

args = parser.parse_args()




if os.path.exists(f'{args.model_path}/pytorch_lora_weights.safetensors'):
    safetensor_weights = True
    weights_path = f'{args.model_path}/pytorch_lora_weights.safetensors'
elif os.path.exists(f'{args.model_path}/pytorch_lora_weights.bin'):
    safetensor_weights = False
    weights_path_bin = f'{args.model_path}/pytorch_lora_weights.bin'

    
# Load the base pipeline and load the LoRA parameters into it. 
pipe = DiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=torch.float16)
pipe = pipe.to("cuda")

if safetensor_weights:
    pipe.load_lora_weights(weights_path, use_safetensors=True)
else:
    pipe.load_lora_weights(torch.load(weights_path_bin))


# Load the refiner.
refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
)
refiner.to("cuda")

suffix = ", (high detailed skin:1.2), 8k uhd, dslr, advertising photography, high quality, film grain, real-world, unedited, photorealistic, Fujifilm XT3, natural"

# negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"

negative_prompt = '(semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck'

prompts = [
"product studio photography of zwx hairwax tin in a human hand",
"product studio photography tin of zwx hairwax on cosmetics shelf",
"product studio photography closeup of zwx hairwax with lake in background",
"product studio photography tin of zwx hairwax in sand with desert sunset in background",
"A magical zwx hairwax tin, futuristic, stunning product studio photography, low-key lighting, bokeh, smoke effects",
"product studio photography tin of zwx hairwax on icy lake with reflections. bokeh, blue sky.",
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




    