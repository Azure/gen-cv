from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
import os
import argparse
import mlflow

device = "cuda"
# use DDIM scheduler, you can modify it to use other scheduler
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=True)

parser = argparse.ArgumentParser()

parser.add_argument("--model_path", type=str, help="path to model")
parser.add_argument("--output_dir", type=str, help="location to store final images")

args = parser.parse_args()

# modify the model path
pipe = StableDiffusionPipeline.from_pretrained(
    args.model_path,
    custom_pipeline="lpw_stable_diffusion",
    scheduler=scheduler,
    safety_checker=None,
    torch_dtype=torch.float16,
).to(device)

# enable xformers memory attention
pipe.enable_xformers_memory_efficient_attention()

suffix = ", (high detailed skin:1.2), 8k uhd, dslr, advertising photography, high quality, film grain, real-world, unedited, photorealistic, Fujifilm XT3, natural"

negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"

# prompts = [
# "product studio photography of zwx hairwax tin in a human hand",
# "product studio photography tin of zwx hairwax on cosmetics shelf",
# "product studio photography closeup tin of zwx hairwax with lake in background",
# "product studio photography tin of zwx hairwax in sand with desert sunset in background",
# "A magical zwx hairwax tin, futuristic, stunning product studio photography, low-key lighting, bokeh, smoke effects",
# "product studio photography tin of zwx hairwax on icy lake with reflections. bokeh, blue sky.",
# ]

prompts = [
"RAW photo of zwx wood figure walking through the desert",
"RAW photo of zwx wood figure swimming in a lake",
"RAW photo of zwx wood figure walking through the forest.",
"RAW photo of zwx wood figure on icy lake with reflections. bokeh, blue sky.",
"A magical zwx wood figure, futuristic, stunning product studio photography, low-key lighting, bokeh, smoke effects",
"RAW photo of zwx wood figure in a busy street in New York.",
]

num_samples = 5
guidance_scale = 7.5
num_inference_steps = 50
height = 512
width = 768

os.mkdir('outputs/final_images')
# os.mkdir(f'{args.output_dir}/final_images')

prompt_count = 1
for prompt in prompts:
    os.mkdir(f'outputs/final_images/prompt-{prompt_count:02}')
    # os.mkdir(f'{args.output_dir}/final_images/prompt-{prompts_count}')

    images = pipe(
        prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_samples,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    ).images

    count = 1
    for image in images:
        # save image to local directory
        image.save(f"./outputs/final_images/prompt-{prompt_count:02}/img-{count:02}.png")
        # image.save(f"{args.output_dir}/final_images/prompt-{prompt_count}/img-{count}.png")
        count += 1
    
    prompt_count +=1