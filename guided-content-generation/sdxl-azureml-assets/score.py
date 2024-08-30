from diffusers import DiffusionPipeline, AutoencoderKL
import torch
import json
import io
import base64
import logging

def init():

    global vae, pipe, refiner

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", 
        torch_dtype=torch.float16
    )
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.to("cuda")

    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    refiner.to("cuda")

    logging.info("Init complete")

def run(input):
    input = json.loads(input)

    # Required parameters
    prompt = input['prompt']

    # Optional parameters with default values
    negative_prompt = input.get('negative_prompt', None)
    width = int(input.get('width', 1024))
    height = int(input.get('height', 1024))
    n_steps = int(input.get('n_steps', 50))
    high_noise_frac = float(input.get('high_noise_frac', 0.7))
    seed = input.get('seed', None)

    generator = torch.Generator(device="cuda").manual_seed(seed) if seed else None

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
        generator=generator
    ).images

    image = refiner(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
        generator=generator
    ).images[0]

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    logging.info("Request processed")
    return [img_str]