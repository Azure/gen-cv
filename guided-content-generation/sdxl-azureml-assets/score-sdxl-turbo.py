from diffusers import AutoPipelineForText2Image
import torch
import json
import io
import base64
import logging

def init():

    global pipe

    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    pipe.to("cuda")

    logging.info("Init complete")

def run(input):
    input = json.loads(input)

    # Required parameters
    prompt = input['prompt']

    # Optional parameters with default values
    negative_prompt = input.get('negative_prompt', None)
    width = int(input.get('width', 512))
    height = int(input.get('height', 512))
    n_steps = int(input.get('n_steps', 4))
    high_noise_frac = float(input.get('high_noise_frac', 0.7)) # not used 
    seed = input.get('seed', None)

    generator = torch.Generator(device="cuda").manual_seed(seed) if seed else None

    image = pipe(prompt=prompt, num_inference_steps=n_steps, guidance_scale=0.0, generator=generator).images[0]

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    logging.info("Request processed")
    return [img_str]