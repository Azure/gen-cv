import torch
import io
import os
import logging
import json
import math
import cv2
import numpy as np
from compel import Compel
from base64 import b64encode
from PIL import Image, ImageDraw
from fastdownload import FastDownload
from safetensors.torch import load_file
from azureml.contrib.services.aml_response import AMLResponse

from transformers import pipeline
from diffusers import DPMSolverMultistepScheduler
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from controlnet_aux import ContentShuffleDetector, HEDdetector, MLSDdetector
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_controlnet import MultiControlNetModel
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, StableDiffusionControlNetInpaintPipeline, ControlNetModel, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline


def prepare_canny_image(image_base, low_threshold=200, high_threshold=200):
    """
    This function takes an image and applies the Canny edge detection algorithm on it. 
    It then concatenates the edge detected image along the channel axis and converts it into an Image object before returning it.
    """
    image = np.array(image_base)

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    return image


def handle_style_image(image_path, shuffle=False):
    """
    This function takes an image path and returns an image object. If shuffle is set to True, it applies a Content Shuffle Detector on the image before returning it.
    """
    img = get_image_object(image_path)
    
    if shuffle:
        processor = ContentShuffleDetector()
        img = processor(img)
    
    return img


def prepare_hed_scribble_image(image_base, hed, image_width):
    """
    This function takes an image, HED detector, and image width as input, and returns the HED detected image with scribbles at the specified image width.
    """
    if not hed:
        hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')

    image = hed(image_base, scribble=True, detect_resolution=image_width, image_resolution=image_width)

    return image


def prepare_mlsd_image(image_base, mlsd, image_width):
    """
    This function takes an image, MLSD detector, and image width as input, and returns the MLSD detected image at the specified image width.
    """
    if not mlsd:
        mlsd = MLSDdetector.from_pretrained('lllyasviel/ControlNet')

    image = mlsd(image_base, detect_resolution=image_width, image_resolution=image_width)

    return image

def prepare_depth_image(image_base, depth_estimator):
    """
    This function takes an image and a depth estimator, and returns a depth estimated image.
    """
    if not depth_estimator:
        depth_estimator = pipeline('depth-estimation')

    image = depth_estimator(image_base)['depth']
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)

    return image


def get_control_net_to_img(model_id="SG161222/Realistic_Vision_V2.0", cont_model="lllyasviel/sd-controlnet-scribble"):
    """
    This function takes a model ID and a control model as input, and returns a pre-trained StableDiffusionControlNetPipeline object with the specified controlnet and scheduler.
    """
    controlnet = get_control_net_model(cont_model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device', device)
    print('Controlnet downloaded')
    pipe = StableDiffusionControlNetPipeline.from_pretrained(model_id, 
                                                            controlnet=controlnet, 
                                                            safety_checker=None, 
                                                            torch_dtype=torch.float16, 
                                                            # cache_dir=cache_dir
                                                            ).to(device)
    
    print('Pipe downloaded')
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    print('Schedule Set')
    pipe.enable_xformers_memory_efficient_attention()
    print('Xformer Set')

    return pipe


def get_txt_img_pipeline(model_id):
    """
    This function takes a model ID as input, and returns a pre-trained StableDiffusionPipeline object with Euler Ancestral Discrete Scheduler.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device', device)
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    eulerScheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe.scheduler = eulerScheduler
    pipe.enable_xformers_memory_efficient_attention()

    return pipe


def get_control_net_model(cont_model):
    """
    This function takes a control model as input, and returns a pre-trained ControlNetModel object.
    """
    controlnet = ControlNetModel.from_pretrained(cont_model, torch_dtype=torch.float16).to("cuda")

    return controlnet

def get_img_img_pipeline(model_id):
    """
    This function takes a model ID as input, and returns a pre-trained StableDiffusionImg2ImgPipeline object with Euler Ancestral Discrete Scheduler.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device', device)
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    eulerScheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe.scheduler = eulerScheduler
    pipe.enable_xformers_memory_efficient_attention()

    return pipe

def get_base_sdxl_pipeline(model_id="stabilityai/stable-diffusion-xl-base-1.0"):
    """
    This function takes a model ID as input, and returns a pre-trained StableDiffusionXLPipeline object with Euler Ancestral Discrete Scheduler.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device', device)
    pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe.enable_xformers_memory_efficient_attention()

    return pipe


def get_img_img_sdxl_pipeline(model_id):
    """
    This function takes a model ID as input, and returns a pre-trained StableDiffusionXLImg2ImgPipeline object with Euler Ancestral Discrete Scheduler.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device', device)
    pipe_img_t_img = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to(device)
    pipe_img_t_img.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe_img_t_img.enable_xformers_memory_efficient_attention()

    return pipe_img_t_img


def get_inpainting_pipeline(model_id):
    """
    This function takes a model ID as input, and returns a pre-trained StableDiffusionInpaintPipeline object with Euler Ancestral Discrete Scheduler.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device', device)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    eulerScheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe.scheduler = eulerScheduler
    pipe.enable_xformers_memory_efficient_attention()

    return pipe

def get_inpainting_cnet_pipeline(model_id):
    """
    This function takes a model ID as input, and returns a pre-trained StableDiffusionInpaintPipeline object with Euler Ancestral Discrete Scheduler.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device', device)
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16).to(device)
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(model_id, controlnet=controlnet, torch_dtype=torch.float16).to("cuda")
    eulerScheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe.scheduler = eulerScheduler
    pipe.enable_xformers_memory_efficient_attention()

    return pipe

def load_model():
    """
    This function loads the necessary models and their dependencies for image generation tasks and returns them as dictionaries.
    """
    model_name_sd15 = "charapennikaurm/Reliberate"
    cnet_model_name = "lllyasviel/control_v11p_sd15_scribble"
    print(f'the model name: [{model_name_sd15}]')

    pipe_txt_img = get_txt_img_pipeline(model_name_sd15)
    pipe_img_img = get_img_img_pipeline(model_name_sd15)
    pipe_base_sdxl = get_base_sdxl_pipeline("stabilityai/stable-diffusion-xl-base-1.0")
    pipe_sdxl_refiner = get_img_img_sdxl_pipeline("stabilityai/stable-diffusion-xl-refiner-1.0")

    cnet_pipe = get_control_net_to_img(model_name_sd15, cnet_model_name)
    cnet_model_scribble = get_control_net_model("lllyasviel/control_v11p_sd15_scribble")
    cnet_model_depth = get_control_net_model("lllyasviel/control_v11f1p_sd15_depth")
    cnet_model_shuffle = get_control_net_model("lllyasviel/control_v11e_sd15_shuffle")

    print('Model objects and their dependencies are loaded')

    mlsd = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
    depth_estimator = pipeline("depth-estimation", model ="Intel/dpt-hybrid-midas")


    inpaint_model_name = "redstonehero/dreamshaper-inpainting"
    pipe_inpaint = get_inpainting_pipeline(inpaint_model_name)

    pipe_inpaint_cnet = get_inpainting_cnet_pipeline(inpaint_model_name)

    compel_proc = Compel(tokenizer=pipe_img_img.tokenizer, text_encoder=pipe_img_img.text_encoder)

    base_models = {
        'cnet_pipe': cnet_pipe,
        'pipe_img_img': pipe_img_img,
        'pipe_txt_img': pipe_txt_img,
        'pipe_base_sdxl': pipe_base_sdxl,
        'pipe_sdxl_refiner': pipe_sdxl_refiner,
        'pipe_inpaint': pipe_inpaint,
        'pipe_inpaint_cnet': pipe_inpaint_cnet,
    }

    cnet_models = {
        'cnet_model_scribble': cnet_model_scribble,
        'cnet_model_depth': cnet_model_depth,
        'cnet_model_shuffle': cnet_model_shuffle,
        'mlsd': mlsd,
        'depth_estimator': depth_estimator
    }

    return base_models, cnet_models, compel_proc

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global base_models, cnet_models, compel_proc
    
    base_models, cnet_models, compel_proc = load_model()

    logging.info("Init complete")


def get_image_object(image_url):
    """
    This function takes an image URL and returns an Image object.
    """
    p = FastDownload().download(image_url, force=True)
    init_image = Image.open(p).convert("RGB")
    return init_image

def prepare_response(images):
    """
    This function takes a list of images and converts them to a dictionary of base64 encoded strings.
    """
    ENCODING = 'utf-8'
    dic_response = {}
    for i, image in enumerate(images):
        output = io.BytesIO()
        image.save(output, format="JPEG")
        base64_bytes = b64encode(output.getvalue())
        base64_string = base64_bytes.decode(ENCODING)
        dic_response[f'image_{i}'] = base64_string

    return dic_response



def design(prompt, image=None, num_images_per_prompt=4, negative_prompt=None, strength=0.65, guidance_scale=7.5, num_inference_steps=50, seed=None, design_type='TXT_TO_IMG', mask=None, other_args=None, data=None):
    """
    This function takes various parameters like prompt, image, seed, design_type, etc., and generates images based on the specified design type. It returns a list of generated images.
    """
    generator = None
    if seed:
        generator = torch.manual_seed(seed)
    else:
        generator = torch.manual_seed(0)

    print('other_args', other_args)
    dic_conditioning_scales = {}
    
    if other_args and 'CNET_CONFIGS' in other_args:
        cnet_configs = other_args['CNET_CONFIGS']
        dic_conditioning_scales = cnet_configs['controlnet_conditioning_scale']
        if 'Scheduler' in cnet_configs and cnet_configs['Scheduler'] == 'DPM':
            print('converting scheduler to DPM')
            base_models["cnet_pipe"].scheduler = DPMSolverMultistepScheduler.from_config(base_models["cnet_pipe"].scheduler.config)
        else:
            base_models["cnet_pipe"].scheduler = EulerAncestralDiscreteScheduler.from_config(base_models["cnet_pipe"].scheduler.config)

    print('dic_conditioning_scales', dic_conditioning_scales)

    li_images = []

    if design_type == 'TXT_TO_IMG':
        li_images = base_models["pipe_txt_img"](prompt_embeds=prompt, num_images_per_prompt=num_images_per_prompt, negative_prompt_embeds=negative_prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images

    elif design_type == 'IMG_TO_IMG':
        li_images = base_models["pipe_img_img"](prompt_embeds=prompt, image=image, num_images_per_prompt=num_images_per_prompt, negative_prompt_embeds=negative_prompt, strength=strength, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images
        
    elif design_type == 'TXT_TO_IMG_SDXL':
        prompt = data['prompt']
        negative_prompt = data['negative_prompt']
        li_base_images = base_models["pipe_base_sdxl"](prompt=prompt, num_images_per_prompt=num_images_per_prompt, negative_prompt=negative_prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images
        for image in li_base_images:
            refined_image = base_models["pipe_sdxl_refiner"](prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, strength=strength, image=image).images[0]
            li_images.append(refined_image)

    elif design_type == 'IMG_TO_IMG_SDXL':
        prompt = data['prompt']
        negative_prompt = data['negative_prompt']
        li_images = base_models["pipe_sdxl_refiner"](prompt=prompt, image=image, num_images_per_prompt=num_images_per_prompt, negative_prompt=negative_prompt, strength=strength, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images
    
    elif design_type == 'CNET_CANNY':
        canny_p = dic_conditioning_scales.get('CANNY', 1.0)
        canny_image = prepare_canny_image(image)
        
        base_models["cnet_pipe"].controlnet = cnet_models['cnet_model_scribble']
        li_images = base_models["cnet_pipe"](prompt_embeds=prompt, image=canny_image, controlnet_conditioning_scale=canny_p, num_images_per_prompt=num_images_per_prompt, negative_prompt_embeds=negative_prompt, guidance_scale=guidance_scale, generator=generator, num_inference_steps=num_inference_steps).images
        li_images.append(canny_image)

    elif design_type == "CNET_CANNY_DEPTH":
        canny_image = prepare_canny_image(image)
        depth_image = prepare_depth_image(image, cnet_models["depth_estimator"])

        canny_p = dic_conditioning_scales.get('CANNY', 1.0)
        depth_p = dic_conditioning_scales.get('DEPTH', 0.3)
        
        base_models["cnet_pipe"].controlnet = MultiControlNetModel([cnet_models['cnet_model_scribble'], cnet_models['cnet_model_depth']])
        li_images = base_models["cnet_pipe"](prompt_embeds=prompt, image=[canny_image, depth_image], controlnet_conditioning_scale=[canny_p, depth_p], num_images_per_prompt=num_images_per_prompt, negative_prompt_embeds=negative_prompt, guidance_scale=guidance_scale, generator=generator, num_inference_steps=num_inference_steps).images
        li_images.append(canny_image)
        li_images.append(depth_image)

    elif design_type == 'IN_PAINTING':
        li_images = base_models["pipe_inpaint"](prompt_embeds=prompt, image=image.resize((512, 512)), mask_image=mask.resize((512, 512)), num_images_per_prompt=num_images_per_prompt, negative_prompt_embeds=negative_prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images

    return li_images


def run(raw_data):
    """
     This function takes raw data as input, processes it, and calls the design function to generate images.
     It then prepares the response and returns it.
    """
    logging.info("Request received")
    print(f'raw data: {raw_data}')
    data = json.loads(raw_data)["data"]
    print(f'data: {data}')

    prompt = data['prompt']
    negative_prompt = data['negative_prompt']
    seed = data['seed']
    num_images_per_prompt = data['num_images_per_prompt']
    guidance_scale = data['guidance_scale']
    num_inference_steps = data['num_inference_steps']
    design_type = data['design_type']

    image_url = None
    mask_url = None
    mask = None
    other_args = None
    image = None
    strength = data['strength']

    if 'mask_image' in data:
        mask_url = data['mask_image']
        mask = get_image_object(mask_url)

    if 'other_args' in data:
        other_args = data['other_args']


    if 'image_url' in data:
        image_url = data['image_url']
        image = get_image_object(image_url)

    if 'strength' in data:
        strength = data['strength']

    prompt_embeds = compel_proc(prompt)
    n_prompt_embeds = compel_proc(negative_prompt)
    with torch.inference_mode():
        images = design(prompt=prompt_embeds, image=image, 
                        num_images_per_prompt=num_images_per_prompt, 
                        negative_prompt=n_prompt_embeds, strength=strength, 
                        guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
                        seed=seed, design_type=design_type, mask=mask, other_args=other_args, data=data)
    
    preped_response = prepare_response(images)
    resp = AMLResponse(message=preped_response, status_code=200, json_str=True)

    return resp

