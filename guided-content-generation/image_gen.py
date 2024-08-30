import os
import re
import json
import time
import base64
import shutil
import random
import requests
import replicate
import streamlit as st
from streamlit import session_state as ss
from VideoTools import VideoCreator, VideoAnalyzer
from instructions import (
    basic_system_message, neutralize_competitors_system_message,
    replace_competitors_system_message, negative_prompt, gpt4o_system_message,
    gpt4o_user_prompt
)
from utils import (
    analyze_image_gpt4o, create_sdxl_image, display_moderation_results,
    azure_image_analysis_predict, check_and_reduce_image_size
)

# Streamlit Page Configuration
st.set_page_config(layout="centered", initial_sidebar_state="auto")
st.title("Create Your Perfect Scene")

styles = ['Photorealistic', 'Comic', 'Pop-art', 'Anime', 'Manga', 'Cyberpunk', 'Steampunk']

with st.sidebar:
    brand_protection_help = (
        "Specify how to handle brands or product names in the user prompt:\n\n"
        "__off:__ leave unchanged.\n\n"
        "__neutralize:__ competitors are replaced by neutral product category.\n\n"
        "__replace:__ competitors are replaced by specified brands."
    )
    
    brand_protection = st.radio("Protect brands/products:", ["off", "neutralize", "replace"], horizontal=True, help=brand_protection_help)
    brands = st.text_input('Brands to protect:', 'Microsoft, XBox', help="Enter brand and product names that you want to enforce.") if brand_protection != "off" else None
    imgen_deployment = st.selectbox("Image generation model:", ss.imgen_models, index=0, help="Image models with specified API key.")
    col1, col2 = st.columns(2)

    if imgen_deployment == 'FLUX.1 [pro]':
        flux_pro_aspect_ratios = ["1:1", "16:9", "2:3", "3:2", "4:5", "5:4", "9:16"]
        flux_pro_aspect_ratio = col1.selectbox("Aspect Ratio", flux_pro_aspect_ratios, index=0)
        flux_pro_n_steps = col2.number_input('Steps', min_value=1, max_value=50, value=25, format="%d", help="Number of diffusion steps between 1 and 50.")
        flux_pro_guidance = col1.number_input('Guidance', min_value=2, max_value=5, value=3, format="%d", help="Controls the balance between adherence to the text prompt and image quality/diversity between 2 and 5. Higher values make the output more closely match the prompt but may reduce overall image quality.")
        flux_pro_interval = col2.number_input('Interval', min_value=1, max_value=4, value=2, format="%d", help="Setting that increases the variance in possible outputs, letting the model be more dynamic in outputs.")

    elif imgen_deployment == 'Stable Diffusion XL':
        sdxl_width = col1.number_input('Width', value=1024, format="%d")
        sdxl_height = col2.number_input('Height', value=1024, format="%d")
        sdxl_steps = col1.number_input('Steps', min_value=30, max_value=100, value=50, format="%d", help="Number of diffusion steps between 30 and 100.")
        sdxl_seed_str = col2.text_input('Seed', value=None)
        sdxl_seed = int(sdxl_seed_str) if sdxl_seed_str else None
        sdxl_use_negative_prompt = st.checkbox("Use negative prompt", True)
        
    elif imgen_deployment == 'DALL E-3':
        dalle_sizes = ['1024x768', '1024x1024', '1792x1024', '1024x1792']
        dalle_quality = col1.checkbox("HD", True)
        dalle_style = col2.checkbox("Vivid", True)
        dalle_size = st.selectbox("Image size:", dalle_sizes, index=1, help="Image resolution width x height")

    elif imgen_deployment == "Stable Diffusion 3":
        sd3_aspect_ratios = ["1:1", "16:9", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"]
        sd3_models = ["sd3-medium", "sd3-large", "sd3-large-turbo"]
        sd3_aspect_ratio = col2.selectbox("Aspect Ratio", sd3_aspect_ratios, index=0)
        sd3_model = col1.selectbox("Model", sd3_models, index=1)

    imgen_style = st.selectbox("Visual style:", styles, index=0)

    st.write('Brand detection models:')
    bd_col1, bd_col2 = st.columns(2)
    llm_image_analysis = bd_col1.checkbox('GPT-4o', True)
    azure_image_analysis = bd_col2.checkbox('AI Vision', False, disabled=not ss.azure_ai_vision, help='Use a custom object detection model trained on your data.')
    
    if azure_image_analysis:
        threshold = st.slider('Azure Image Analysis Threshold', 0.0, 1.0, 0.6, 0.1, help='Lower values increase the likelihood of detecting objects but also increases false positives.')

    help_str = "The selected severity is the lowest level that is considered harmful. Example: selecting 'medium' means that all content identified as medium or higher would be flagged."
    st.markdown("Image moderation thresholds:", help=help_str)
    col1, col2 = st.columns(2)
    st.markdown(ss.custom_css, unsafe_allow_html=True)
    hate_thresh = ss.severity_to_id[col1.select_slider("Hate", ss.severity_to_id.keys(), 'low')]
    selfharm_thresh = ss.severity_to_id[col2.select_slider("SelfHarm", ss.severity_to_id.keys(), 'low')]
    sexual_thresh = ss.severity_to_id[col1.select_slider("Sexual", ss.severity_to_id.keys(), 'low')]
    violence_thresh = ss.severity_to_id[col2.select_slider("Violence", ss.severity_to_id.keys(), 'low')]

    save, cache = st.columns(2)
    save_images = save.checkbox("Save images", True)
    streamlit_cache = cache.checkbox("Image cache", False, help="Applies to image prompts and generations. Video analysis results are always cached.")

    if st.button("Clear cache and restart", use_container_width=True, type='primary'):
        st.session_state.clear()
        st.cache_data.clear()
        st.rerun()

def img2video(image_dir, filename):
    """Converts an image to a video clip using VideoCreator."""
    with st.spinner('Transforming image into video clip ...'):
        video_creator = VideoCreator()
        video_target_folder = "./videos/generated"
        resized_filename = f"resized-{filename}"
        source_img_path = os.path.join(image_dir, filename)
        temp_image_path = os.path.join(video_target_folder, resized_filename)
        shutil.copy(source_img_path, temp_image_path)
        video_creator.resize_image_to_allowed_resolutions(video_target_folder, resized_filename)
        video_path = video_creator.image_to_video(video_target_folder, resized_filename, ss.stability_api_key)
        st.video(video_path, autoplay=True, loop=True)

user_prompt = st.chat_input("Describe your image")
if user_prompt:
    if brand_protection == "off":
        system_message = basic_system_message.format(style=imgen_style, model=imgen_deployment)
    elif brand_protection == "neutralize":
        system_message = neutralize_competitors_system_message.format(style=imgen_style, model=imgen_deployment, brands=brands)
    elif brand_protection == "replace":
        system_message = replace_competitors_system_message.format(style=imgen_style, model=imgen_deployment, brands=brands)

    ss.sys_messg = system_message

    @st.cache_data(show_spinner=False)
    def refine_user_prompt(prompt, model, system_message, cache=0.0):
        """Refines the user prompt by interacting with the model."""
        messages = [{'role': 'system', 'content': system_message}, {'role': 'user', 'content': prompt}]
        response = ss.aoai_client.chat.completions.create(model=model, messages=messages, temperature=0, max_tokens=200)
        return response

    # VALIDATE AND REFINE USER PROMPT
    with st.spinner('Validating and refining user prompt ...'):
        cache = 0.0 if streamlit_cache else random.random()
        try:
            response = refine_user_prompt(user_prompt, ss.prompt_moderation_deployment, system_message, cache)
            prompt_filter = response.prompt_filter_results[0]['content_filter_results']
            prompt_filter_markdown = display_moderation_results(prompt_filter)
            st.markdown(":blue[**Prompt ok**: " + prompt_filter_markdown + "]", unsafe_allow_html=True)
            refined_prompt = response.choices[0].message.content
        except Exception as e:
            if hasattr(e, 'code') and e.code == 'content_filter':
                prompt_filter = e.body['innererror']['content_filter_result']
                prompt_filter_markdown = display_moderation_results(prompt_filter)
                st.markdown("**Prompt filtered**: " + prompt_filter_markdown, unsafe_allow_html=True)
                st.markdown("**Please review our guidelines and try again.**")
            else:
                st.write(f"Exception occurred:\n{e}")
            refined_prompt = None
            st.stop()

    st.write(refined_prompt)

    # GENERATE IMAGE
    def find_max_id(image_dir):
        """Finds the maximum ID of images in the directory."""
        max_id = 0
        pattern = re.compile(r'image_(\d{4})\.png')
        for filename in os.listdir(image_dir):
            match = pattern.match(filename)
            if match:
                max_id = max(max_id, int(match.group(1)))
        return max_id

    image_dir = os.path.join(os.curdir, 'images')
    os.makedirs(image_dir, exist_ok=True)
    current_image_path = os.path.join(image_dir, 'generated_image.png')

    @st.cache_data(show_spinner=False)
    def create_image(prompt, cache=0.0):
        """Creates an image based on the given prompt and deployment settings."""
        generated_image, caption = None, prompt  # Default caption is the prompt itself
        
        if imgen_deployment == "FLUX.1 [pro]":
            input_data = {
                "prompt": prompt,
                "aspect_ratio": flux_pro_aspect_ratio,
                "safety_tolerance": 5,
                "steps": flux_pro_n_steps,
                "Guidance": flux_pro_guidance,
                "Interval": flux_pro_interval,
            }
            api = replicate.Client(api_token=ss.replicate_api_key)
            output = api.run("black-forest-labs/flux-pro", input=input_data)
            image_url = output
            generated_image = requests.get(image_url).content
        
        elif imgen_deployment == "DALL E-3":
            quality = 'hd' if dalle_quality else 'standard'
            style = 'vivid' if dalle_style else 'natural'
            try:
                result = ss.aoai_client.images.generate(
                    model=ss.dalle_deployment,
                    prompt=prompt,
                    n=1,
                    quality=quality,
                    size=dalle_size,
                    style=style
                )
            except Exception as e:
                st.write(e)

            json_response = json.loads(result.model_dump_json())
            dalle_revised_prompt = json_response.get('data', [{}])[0].get('revised_prompt', None)
            caption = dalle_revised_prompt

            image_url = json_response["data"][0]["url"]
            generated_image = requests.get(image_url).content

        elif imgen_deployment == "Stable Diffusion XL":
            input_data = {
                'prompt': prompt,
                'negative_prompt': negative_prompt if sdxl_use_negative_prompt else None,
                'width': sdxl_width,
                'height': sdxl_height, 
                'n_steps': sdxl_steps,
                'high_noise_frac': 0.7,
                'seed': sdxl_seed
            }
            response = create_sdxl_image(input_data, ss.aml_imgen_online_endpoint_url, ss.aml_imgen_api_key, ss.aml_deployment_name)
            encoded_image = response.json()[0]
            generated_image = base64.b64decode(encoded_image)

        elif imgen_deployment == "Stable Diffusion 3":
            response = requests.post(
                f"https://api.stability.ai/v2beta/stable-image/generate/sd3",
                headers={"authorization": f"Bearer {ss.stability_api_key}", "accept": "image/*"},
                data={"prompt": prompt, "model": sd3_model, "aspect_ratio": sd3_aspect_ratio, "output_format": "png"},
            )
            if response.status_code == 200:
                generated_image = response.content
            else:
                raise Exception(str(response.json()))

        if generated_image:
            with open(current_image_path, "wb") as image_file:
                image_file.write(generated_image)
            
            if save_images:
                max_id = find_max_id(image_dir)
                save_image_path = os.path.join(image_dir, f'image_{max_id + 1:04}.png')
                with open(save_image_path, "wb") as image_file:
                    image_file.write(generated_image)

        return generated_image, caption

    cache = 0.0 if streamlit_cache else random.random()
    image, caption = create_image(refined_prompt, cache)

    st.image(image, caption=caption)

    # Download Image and Generate Video Options
    download_col, img2video_col = st.columns(2)
    with open(current_image_path, "rb") as file:
        download_col.download_button(label="Download image", data=file, file_name="generated_image.png", mime="image/png", use_container_width=True)
        
        if "Stable Diffusion 3" in ss.imgen_models:
            img2video_col.button(label="Generate video clip", key="img2video", on_click=img2video, args=(image_dir, 'generated_image.png',), use_container_width=True)

    # ANALYZE IMAGE FOR BRANDS
    with st.spinner('Check image for brands ...'):
        if azure_image_analysis:
            start_time = time.time()
            response_json = azure_image_analysis_predict(current_image_path, ss.vision_model_name, ss.vision_endpoint, ss.vision_key)
            duration = time.time() - start_time
            brands = set(object['tags'][0]['name'] for object in response_json['customModelResult']['objectsResult']['values'] if object['tags'][0]['confidence'] > threshold)
            brands_string = "Azure AI Vision Analysis: " + ("Found " + ", ".join(brands) if brands else "No brands found")
            st.write(f":blue[{brands_string} ({duration:.1f} s)]")

        if llm_image_analysis:
            start_time = time.time()
            response = analyze_image_gpt4o(
                image_path=current_image_path,
                system_message=gpt4o_system_message,
                user_prompt=gpt4o_user_prompt,
                api_key=ss.aoai_key,
                aoai_endpoint=ss.aoai_endpoint,
                aoai_deployment=ss.gpt_deployment,
                seed=0,
                api='aoai'
            )
            detected_brands = response.json()['choices'][0]['message']['content']
            duration = time.time() - start_time
            st.write(f":blue[{detected_brands} ({duration:.1f} s)]")

    # ANALYZE IMAGE FOR HARMFUL CONTENT
    with st.spinner('Check image for harmful content ...'):
        analyzer = VideoAnalyzer(ss.aoai_client, ss.gpt_deployment, ss.content_safety_endpoint, ss.content_safety_key)
        content_safety_image_path = check_and_reduce_image_size(current_image_path)

        with open(content_safety_image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        cf_results = analyzer.content_safety_moderate_image(base64_image)
        
        severity_thresholds = {'Hate': hate_thresh, 'SelfHarm': selfharm_thresh, 'Sexual': sexual_thresh, 'Violence': violence_thresh}
        
        cs_results_dict = {}
        for item in cf_results['categoriesAnalysis']:
            category = item['category']
            severity = item['severity']
            cs_results_dict[category] = {
                'filtered': severity >= severity_thresholds[category],
                'severity': ss.id_to_severity[severity]
            }
        
        cs_results_markdown = display_moderation_results(cs_results_dict)
        st.markdown("**Image content safety**: " + cs_results_markdown, unsafe_allow_html=True)
