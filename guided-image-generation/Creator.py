# Import libraries
from openai import AzureOpenAI
import streamlit as st
import os
from dotenv import load_dotenv
import json
import requests
import re
from PIL import Image
import base64
import time
import random

# Import utils library
from utils import analyze_image_gpt4v, create_sdxl_image, display_moderation_results, check_and_reduce_image_size, analyze_content_safety

# Set up initial Streamli parameters from .env file
st.set_page_config(
    page_title="Image Generator",
    layout="centered",
    initial_sidebar_state="auto",
    )

# Load configuration data from .env file
load_dotenv()

# Azure OpenAI apikey and endpoint
aoai_key = os.environ["AOAI_KEY"]
aoai_endpoint = os.environ['AOAI_ENDPOINT']
aoai_gpt_4=os.environ['AOAI_GPT_4']
aoai_gpt_35=os.environ['AOAI_GPT_35']
aoai_gpt_4_vision=os.environ['AOAI_GPT_4_VISION']
aoai_dall_e=os.environ['AOAI_DALL_E']

# Azure Content Safety
acs_endpoint=os.environ['ACS_ENDPOINT']
acs_key=os.environ['ACS_KEY']

# Create AOAI client
client = AzureOpenAI(
    api_version="2023-12-01-preview",  
    api_key=aoai_key,  
    azure_endpoint=aoai_endpoint)

# Read configuration file
try:
    config_file = open('config-journey.json', 'r')
    config = json.load(config_file)
    # Set system prompts
    basic_system_message = config['BASIC_SYSTEM_MESSAGE']
    unwanted_elems_system_message = config['UNWANTED_ELEMS_SYSTEM_MESSAGE']
    gpt4v_system_message = config['GPT4V_SYSTEM_MESSAGE']
    gpt4v_user_prompt = config['GPT4V_USER_PROMPT']
    with_stable_diffusion = config['STABLE_DIFFUSION']
    title = config['TITLE']
    styles = config['STYLES']
    with_text_generation = config['TEXT_GENERATION']
    text_generation_languages = config['TEXT_GENERATION_LANGUAGES']
    text_generation_prompt = config['TEXT_GENERATION_PROMPT']
    
except Exception as e:
        st.write(e)

if with_stable_diffusion == "True":
    # Stable Diffusion Azure ML Online apikey, endpoint and deployment name
    aml_imgen_api_key = os.environ['AML_IMGEN_API_KEY']
    aml_imgen_url = os.environ['AML_IMGEN_URL']
    aml_imgen_deployment_name = os.environ['AML_IMGEN_NAME']

    print(aml_imgen_url)

# Set the MS logo
st.image("microsoft.png", width=100)

# Set title
st.title(title)

ss = st.session_state

# List of GPT models for the listbox
deployments_text = {'AOAI GPT-4-turbo' : aoai_gpt_4, 
                    'AOAI GPT-3.5-turbo-16k' : aoai_gpt_35}

# List of image generation models for the listbox
if with_stable_diffusion == "True":
    deployments_image = {'DALL E-3' : aoai_dall_e,
                        'Stable Diffusion' : aml_imgen_deployment_name}
else: # DALL E-3 only
    deployments_image = {'DALL E-3' : aoai_dall_e}


# List of Dall-e sizes
sizes = ['1024x768', '1024x1024', '1792x1024', '1024x1792']

with st.sidebar:
    # Set up the listbox of GPT models
    llm_deployment = st.selectbox("User input refinemenet model:", 
                                  list(deployments_text.keys()), 
                                  index=0, 
                                  help="Azure OpenAI deployment for analyzing and refining the user input")

    # Set up the checkbox to remove unwanted terms or not    
    filter_unwanted_terms = st.checkbox('Filter unwanted elements in prompt', 
                                        True, 
                                        help="Remove unwanted elements from initial user prompt.")
    
    # Set up the listbox of image generation models
    if with_stable_diffusion == "True":
        imgen_deployment = st.selectbox("Vision generation model:",
                                        list(deployments_image.keys()),
                                        index=0, 
                                        help="Image generation model")
    else:
        imgen_deployment = 'DALL E-3'
        st.write('Dall E-3 Quality and Style:')

    col1, col2 = st.columns(2)
    # If selected, set up Dall-E options (HD and Vivid)
    if imgen_deployment == 'DALL E-3':
        dalle_quality = col1.toggle("HD", True, help="Quality can be 'HD' or 'Standard'")
        dalle_style = col2.toggle("Vivid", True, help="Style can be 'Vivid' or 'Natural'")
    elif imgen_deployment == 'Stable Diffusion': # If selected, set up Stable Diffusion options (number of steps and Guidance scale)
        n_steps = int(col1.text_input('Steps', value='50'))
        guidance_scale = float(col2.text_input('Guidance scale', value='7.5'))

    # Set up image resolution width and height
    image_size = st.selectbox("Image size:", sizes, index=1, help="Image resolution width x height")
    imgen_style = st.selectbox("Visual style:", styles, index=0,)

    # Set up the checkbox to use or not GPT-4-Vision to detect unwanted elements in the image
    st.write('Unwanted element detection:')
    bd_col1, bd_col2 = st.columns(2)
    gpt4v = bd_col1.toggle('GPT4-Vision', True)

    # Severity level of image moderation    
    cs_severity2level = {'safe' : 0, 'low' : 2, 'medium' : 4, 'high' : 6}
    cs_threshold = cs_severity2level[st.select_slider("Image Moderation Threshold", cs_severity2level.keys(), 'low', help='Images with the selected severity level and higher will be rejected.')]

    if with_text_generation == "True":
        # Set up the checkbox to generate the text generation languages
        text_generation_language = st.selectbox("Language of the text generated:", 
                                    text_generation_languages, 
                                    index=0, 
                                    help="Generate the text in this language")

    # Set up options to Save the image or not and to use Cache or not
    save, cache = st.columns(2)      
    save_images = save.toggle("Save image", True)
    streamlit_cache =cache.toggle("Cache ", True) 

    st.divider()

    # Button to clean and restart
    if st.button("Clear cache and restart", use_container_width=True, type='primary'):
        st.session_state.clear()
        st.cache_data.clear()
        st.rerun()

# Textbox for the user's input
user_prompt = st.chat_input("Describe your image")

# Reserve the space for the text area of the system prompt
# system_prompt_place = st.empty()

# Set the system prompt depending if GPT4 has to filter or not unwanted terms
if filter_unwanted_terms:
    system_message_ini = unwanted_elems_system_message.format(style=imgen_style, model=imgen_deployment)
else:
    system_message_ini = basic_system_message.format(style=imgen_style, model=imgen_deployment)

# Set value of text area for system prompt
# with system_prompt_place:
#     system_message = st.text_area("System prompt", value=system_message_ini, height=155)

system_message = system_message_ini

if user_prompt:

    # Build image generation prompt based on user input

    def refine_user_prompt(prompt, model, system_message, cache=0.0):
        messages = [{'role' : 'system', 'content' : system_message},
                    {'role' : 'user', 'content' : prompt}]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0, #0.9, #0,
            max_tokens=200 #350, #200,
        )
        
        return response

    # VALIDATE AND REFINE USER PROMPT

    cache = 0.0 if streamlit_cache else random.random() # insert random number into chached functions to avoid using cache 
    try:
        with st.spinner('Refining Prompt ...'):
            ini = time.time()

            response = refine_user_prompt(user_prompt, deployments_text[llm_deployment], system_message, cache)

            end = time.time()
            print(f'\nuser_prompt: [{user_prompt}] ---------------------------------------------')
            print(f'{deployments_text[llm_deployment]} takes {round(end - ini,2)} seconds.')

            prompt_filter = response.prompt_filter_results[0]['content_filter_results']
            prompt_filter_markdown = display_moderation_results(prompt_filter)
            st.markdown("**Prompt ok**: "+prompt_filter_markdown, unsafe_allow_html=True)
            refined_prompt = response.choices[0].message.content  #.replace('\n', '.')

    except Exception as e:

        if e.code == 'content_filter':
            prompt_filter = e.body['innererror']['content_filter_result']
            prompt_filter_markdown = display_moderation_results(prompt_filter)
            st.markdown("**Prompt filtered**: "+prompt_filter_markdown, unsafe_allow_html=True)
            st.markdown("**Please review our guidelines and try again.**")
        else:
            st.write (f"Exception occured:\n{e}")
        refined_prompt = None
        st.stop()

    # st.write(refined_prompt)

    # GENERATE IMAGE:

    # Get the number of images in the directory to calculate the next number for the filename
    def find_max_id(image_dir):
        max_id = 0
        pattern = re.compile(r'image_(\d{4})\.png')
        for filename in os.listdir(image_dir):
            match = pattern.match(filename)
            if match:
                max_id = max(max_id, int(match.group(1)))

        return max_id
    
    # Prepare the filename
    image_dir = os.path.join(os.curdir, 'images')
    if not os.path.isdir(image_dir):
        os.mkdir(image_dir)
    current_image_path = os.path.join(image_dir, 'generated_image.png')

    with st.spinner('Creating image ...'):

        # Create the image with the selected model

        @st.cache_data(show_spinner=False)
        def create_image(prompt, cache=0.0):

            # Create the model with Dall-E    
            if imgen_deployment == "DALL E-3":
                quality = 'hd' if dalle_quality else 'standard'
                style = 'vivid' if dalle_style else 'natural'

                try:
                    print(f'model: {deployments_image[imgen_deployment]}')
                    print(f'prompt: [{prompt}]')
                    print(f'quality: [{quality}]')
                    print(f'size: [{image_size}]')
                    print(f'style: [{style}]')

                    ini = time.time()
                    result = client.images.generate(
                        model=deployments_image[imgen_deployment],
                        prompt=prompt,
                        n=1,
                        quality=quality,
                        size=image_size,
                        style=style)
                    
                    end = time.time()
                    print(f'Dall-E 3 takes {round(end - ini, 2)} seconds.')
                    st.write(f'Dall-E 3 takes {round(end - ini,  2)} seconds.')

                except Exception as e:
                    st.write(e)
                
                json_response = json.loads(result.model_dump_json())
                dalle_revised_prompt = json_response.get('data', [{}])[0].get('revised_prompt', None)
                caption = dalle_revised_prompt

                # Retrieve the generated image
                image_url = json_response["data"][0]["url"]  # extract image URL from response
                generated_image = requests.get(image_url).content  # download the image
                with open(current_image_path, "wb") as image_file:
                    image_file.write(generated_image)

            # Create the model with Stable Diffusion
            elif imgen_deployment == "Stable Diffusion":
               
                negative_prompt = """
                long neck, out of frame, extra fingers, mutated hands, monochrome, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, glitchy, bokeh, (((long neck))), ((flat chested)), ((((visible hand)))), ((((ugly)))), (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))) red eyes, multiple subjects, extra heads
                """

                width = int(image_size.split('x')[0])
                height = int(image_size.split('x')[1])

                seed = None

                input = {'prompt' : prompt,
                         'negative_prompt' : negative_prompt,
                         'width' : width,
                         'height' : height,
                         'n_steps' : n_steps,
                         'high_noise_frac' : 0.7,
                         'guidance_scale' : guidance_scale,
                        # 'seed' : seed
                        } 

                print("\nINPUT SD: ", input)

                try:
                    ini = time.time()
                    
                    print('sdxl deployment name: ', deployments_image[imgen_deployment])

                    response = create_sdxl_image(input, aml_imgen_url, aml_imgen_api_key, deployments_image[imgen_deployment])
                    
                    end = time.time()
                    print(f'Stable Diffusion takes {round(end - ini, 2)} seconds.')
                    st.write(f'Stable Diffusion takes {round(end - ini, 2)} seconds.')

                    # Get the image
                    encoded_image = response.json()[0]# ['generated_image']

                    # Decode the base64 string to an image
                    generated_image = base64.b64decode(encoded_image)

                    # save for later steps
                    with open(current_image_path, "wb") as image_file:
                        image_file.write(generated_image)

                    caption = prompt
                
                except Exception as e:
                    st.write(e)

            # optionally save image with appended id
            if save_images:
                max_id = find_max_id(image_dir)
                new_id = max_id + 1
                save_image_path = os.path.join(image_dir, f'image_{new_id:04}.png')
                #print('FILENAME: ' + save_image_path)
                with open(save_image_path, "wb") as image_file:
                    image_file.write(generated_image)
            
            return generated_image, caption

        # insert random number into chached functions to avoid using cache
        cache = 0.0 if streamlit_cache else random.random() 
        
        # Set up the generated image
        image, caption = create_image(refined_prompt, cache)
        st.image(image, caption=caption) # caption=user_prompt)
        
        # Download the image
        with open(current_image_path, "rb") as file:
            btn = st.download_button(label="Download image", data=file, file_name="generated_image.png", mime="image/png")

    # ANALYZE IMAGE FOR UNWANTED ELEMENTS 

    with st.spinner('Check image for unwanted elements...'):
        if gpt4v:
            try:
                start_time = time.time()
                response = analyze_image_gpt4v(image_path=current_image_path,
                        system_message=gpt4v_system_message,
                        user_prompt=gpt4v_user_prompt,
                        api_key=aoai_key,
                        seed=0,
                        aoai_endpoint=aoai_endpoint,
                        aoai_deployment=aoai_gpt_4_vision)
                detected_unwanted_elems = response.json()['choices'][0]['message']['content']
                duration = time.time() - start_time
                
                st.write(f":blue[{detected_unwanted_elems} ({duration:.1f} s)]")

            except Exception as e:
                st.write(e)

    # ANALYZE IMAGE FOR HARMFUL CONTENT
    
    with st.spinner('Check image for harmful content ...'):
        try:
            content_safety_image_path = check_and_reduce_image_size(current_image_path) # reduces size as needed for Azure Content Safety
            result = analyze_content_safety(content_safety_image_path, endpoint=acs_endpoint, key=acs_key)

            cs_level2severity = {0 : 'safe', 2 : 'low', 4 : 'medium', 6 : 'high'}

            cs_results_dict = {}
            for item in result.categories_analysis:
                cs_results_dict[item['category']] = {
                    'filtered' : False if item['severity'] < cs_threshold else True,
                    'severity' : cs_level2severity[item['severity']]}
            
            cs_results_markdown = display_moderation_results(cs_results_dict)
            st.markdown("**Image content safety**: "+ cs_results_markdown, unsafe_allow_html=True)
        
        except Exception as e:
            st.write(e)
    
    # CREATE A TEXT
    
    if with_text_generation == "True":
        with st.spinner('Creating a text...'):
            def create_text(prompt, model):
                # Create the text in the selected language
                print('text_prompt: [' + text_generation_prompt + ']')
                final_text_generation_prompt = text_generation_prompt.format(text_generation_language=text_generation_language)
                print('FINAL Text prompt: ' + final_text_generation_prompt)

                messages = [{'role' : 'system', 'content' : final_text_generation_prompt},
                            {'role' : 'user', 'content' : prompt}]

                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7, #0.9, 
                    max_tokens=200, #300
                )
                
                return response

            # insert random number into chached functions to avoid using cache
            cache = 0.0 if streamlit_cache else random.random() 
            try:
                response = create_text(refined_prompt, deployments_text[llm_deployment])
                text_generated = response.choices[0].message.content
                st.markdown("**A TEXT ABOUT YOUR INPUT:** ", unsafe_allow_html=True)
                st.write(text_generated.replace("\n", "<BR>"), unsafe_allow_html=True)

            except Exception as e:
                st.write (f"Exception occured:\n{e}")
                st.stop()
