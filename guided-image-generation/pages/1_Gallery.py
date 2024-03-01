import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import streamlit as st
import os
from PIL import Image, ImageFilter
from dotenv import load_dotenv
from utils import analyze_image_gpt4v, check_and_reduce_image_size, analyze_content_safety, display_moderation_results
import time 
import json

# Read configuration file
try:
    config_file = open('config-journey.json', 'r')
    config = json.load(config_file)
    # Set system prompts
    gpt4v_system_message = config['GPT4V_SYSTEM_MESSAGE']
    gpt4v_user_prompt = config['GPT4V_USER_PROMPT']
    
except Exception as e:
        st.write(e)

st.set_page_config(
    page_title="Image Generator",
    layout="wide",
    initial_sidebar_state="auto",)

st.title("Image Gallery")
st.write ("Select an image for analysis")

load_dotenv()

aoai_key = os.environ["AOAI_KEY"]
api_base = os.environ['AOAI_ENDPOINT']
aoai_gpt_4_vision=os.environ['AOAI_GPT_4_VISION']

# Azure Content Safety
cosa_endpoint=os.environ['ACS_ENDPOINT']
cosa_key=os.environ['ACS_KEY']

def list_folders_with_entry(entry_folder):
    folders = [entry_folder]
    for root, dirs, _ in os.walk(entry_folder):
        for dir in dirs:
            folders.append(os.path.join(root, dir))
    return folders

folders = list_folders_with_entry('images')

def apply_blur(image):
    return image.filter(ImageFilter.GaussianBlur(radius=5))


with st.sidebar:
    gallery_folder = st.selectbox("Image folder:", folders, index=0)
    blur = st.toggle('Blur images', False)
    st.write('Unwanted element detection model')
    gpt4v = st.toggle('GPT-4 Vision', True)
    azure_image_analysis = False #st.toggle('Azure Image Analysis', True)
    if azure_image_analysis:
        threshold = st.slider('Azure Image Analysis Threshold', 0.0, 1.0, 0.6, 0.1, help='Lower values increase the likelihood of detecting objects but also increases false positives.')
    
    cs_severity2level = {'safe' : 0, 'low' : 2, 'medium' : 4, 'high' : 6}
    cs_threshold = cs_severity2level[st.select_slider("Image Moderation Threshold", cs_severity2level.keys(), 'low', help='Images with the selected severity level and higher will be rejected. ')]
      
    save_images = st.toggle("Save images", True)

grid, detail = st.columns(2)

def check_unwanted_words(image_path):
    with detail:
        image = Image.open(image_path)
        if blur:
            image = apply_blur(image)
        # st.image(image)
               
        # st.image(image_path)
        with st.spinner('Check image for unwanted elements...'):
            if gpt4v:
                if not azure_image_analysis: # display image only if not already
                    image = Image.open(image_path)
                    if blur:
                        image = apply_blur(image)
                    st.image(image) 
                start_time = time.time()
                response = analyze_image_gpt4v(image_path=image_path,
                        system_message=gpt4v_system_message,
                        user_prompt=gpt4v_user_prompt,
                        api_key=aoai_key,
                        seed=0,
                        aoai_endpoint=api_base,
                        aoai_deployment=aoai_gpt_4_vision)   
                detected_brands = response.json()['choices'][0]['message']['content']
                duration = time.time() - start_time
                
                st.write(f"{detected_brands} ({duration:.1f} s)")

        with st.spinner('Check image for harmful content ...'):
            content_safety_image_path = check_and_reduce_image_size(image_path) # reduces size as needed for Azure Content Safety
            result = analyze_content_safety(content_safety_image_path, endpoint=cosa_endpoint, key=cosa_key)

            cs_level2severity = {0 : 'safe', 2 : 'low', 4 : 'medium', 6 : 'high'}

            cs_results_dict = {}
            for item in result.categories_analysis:
                cs_results_dict[item['category']] = {
                    'filtered' : False if item['severity'] < cs_threshold else True,
                    'severity' : cs_level2severity[item['severity']]}
                
            cs_results_markdown = display_moderation_results(cs_results_dict)
            st.markdown("**Image content safety**: "+ cs_results_markdown, unsafe_allow_html=True)


supported_extensions = ['.png', '.jpg', '.jpeg']

# Filter the image files based on the supported extensions
image_files = [f for f in os.listdir(gallery_folder) 
               if os.path.isfile(os.path.join(gallery_folder, f)) 
               and any(f.lower().endswith(ext) for ext in supported_extensions)]

with grid:

    num_columns = 2

    # Iterate over images and place them in columns
    for i in range(0, len(image_files), num_columns):
        columns = st.columns(num_columns)
        for col, image_file in zip(columns, image_files[i:i+num_columns]):
            image_path = os.path.join(gallery_folder, image_file)
            image = Image.open(image_path)

            # Resize image for thumbnail
            image.thumbnail((300, 300))

            if blur:
                image = apply_blur(image)
        
            # Display image with a button in a column
            with col:
                st.image(image)
                if st.button(label=f"Analyze", key=image_file):
                    check_unwanted_words(image_path)
