import os
import base64
import time
import streamlit as st
from streamlit import session_state as ss
from PIL import Image, ImageFilter

from VideoTools import VideoAnalyzer
from instructions import gpt4o_system_message, gpt4o_user_prompt
from utils import (
    analyze_image_gpt4o,
    azure_image_analysis_predict,
    azure_image_analysis_create_image,
    check_and_reduce_image_size,
    display_moderation_results,
)

# Streamlit configuration
st.set_page_config(
    layout="wide",
    initial_sidebar_state="auto",
)

st.title("Image Gallery")
st.write("Select an image for analysis")

def list_folders_with_entry(entry_folder):
    """List all folders within the specified directory."""
    folders = [entry_folder]
    for root, dirs, _ in os.walk(entry_folder):
        for dir in dirs:
            folders.append(os.path.join(root, dir))
    return folders

def apply_blur(image):
    """Apply Gaussian blur to an image."""
    return image.filter(ImageFilter.GaussianBlur(radius=5))

folders = list_folders_with_entry('images')

with st.sidebar:
    gallery_folder = st.selectbox("Image folder:", folders, index=0)
    blur = st.checkbox('Blur images', False)

    st.write('Brand detection models:')
    col1, col2 = st.columns(2)
    llm_image_analysis = col1.checkbox('GPT-4o', True)

    azure_image_analysis = col2.checkbox(
        'AI Vision', False, 
        disabled=not ss.azure_ai_vision, 
        help='Use a custom object detection model trained on your data.'
    )

    if azure_image_analysis:
        threshold = st.slider(
            'Azure Image Analysis Threshold', 0.0, 1.0, 0.6, 0.1,
            help='Lower values increase the likelihood of detecting objects but also increase false positives.'
        )

    help_str = (
        "The selected severity is the lowest level that is considered harmful. "
        "Example: selecting 'medium' means that all content identified as medium or higher would be flagged."
    )
    st.markdown("Image moderation thresholds:", help=help_str)
    col1, col2 = st.columns(2)
    st.markdown(ss.custom_css, unsafe_allow_html=True)
    
    hate_thresh = ss.severity_to_id[col1.select_slider("Hate", ss.severity_to_id.keys(), 'low')]
    selfharm_thresh = ss.severity_to_id[col2.select_slider("SelfHarm", ss.severity_to_id.keys(), 'low')]
    sexual_thresh = ss.severity_to_id[col1.select_slider("Sexual", ss.severity_to_id.keys(), 'low')]
    violence_thresh = ss.severity_to_id[col2.select_slider("Violence", ss.severity_to_id.keys(), 'low')]

grid, detail = st.columns(2)

def check_brands(image_path):
    """Check image for brands using GPT-4o or Azure Image Analysis."""
    with detail:
        image = Image.open(image_path)
        with st.spinner('Check image for brands ...'):
            if azure_image_analysis:
                start_time = time.time()
                response_json = azure_image_analysis_predict(
                    image_path, ss.azure_ai_vision_deployment, ss.azure_ai_vision_endpoint, ss.azure_ai_vision_key
                )
                duration = time.time() - start_time

                if response_json['customModelResult']['objectsResult']['values']:
                    objects = response_json['customModelResult']['objectsResult']['values']
                    brands = {obj['tags'][0]['name'] for obj in objects if obj['tags'][0]['confidence'] > threshold}

                annotated_image = azure_image_analysis_create_image(image_path, response_json, threshold, 9)

                if blur:
                    image = apply_blur(image)
                    st.image(image)
                else:
                    st.pyplot(fig=annotated_image)

                brands_string = "AI Vision Analysis: Found " + ", ".join(brands) if brands else "AI Vision Analysis: No brands found"
                st.write(f"{brands_string} ({duration:.1f} s)")

            if llm_image_analysis:
                if not azure_image_analysis:
                    if blur:
                        image = apply_blur(image)
                    st.image(image)

                start_time = time.time()
                response = analyze_image_gpt4o(
                    image_path=image_path,
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
                st.write(f"{detected_brands} ({duration:.1f} s)")

        with st.spinner('Check image for harmful content ...'):
            analyzer = VideoAnalyzer(ss.aoai_client, ss.gpt_deployment, ss.content_safety_endpoint, ss.content_safety_key)
            content_safety_image_path = check_and_reduce_image_size(image_path)

            with open(content_safety_image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            cf_results = analyzer.content_safety_moderate_image(base64_image)

            severity_thresholds = {
                'Hate': hate_thresh, 
                'SelfHarm': selfharm_thresh, 
                'Sexual': sexual_thresh, 
                'Violence': violence_thresh
            }

            cs_results_dict = {
                item['category']: {
                    'filtered': item['severity'] >= severity_thresholds[item['category']],
                    'severity': ss.id_to_severity[item['severity']]
                }
                for item in cf_results['categoriesAnalysis']
            }
            
            cs_results_markdown = display_moderation_results(cs_results_dict)
            st.markdown("**Image content safety**: " + cs_results_markdown, unsafe_allow_html=True)

supported_extensions = ['.png', '.jpg', '.jpeg']

# Filter the image files based on the supported extensions
image_files = [
    f for f in os.listdir(gallery_folder) 
    if os.path.isfile(os.path.join(gallery_folder, f)) and 
    any(f.lower().endswith(ext) for ext in supported_extensions)
]

with grid:
    num_columns = 2

    # Iterate over images and place them in columns
    for i in range(0, len(image_files), num_columns):
        columns = st.columns(num_columns)
        for col, image_file in zip(columns, image_files[i:i + num_columns]):
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
                    check_brands(image_path)
