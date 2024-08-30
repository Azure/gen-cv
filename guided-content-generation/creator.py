import os
from openai import AzureOpenAI
import streamlit as st
from streamlit import session_state as ss
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Initialize session state for credentials
if "credentials" not in ss:
    ss.credentials = True

    # Set Azure OpenAI and other service credentials
    ss.aoai_endpoint = os.getenv('AOAI_ENDPOINT')
    ss.aoai_key = os.getenv('AOAI_KEY')
    ss.gpt_deployment = os.getenv('GPT_DEPLOYMENT')
    ss.prompt_moderation_deployment = os.getenv('PROMPT_MODERATION_DEPLOYMENT')
    ss.dalle_deployment = os.getenv('DALLE_DEPLOYMENT')

    # Azure OpenAI Whisper service (using Sweden Central)
    ss.aoai_endpoint_swece = os.getenv('AOAI_ENDPOINT_SWECE')
    ss.aoai_key_swece = os.getenv('AOAI_KEY_SWECE')
    ss.whisper_deployment = os.getenv('WHISPER_DEPLOYMENT')

    # Content Safety service
    ss.content_safety_endpoint = os.getenv('CONTENT_SAFETY_ENDPOINT')
    ss.content_safety_key = os.getenv('CONTENT_SAFETY_KEY')

    # Optional: Stability AI and Replicate API keys for image and video generation
    ss.stability_api_key = os.getenv('STABILITY_API_KEY')
    ss.replicate_api_key = os.getenv('REPLICATE_API_KEY')

    # Optional: Azure ML Online Endpoint for SDXL image generation
    ss.aml_imgen_api_key = os.getenv('AML_IMGEN_API_KEY')
    ss.aml_imgen_online_endpoint_url = os.getenv('AML_IMGEN_ONLINE_ENDPOINT_URL')
    ss.aml_deployment_name = os.getenv('AML_DEPLOYMENT_NAME')

    # Optional: Custom Azure AI Vision Brand Detection Model
    ss.azure_ai_vision_endpoint = os.getenv('AZURE_AI_VISION_ENDPOINT')
    ss.azure_ai_vision_key = os.getenv('AZURE_AI_VISION_KEY')
    ss.azure_ai_vision_deployment = os.getenv('AZURE_AI_VISION_DEPLOYMENT')

    ss.azure_ai_vision = bool(ss.azure_ai_vision_key)

    # Initialize Azure OpenAI clients
    ss.aoai_client = AzureOpenAI(
        api_version="2024-05-01-preview",
        api_key=ss.aoai_key,
        azure_endpoint=ss.aoai_endpoint
    )
    ss.aoai_client_swece = AzureOpenAI(
        api_version="2024-05-01-preview",
        api_key=ss.aoai_key_swece,
        azure_endpoint=ss.aoai_endpoint_swece
    )

    # Build list of available image generation models based on API keys
    model_mapping = {
        "dalle_deployment": "DALL E-3",
        "stability_api_key": "Stable Diffusion 3",
        "replicate_api_key": "FLUX.1 [pro]",
        "aml_imgen_api_key": "Stable Diffusion XL",
    }
    ss.imgen_models = [model for key, model in model_mapping.items() if getattr(ss, key, "")]

    # Mapping for severity levels
    ss.severity_to_id = {"safe": 0, "low": 2, "med": 4, "high": 6}
    ss.id_to_severity = {0: "safe", 2: "low", 4: "med", 6: "high"}

    # Custom CSS for Streamlit page elements
    ss.custom_css = """
    <style>
    [data-testid="stTickBarMin"],
    [data-testid="stTickBarMax"] {
        font-size: 0px;
    }
    </style>
    """

# Page setup using Streamlit's navigation
pg = st.navigation([
    st.Page("image_gen.py", title="Image Generation", icon=":material/palette:", default=True),
    st.Page("video.py", title="Video Analysis", icon=":material/play_circle:"),
    st.Page("gallery.py", title='Gallery', icon=":material/photo_library:"),
    st.Page("organize.py", title='Organizer', icon=":material/folder_managed:"),
   # st.Page("debug.py", title="Debug", icon=":material/code:")
])
pg.run()
