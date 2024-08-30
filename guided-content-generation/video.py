import os
import time
import streamlit as st
from streamlit import session_state as ss

from VideoTools import VideoExtractor, VideoAnalyzer
from utils import dict_to_markdown_table

# Set up the Streamlit page configuration
st.set_page_config(layout="wide", initial_sidebar_state="auto")

# Initialize the VideoAnalyzer
video_analyzer = VideoAnalyzer(
    ss.aoai_client, ss.gpt_deployment, ss.content_safety_endpoint, ss.content_safety_key
)

st.title('Video Analysis')

# Sidebar for user input
with st.sidebar:
    # Select video subfolder and file
    video_entry_folder = './videos'
    video_subfolders = [
        f for f in os.listdir(video_entry_folder) if os.path.isdir(os.path.join(video_entry_folder, f))
    ]
    video_subfolder = st.selectbox('Video folder', video_subfolders)
    video_subfolder_path = os.path.join(video_entry_folder, video_subfolder)
    video_names = [
        f for f in os.listdir(video_subfolder_path) if f.lower().endswith('.mp4')
    ]

    with st.form('my_form', clear_on_submit=False, border=False):
        video_filename = st.selectbox('Video', video_names)
        video_path = os.path.join(video_subfolder_path, video_filename)

        # Frame sampling settings
        frames_per_scene = st.slider("Frames per scene", min_value=1, max_value=10, value=2)
        drop_similar_frames = st.toggle("Drop similar frames", value=True)
        help_str = (
            "Hash difference threshold to consider frames as similar. "
            "Increase threshold to reduce number of frames."
        )
        frame_similarity_threshold = st.slider(
            "Similarity threshold", min_value=1, max_value=30, value=20, 
            help=help_str, disabled=not drop_similar_frames
        )

        transcribe_audio = st.toggle("Transcribe audio", value=True)
    
        # Azure Content Safety settings
        help_str = (
            "The selected severity is the lowest level that is considered harmful. "
            "Example: selecting 'medium' means that all content identified as medium or higher would be flagged."
        )
        st.markdown("Video moderation thresholds:", help=help_str)
        col1, col2 = st.columns(2)
        st.markdown(ss.custom_css, unsafe_allow_html=True)
        hate_thresh = ss.severity_to_id[col1.select_slider("Hate", ss.severity_to_id.keys(), 'low')]
        selfharm_thresh = ss.severity_to_id[col2.select_slider("SelfHarm", ss.severity_to_id.keys(), 'low')]
        sexual_thresh = ss.severity_to_id[col1.select_slider("Sexual", ss.severity_to_id.keys(), 'low')]
        violence_thresh = ss.severity_to_id[col2.select_slider("Violence", ss.severity_to_id.keys(), 'low')]

        st.divider()
        submit_button = st.form_submit_button(label='Analyze video', use_container_width=True, type='primary')

# Video App Flow using Streamlit caching
col1, col2 = st.columns(2)

# 1. Display video
uri = video_path
col1.video(data=uri)

# 2. Preprocessing (frames sampling)
@st.cache_data(show_spinner="Processing video frames")
def frames_from_uri(uri, frames_per_scene, drop_similar_frames, frame_similarity_threshold):
    video_extractor = VideoExtractor(uri)
    frames, scenes_list = video_extractor.extract_frames_from_scenes(frames_per_scene)

    no_of_scenes = len(scenes_list)
    no_of_frames = len(frames)
    no_of_unique_frames = None

    if drop_similar_frames:
        frames = video_extractor.drop_similar_frames(frames, threshold=frame_similarity_threshold)
        no_of_unique_frames = len(frames)
    
    return frames, no_of_scenes, no_of_frames, no_of_unique_frames

start_time = time.time()

ss.frames, no_of_scenes, no_of_frames, no_of_unique_frames = frames_from_uri(
    uri, frames_per_scene, drop_similar_frames, frame_similarity_threshold
)

end_time = time.time()
duration = end_time - start_time
minutes = int(duration // 60)
seconds = int(duration % 60)
print(f"Sampling duration: {minutes:02}:{seconds:02}")

# Display sampling results
col1.write(
    f"{no_of_scenes} scenes detected. {no_of_frames} frames extracted." +
    (f" Reduced to {no_of_unique_frames} unique frames." if no_of_unique_frames else "")
)

# 3. Transcribe video
@st.cache_data(show_spinner="Transcribing video")
def get_transcription(uri):
    video_extractor = VideoExtractor(uri)
    transcription = video_extractor.transcribe_video(
        uri, ss.aoai_client_swece, ss.whisper_deployment
    )
    return transcription

if transcribe_audio:
    start_time = time.time()
    ss.transcription = get_transcription(uri)
    end_time = time.time()
    duration = end_time - start_time
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    print(f"Transcription duration: {minutes:02}:{seconds:02}")

# 4. Extract insights with LLM
@st.cache_data(show_spinner="Extracting insights with LLM")
def get_llm_insights(base64frames, transcription=None, system_message=None, max_retries=3, retry_delay=2):
    llm_insights = video_analyzer.video_chat(
        base64frames, transcription, system_message, max_retries, retry_delay
    )
    return llm_insights

frames_list = [entry['frame_base64'] for entry in ss.frames]

start_time = time.time()
print(f'Start LLM with {len(frames_list)} frames ...')
ss.video_insights = get_llm_insights(frames_list, transcription=ss.transcription)
end_time = time.time()
duration = end_time - start_time
minutes = int(duration // 60)
seconds = int(duration % 60)
print(f"LLM duration: {minutes:02}:{seconds:02}")

markdown_placeholder = col2.empty()
video_insights_markdown = dict_to_markdown_table(ss.video_insights)
markdown_placeholder.markdown(video_insights_markdown)

# 5. Get Content Safety results
@st.cache_data(show_spinner="Analyzing video with Azure Content Safety")
def get_moderation_insights(video_frames: list, severity_thresholds=None):
    nsfw_violations = video_analyzer.content_safety_moderate_video_parallel(
        ss.frames, severity_thresholds
    )
    return nsfw_violations

severity_thresholds = {
    'Hate': hate_thresh,
    'SelfHarm': selfharm_thresh,
    'Sexual': sexual_thresh,
    'Violence': violence_thresh
}

start_time = time.time()
moderation_insights = get_moderation_insights(ss.frames, severity_thresholds)
end_time = time.time()
duration = end_time - start_time
minutes = int(duration // 60)
seconds = int(duration % 60)
print(f"Content Safety duration: {minutes:02}:{seconds:02}")

# 6. Combine LLM and Content Safety insights and show them in the UI
ss.video_insights['Content Safety'] = [moderation_insights]
video_insights_markdown = dict_to_markdown_table(ss.video_insights)
markdown_placeholder.markdown(video_insights_markdown)

print(f'Video analyzer done processing {video_path}')
