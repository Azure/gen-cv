import os
import json
import time
import io
import base64
from datetime import timedelta
from typing import List, Dict

import requests
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imagehash
from moviepy.editor import VideoFileClip, concatenate_videoclips
from scenedetect import detect, AdaptiveDetector
from concurrent.futures import ThreadPoolExecutor, as_completed


def download_youtube_video(url, target_dir="videos", max_retries=2):
    """
    Downloads a YouTube video to a specified directory and retrieves its metadata.

    Args:
        youtube_video_id (str): The ID of the YouTube video to download.
        target_dir (str, optional): The directory to save the downloaded video. Defaults to VIDEO_DIR.
        max_retries (int, optional): The maximum number of retries for downloading the video. Defaults to 2.

    Returns:
        dict: A dictionary containing the video's metadata. 
    """

    try:
        yt = YouTube(url=url)
        title = yt.title
        # yt.title = youtube_video_id
        stream = yt.streams.get_highest_resolution()

        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
                
        local_path = stream.download(target_dir, max_retries=max_retries)

        return {
            "local_path" : local_path,
            "title":title,
            "description":yt.description,
            "length":yt.length,
            "author":yt.author,
            "views":yt.views,
            "publish_date":yt.publish_date,
            "keywords":yt.keywords,
            "thumbnail_url":yt.thumbnail_url,
            }

    except Exception as e:
        print(f"Unable to download video {title} after {max_retries} retries: {e}")


class VideoCreator:

    @staticmethod
    def resize_image_to_allowed_resolutions(gen_folder, source_image, save_as=None):

        image_path = os.path.join(gen_folder, source_image)

        allowed_resolutions = {
            (16, 9): (1024, 576),
            (9, 16): (576, 1024),
            (1, 1): (768, 768)
        }
        
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                aspect_ratio = width / height
                
                # Determine which resolution to use
                if aspect_ratio == 16 / 9:
                    target_resolution = allowed_resolutions[(16, 9)]
                elif aspect_ratio == 9 / 16:
                    target_resolution = allowed_resolutions[(9, 16)]
                elif aspect_ratio == 1:
                    target_resolution = allowed_resolutions[(1, 1)]
                else:
                    raise ValueError(f"Unsupported aspect ratio: {aspect_ratio}")
                
                # Resize the image
                img = img.resize(target_resolution, Image.LANCZOS)
                
                # Determine the save path
                save_path = os.path.join(gen_folder, save_as) if save_as else image_path
                img.save(save_path)
                print(f"Image saved as {save_path}")
        
        except Exception as e:
            print(f"An error occurred: {e}")

    @staticmethod
    def image_to_video(gen_folder, source_image, stability_ai_key):
        if not os.path.exists(gen_folder):
            os.makedirs(gen_folder)

        source_image_path = os.path.join(gen_folder, source_image)
        response = requests.post(
            "https://api.stability.ai/v2beta/image-to-video",
            headers={
                "authorization": f"Bearer {stability_ai_key}"
            },
            files={
                "image": open(source_image_path, "rb")
            },
            data={
                "seed": 0,
                "cfg_scale": 1.8,
                "motion_bucket_id": 127
            },
        )

        if response.status_code != 200:
            raise Exception(f"Failed to start video generation: {response.text}")

        generation_id = response.json().get('id')
        if not generation_id:
            raise Exception("No generation ID received")

        target_image_path = VideoCreator.get_next_filename(gen_folder, source_image)
        while True:
            response = requests.get(
                f"https://api.stability.ai/v2beta/image-to-video/result/{generation_id}",
                headers={
                    'accept': "video/*",
                    'authorization': f"Bearer {stability_ai_key}"
                },
            )

            if response.status_code == 200:
                with open(target_image_path, 'wb') as file:
                    file.write(response.content)
                print(f"Generation complete: {target_image_path}")
                break
            elif response.status_code == 202:
                print("Generation in-progress, retrying in 10 seconds...")
                time.sleep(10)
            else:
                raise Exception(f"Error during video generation: {response.text}")
            
        return target_image_path

    @staticmethod
    def get_next_filename(gen_folder, source_image):
        prefix, _ = os.path.splitext(source_image)
        counter = 1
        while True:
            new_filename = f"{prefix}-{counter:03d}.mp4"
            if not os.path.exists(os.path.join(gen_folder, new_filename)):
                return os.path.join(gen_folder, new_filename)
            counter += 1

    @staticmethod
    def concatenate_videos(gen_folder, video_files, output_file):
        clips = []
        try:
            # Load each video file as a VideoFileClip object
            for video_file in video_files:
                source_path = os.path.join(gen_folder, video_file)
                clip = VideoFileClip(source_path)
                clips.append(clip)

            # Concatenate all clips
            final_clip = concatenate_videoclips(clips, method="compose")

            # Write the result to a file
            target_path = os.path.join(gen_folder, output_file)
            final_clip.write_videofile(target_path, codec='libx264', audio_codec='aac')

            # Close all clips
            for clip in clips:
                clip.close()
            
            print(f"Video saved as {target_path}")
        
        except Exception as e:
            print(f"An error occurred: {e}")


class VideoExtractor:
    """
    A class to extract and process video frames.
    """

    def __init__(self, uri: str):
        """
        Initialize the VideoExtractor with a video URI.

        Args:
            uri (str): The URI of the video file.
        """
        self.uri = uri
        self.cap = cv2.VideoCapture(uri)
        if not self.cap.isOpened():
            raise ValueError("Error opening video file")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps
    
    def transcribe_video(self, uri: str, openai_client, model) -> str:
        # Extract audio from video
        clip = VideoFileClip(uri)

        if clip.audio is not None:
            audio_path = os.path.join(".", "audio-track.mp3")
            clip.audio.write_audiofile(audio_path, bitrate="32k")
            clip.audio.close()
            print(f"Extracted audio to {audio_path}. Transcription in progress ...")
            transcription = openai_client.audio.transcriptions.create(
                model=model,
                file=open(audio_path, "rb"),
                response_format="text")
        else:
            transcription = None
            print("No audio track found in the video.")
            
        clip.close()
        return transcription
    
    
    def extract_video_frames(self, interval: float) -> List[Dict[str, str]]:
        """
        Extract frames from the video at regular intervals.

        Args:
            interval (float): Interval in seconds to extract frames.

        Returns:
            List[Dict[str, str]]: List of dicts with timestamp and base64-encoded images with timestamps visually added.
        """
        frame_indices = np.arange(0, self.duration, interval) * self.fps
        frame_indices = frame_indices.astype(int)
        frames = []

        for frame_index in frame_indices:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = self.cap.read()
            if not ret:
                continue
            timestamp_sec = frame_index / self.fps
            hours = int(timestamp_sec // 3600)
            minutes = int(timestamp_sec // 60)
            seconds = int(timestamp_sec % 60)
            timestamp = f"{hours:02}:{minutes:02}:{seconds:02}"
            timestamp_text = f"video_time: {timestamp}"
            
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # Update this path if necessary
            font_size = 16  # Font size in pixels
            font = ImageFont.truetype(font_path, font_size)
            
            # Create a new image with extra height for the text
            stripe_height = font_size + 4  # Height of the black stripe
            new_frame_height = frame.shape[0] + stripe_height
            new_frame = np.zeros((new_frame_height, frame.shape[1], 3), dtype=np.uint8)
            
            # Copy the original frame into the new frame
            new_frame[:frame.shape[0], :] = frame
            
            # Convert the new frame to a PIL image to draw text
            pil_img = Image.fromarray(new_frame)
            draw = ImageDraw.Draw(pil_img)
            
            # Draw the black stripe
            draw.rectangle([(0, frame.shape[0]), (frame.shape[1], new_frame_height)], fill=(0, 0, 0))
            
            # Draw the text
            text_x = 5
            text_y = frame.shape[0] + 1
            draw.text((text_x, text_y), timestamp_text, font=font, fill=(255, 255, 255))
            
            # Convert back to OpenCV image
            new_frame = np.array(pil_img)
            
            _, buffer = cv2.imencode('.jpg', new_frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            frames.append({"timestamp": timestamp, "frame_base64": frame_base64})
        
        print(f"{len(frames)} frames extracted")

        return frames


    def extract_frames_from_scenes(self, frames_per_scene: int) -> List[Dict[str, str]]:
            """
            Detect scenes in the video and extract frames.

            Args:
                frames_per_scene (int): Number of frames to extract per scene.

            Returns:
                List[Dict[str, str]]: List of dicts with timestamp and base64-encoded images with timestamps visually added.
            """
            scene_list = detect(self.uri, AdaptiveDetector())
            print(f"{len(scene_list)} scenes detected.")
            frames = []

            for scene in scene_list:
                start_frame = scene[0].get_frames()
                end_frame = scene[1].get_frames()
                scene_length = end_frame - start_frame

                for i in range(frames_per_scene):
                    frame_index = start_frame + int((i + 1) / (frames_per_scene + 1) * scene_length)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ret, frame = self.cap.read()
                    if not ret:
                        continue
                    timestamp_sec = frame_index / self.fps
                    hours = int(timestamp_sec // 3600)
                    minutes = int(timestamp_sec // 60)
                    seconds = int(timestamp_sec % 60)
                    timestamp = f"{hours:02}:{minutes:02}:{seconds:02}"
                    timestamp_text = f"video_time: {timestamp}"
                    
                    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # Update this path if necessary
                    font_size = 16  # Font size in pixels
                    font = ImageFont.truetype(font_path, font_size)
                    
                    # Create a new image with extra height for the text
                    stripe_height = font_size + 4  # Height of the black stripe
                    new_frame_height = frame.shape[0] + stripe_height
                    new_frame = np.zeros((new_frame_height, frame.shape[1], 3), dtype=np.uint8)
                    
                    # Copy the original frame into the new frame
                    new_frame[:frame.shape[0], :] = frame
                    
                    # Convert the new frame to a PIL image to draw text
                    pil_img = Image.fromarray(new_frame)
                    draw = ImageDraw.Draw(pil_img)
                    
                    # Draw the black stripe
                    draw.rectangle([(0, frame.shape[0]), (frame.shape[1], new_frame_height)], fill=(0, 0, 0))
                    
                    # Draw the text
                    text_x = 5
                    text_y = frame.shape[0] + 1
                    draw.text((text_x, text_y), timestamp_text, font=font, fill=(255, 255, 255))
                    
                    # Convert back to OpenCV image
                    new_frame = np.array(pil_img)
                    
                    _, buffer = cv2.imencode('.jpg', new_frame)
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    frames.append({"timestamp": timestamp, "frame_base64": frame_base64})

            print(f"{len(frames)} frames extracted")
            
            return frames, scene_list

    def drop_similar_frames(self, frames: List[Dict[str, str]], hash_size: int = 8, threshold: int = 5) -> List[Dict[str, str]]:
        """
        Drop visually similar frames based on perceptual hashing.

        Args:
            frames (List[Dict[str, str]]): List of dicts with timestamp and base64-encoded images.
            hash_size (int, optional): Size of the hash. Default is 8.
            threshold (int, optional): Hash difference threshold to consider frames similar. Default is 5.

        Returns:
            List[Dict[str, str]]: List of unique dicts with timestamp and base64-encoded images.
        """
        def calculate_hash(image_base64: str) -> str:
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            return imagehash.phash(image, hash_size=hash_size)
        
        unique_frames = []
        seen_hashes = []

        for frame_dict in frames:
            frame_hash = calculate_hash(frame_dict["frame_base64"])
            if all(frame_hash - h > threshold for h in seen_hashes):
                unique_frames.append(frame_dict)
                seen_hashes.append(frame_hash)

        print(f"{len(unique_frames)} unique frames extracted")

        return unique_frames

    @staticmethod
    def display_frames(frames: List[Dict[str, str]], height: int = 100):
        """
        Display frames in a Jupyter Notebook.

        Args:
            frames (List[Dict[str, str]]): List of dicts with timestamp and base64-encoded images.
            height (int, optional): Height of the displayed frames. Default is 100.
        """
        from IPython.display import display, HTML
        html_content = '<div style="overflow-x: auto; white-space: nowrap;">'

        for frame_dict in frames:
            html_content += f'''
            <div style="display: inline-block; margin-right: 10px; text-align: center;">
                <img src="data:image/jpeg;base64,{frame_dict["frame_base64"]}" height="{height}px" style="display: block; margin: 0 auto;">
                <div>{frame_dict["timestamp"]}</div>
            </div>
            '''

        html_content += '</div>'
        display(HTML(html_content))

class VideoAnalyzer:
    def __init__(self, openai_client, model, content_safety_endpoint, content_safety_key):
        self.openai_client = openai_client
        self.model = model
        self.content_safety_endpoint = content_safety_endpoint
        self.content_safety_key = content_safety_key

    def video_chat(self, base64frames, transcription=None, system_message=None, max_retries=3, retry_delay=2):
        sys_message_transcription_note = None
        user_message_transcription_note = "No audio transcription was provided for this video"

        if transcription:
            sys_message_transcription_note = "Following the video frames, you will receive the transcription of the audio track, which does not contain timestamps."
            user_message_transcription_note = f"The audio transcription is: {transcription}"

        if system_message is None:
            system_message = f"""
            You are an expert in analyzing and moderating video files.
            You are provided with extracted frames from the video. Each frame includes a timestamp on the lower left in the format 'video_time: hh:mm:ss'. 
            Use these timestamps to understand the course of the video and for any reference to video timing.
            {sys_message_transcription_note}
            Use the provided video data to extract the following information:
            - video summary
            - any visual generative AI flaws or artifacts in the frames. Examples: Incorrect number of human fingers, Asymmetrical or distorted facial features, Unnatural body proportions or poses
            - scenes that show alcohol or alcohol consumption
            - scenes which show illegal drugs or narcotics
            - scenes which show smoking or tobacco consumption
            - scenes which show weapons like guns, rifles, knives etc.
            - brands, brand logos or products you are able to identify
            - problematic stereotypes that you are able to identify. Examples: Gender stereotypes, racial or ethnic stereotypes, body image stereotypes, disability stereotypes, cultural stereotypes
                 
            Provide the information as a valid JSON object in this format:
            {{
                "Video summary": "provide a short summary of the video without referring to timestamps" (string),
                "AI artifacts": ["list of timestamps in which generative AI flaws or artifacts appeared. Empty list if none"],
                "Alcohol": ["list of timestamps in which alcohol appeared. Empty list if none"],
                "Drugs/narcotics": ["list of timestamps in which illegal drugs or narcotics appeared. Empty list if none"],
                "Smoking": ["list of timestamps in which smoking or tobacco consumption appeared. Empty list if none"],
                "Weapons": ["list of timestamps in which weapons appeared. Empty list if none"],
                "Brands/products": [
                    {{"name of detected brand or product 1" : ["list of timestamps where brand or product 1 appeared"]}},
                    {{"name of detected brand or product 2" : ["list of timestamps where brand or product 2 appeared"]}},
                    {{"name of detected brand or product n" : ["list of timestamps where brand or product n appeared"]}}
                ], # empty list if none
                "Stereotypes": [
                    {{"kind of stereotype 1" : ["list of timestamps where stereotype 1 appeared"]}},
                    {{"kind of stereotype 2" : ["list of timestamps where stereotype 2 appeared"]}},
                    {{"kind of stereotype n" : ["list of timestamps where stereotype n appeared"]}}
                ], # empty list if none
            }}
            """

        for attempt in range(max_retries):
            print(f"VideoAnalyzer.video_chat() Attempt {attempt}")
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": "These are the frames from the video.",},
                    {"role": "user", "content": [
                        *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "auto"}}, base64frames),
                        {"type": "text", "text": user_message_transcription_note},
                    ]}
                ],
                temperature=0,
                seed=0,
                response_format={"type": "json_object"},
            )

            try:
                response_dict = json.loads(response.choices[0].message.content)
                return response_dict

            except (json.JSONDecodeError, ValueError) as e:
                print('Error extracting JSON from LLM response. Retrying...')
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise e

        raise RuntimeError("Failed to obtain valid response from the model after retries")

    def content_safety_moderate_image(self, base64_image: str):
        url = f"{self.content_safety_endpoint}/contentsafety/image:analyze?api-version=2024-02-15-preview"
        headers = {
            'Ocp-Apim-Subscription-Key': self.content_safety_key,
            'Content-Type': 'application/json'
        }
        payload = {
            "image": {
                "content": base64_image
            },
            "categories": ["Hate", "SelfHarm", "Sexual", "Violence"],
            "outputType": "FourSeverityLevels"
        }

        response = requests.post(url, headers=headers, json=payload)
        return response.json()

    def content_safety_moderate_video(self, video_frames: list, severity_thresholds=None):
        if severity_thresholds is None:
            severity_thresholds = {'Hate': 2, 'SelfHarm': 2, 'Sexual': 2, 'Violence': 2}

        id_to_severity = {0: "safe", 2: "low", 4: "medium", 6: "high"}
        # severity_to_id = {v: k for (k, v) in id_to_severity.items()}

        nsfw_violations = {key: [] for key in severity_thresholds}

        for video_frame in video_frames:
            moderation_results = self.content_safety_moderate_image(video_frame['frame_base64'])
            video_frame['moderation_results'] = moderation_results

            for check in moderation_results['categoriesAnalysis']:
                timestamp = video_frame['timestamp']
                detected_risk_label = id_to_severity[check['severity']]
                threshold_label = id_to_severity[severity_thresholds[check['category']]]

                if check['severity'] >= severity_thresholds[check['category']]:
                    print(f"Severity at frame {timestamp} for category {check['category']} is {detected_risk_label} and would be blocked at given threshold of {threshold_label}")
                    nsfw_violations[check['category']].append(timestamp)

        return nsfw_violations

    def content_safety_moderate_video_parallel(self, video_frames: list, severity_thresholds=None):
        if severity_thresholds is None:
            severity_thresholds = {'Hate': 2, 'SelfHarm': 2, 'Sexual': 2, 'Violence': 2}

        id_to_severity = {0: "safe", 2: "low", 4: "medium", 6: "high"}

        nsfw_violations = {key: [] for key in severity_thresholds}

        def process_frame(video_frame):
            moderation_results = self.content_safety_moderate_image(video_frame['frame_base64'])
            video_frame['moderation_results'] = moderation_results
            frame_violations = {key: [] for key in severity_thresholds}

            for check in moderation_results['categoriesAnalysis']:
                if check['severity'] >= severity_thresholds[check['category']]:
                    
                    # print(f"Time: {video_frame['timestamp']} - {check['category']}: Detected: {check['severity']}, Thresh: {severity_thresholds[check['category']]}")
                    frame_violations[check['category']].append(video_frame['timestamp'])

            return frame_violations

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_frame, frame) for frame in video_frames]
            for future in as_completed(futures):
                frame_violations = future.result()
                for category, timestamps in frame_violations.items():
                    nsfw_violations[category].extend(timestamps)

        return nsfw_violations

    def video_chat_questions(self, base64frames, questions, transcription=None, system_message=None, max_retries=1, retry_delay=2): # 3
            sys_message_transcription_note = None
            user_message_transcription_note = "No audio transcription was provided for this video"

            if transcription:
                sys_message_transcription_note = "Following the video frames, you will receive the transcription of the audio track, which does not contain timestamps."
                user_message_transcription_note = f"The audio transcription is: {transcription}"

            import random
            random_prefix = random.randint(1, 10000)

            if system_message is None:
                system_message = f"""
                You are an expert in analyzing and moderating video files.
                You are provided with extracted frames from the video. Each frame includes a timestamp on the lower left in the format 'video_time: hh:mm:ss'. 
                Use these timestamps to understand the course of the video and for any reference to video timing.
                {sys_message_transcription_note}
                Use the provided video data to answer user questions.
                    
                Provide the information as a valid JSON object in this format:
                {{
                    "<your answer to question 1>": ["list of timestamps related to answer of question 1"], # empty list if not applicable
                    "<your answer to question 2>": ["list of timestamps related to answer of question 2"], # empty list if not applicable
                    "<your answer to question n>": ["list of timestamps related to answer of question n"], # empty list if not applicable
                }}
                """
            question_messages = [{"type": "text", "text": question} for question in questions]

            for attempt in range(max_retries):
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": "These are the frames from the video.",},
                        {"role": "user", "content": [
                            *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "auto"}}, base64frames),
                            {"type": "text", "text": user_message_transcription_note},
                            
                        ],
                        },
                        {"role": "user", "content": "Here are the user questions:"},
                        {"role": "user", "content": [*question_messages]},

                    ],
                    temperature=0,
                    seed=0,
                    response_format={"type": "json_object"},
                )

                try:
                    response_dict = json.loads(response.choices[0].message.content)
                    return response_dict

                except (json.JSONDecodeError, ValueError) as e:
                    print('Error extracting JSON from LLM response. Retrying...')
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    else:
                        raise e

            raise RuntimeError("Failed to obtain valid response from the model after retries")
