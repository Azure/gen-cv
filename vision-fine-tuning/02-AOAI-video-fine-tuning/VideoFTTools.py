import base64
import json
import os
import time
from io import BytesIO
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from azure.storage.blob import BlobServiceClient, ContentSettings
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

# Not required in current version
# from moviepy.editor import VideoFileClip, concatenate_videoclips
# import imagehash

class DatasetHelper:
    @staticmethod
    def plot_label_counts(df, splitname=None):
        """
        Plots a bar chart showing the frequency of each label in the dataset.

        Parameters:
        - df (pd.DataFrame): The DataFrame containing the data with a 'label' column.
        - splitname (str, optional): A descriptive name for the dataset split 
          (e.g., 'train', 'test', 'validation'). This will be included in the chart title if provided.

        Example:
        >>> DatasetHelper.plot_label_counts(train_df, "Train Set")
        """
        value_counts = df['label'].value_counts()

        plt.figure(figsize=(20, 6)) 
        bars = plt.bar(value_counts.index, value_counts.values)

        # Annotate absolute counts on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, str(height),
                     ha='center', va='bottom', fontsize=8)

        plt.xlabel('Label', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        prefix = f"{splitname:}" if splitname else ""
        plt.title(f'{prefix}: {len(value_counts)} classes. {df.shape[0]} instances', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=8)  
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_video_duration_histogram(df, dataset_path):
        video_durations = []

        for clip_path in df['clip_path']:
            full_path = os.path.join(dataset_path, clip_path) 
            cap = cv2.VideoCapture(full_path) 
            
            if not cap.isOpened():
                print(f"Unable to open video: {full_path}")
                continue
            
            # Get frame count and frame rate
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            
            if frame_rate > 0:
                video_length = frame_count / frame_rate  # Length in seconds
                video_durations.append(video_length)
            else:
                print(f"Invalid frame rate in video: {full_path}")
            
            cap.release()  # Release the video file

        # Plot histogram of video durations
        plt.figure(figsize=(15, 6))
        plt.hist(video_durations, bins=100, edgecolor='black')
        plt.xlabel('Video Duration (seconds)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Histogram of Video Lengths', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

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
            minutes = int(timestamp_sec // 60)
            seconds = int(timestamp_sec % 60)
            milliseconds = int((timestamp_sec - int(timestamp_sec)) * 1000)
            timestamp = f"{minutes:02}:{seconds:02}:{milliseconds:03}"
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
    

    def extract_n_video_frames(self, n: int) -> List[Dict[str, str]]:
        """
        Extract a specified number of frames, equally distributed over the video's duration.

        Args:
            n (int): The number of frames to extract.

        Returns:
            List[Dict[str, str]]: List of dicts with timestamp and base64-encoded images with timestamps visually added.
        """
        if n <= 0:
            raise ValueError("The number of frames to extract must be greater than zero.")
        if n > self.frame_count:
            raise ValueError("The number of frames to extract cannot exceed the total frame count.")

        interval = self.duration / n
        frame_indices = (np.linspace(0, self.duration, n, endpoint=False) * self.fps).astype(int)
        frames = []

        for frame_index in frame_indices:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = self.cap.read()
            if not ret:
                continue
            timestamp_sec = frame_index / self.fps
            minutes = int(timestamp_sec // 60)
            seconds = int(timestamp_sec % 60)
            milliseconds = int((timestamp_sec - int(timestamp_sec)) * 1000)
            timestamp = f"{minutes:02}:{seconds:02}:{milliseconds:03}"
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
        
        # print(f"{len(frames)} frames extracted")

        return frames

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
    def __init__(self, openai_client, model):
        self.openai_client = openai_client
        self.model = model

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
            if attempt > 0:
                print(f"VideoAnalyzer.video_chat() Retry attempt {attempt}")
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

class Evaluator:
    @staticmethod
    def plot_learning_curves(df, smoothing_window=10):
        """
        Plots learning curves for a fine-tuning job with smoothing for train metrics.

        Parameters:
        df (pd.DataFrame): Dataframe containing the training and validation metrics.
        smoothing_window (int): Window size for rolling mean smoothing.
        """
        epochs = df[df['full_valid_loss'].notna()]['step'].values

        # Interpolate missing values for validation metrics
        df['valid_loss_interpolated'] = df['valid_loss'].interpolate()
        df['valid_mean_token_accuracy_interpolated'] = df['valid_mean_token_accuracy'].interpolate()

        # Compute smoothed train metrics
        df['train_loss_smoothed'] = df['train_loss'].rolling(window=smoothing_window, min_periods=1).mean()
        df['train_mean_token_accuracy_smoothed'] = df['train_mean_token_accuracy'].rolling(window=smoothing_window, min_periods=1).mean()

        # Initialize subplots
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))

        # Plot Train Loss
        axs[0, 0].plot(df['step'], df['train_loss'], label='Train Loss (Original)', linewidth=0.5, alpha=0.7)
        axs[0, 0].plot(df['step'], df['train_loss_smoothed'], label='Train Loss (Smoothed)', linewidth=2, color='blue')
        for epoch in epochs:
            axs[0, 0].axvline(x=epoch, color='gray', linestyle='--', linewidth=0.5)
        axs[0, 0].set_title("Train Loss")
        axs[0, 0].set_xlabel("Step")
        axs[0, 0].set_ylabel("Loss")
        axs[0, 0].legend()

        # Plot Train Mean Token Accuracy
        axs[0, 1].plot(df['step'], df['train_mean_token_accuracy'], label='Train Mean Token Accuracy (Original)', linewidth=0.5, alpha=0.7, color='orange')
        axs[0, 1].plot(df['step'], df['train_mean_token_accuracy_smoothed'], label='Train Mean Token Accuracy (Smoothed)', linewidth=2, color='darkorange')
        for epoch in epochs:
            axs[0, 1].axvline(x=epoch, color='gray', linestyle='--', linewidth=0.5)
        axs[0, 1].set_title("Train Mean Token Accuracy")
        axs[0, 1].set_xlabel("Step")
        axs[0, 1].set_ylabel("Accuracy")
        axs[0, 1].legend()

        # Plot Validation Loss
        axs[1, 0].plot(df['step'], df['valid_loss_interpolated'], label='Validation Loss', color='green')
        axs[1, 0].scatter(df['step'], df['full_valid_loss'], color='red', label='Full Validation Loss (Epoch)', edgecolor='black')
        for epoch in epochs:
            axs[1, 0].axvline(x=epoch, color='gray', linestyle='--', linewidth=0.5)
        axs[1, 0].set_title("Validation Loss")
        axs[1, 0].set_xlabel("Step")
        axs[1, 0].set_ylabel("Loss")
        axs[1, 0].legend()

        # Plot Validation Mean Token Accuracy
        axs[1, 1].plot(df['step'], df['valid_mean_token_accuracy_interpolated'], label='Validation Mean Token Accuracy', color='purple')
        axs[1, 1].scatter(df['step'], df['full_valid_mean_token_accuracy'], color='red', label='Full Validation Accuracy (Epoch)', edgecolor='black')
        for epoch in epochs:
            axs[1, 1].axvline(x=epoch, color='gray', linestyle='--', linewidth=0.5)
        axs[1, 1].set_title("Validation Mean Token Accuracy")
        axs[1, 1].set_xlabel("Step")
        axs[1, 1].set_ylabel("Accuracy")
        axs[1, 1].legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def compare_model_metrics(df):
        """
        Compares the performance of two models based on accuracy, precision, and recall.

        Parameters:
        df (pd.DataFrame): Dataframe containing 'label', 'base_predicted_label', 'ft_predicted_label'.
        """
        # Compute metrics for the base model
        base_accuracy = accuracy_score(df['label'], df['base_predicted_label'])
        base_precision = precision_score(df['label'], df['base_predicted_label'], average='weighted', zero_division=True)
        base_recall = recall_score(df['label'], df['base_predicted_label'], average='weighted', zero_division=True)

        # Compute metrics for the fine-tuned model
        ft_accuracy = accuracy_score(df['label'], df['ft_predicted_label'])
        ft_precision = precision_score(df['label'], df['ft_predicted_label'], average='weighted', zero_division=True)
        ft_recall = recall_score(df['label'], df['ft_predicted_label'], average='weighted', zero_division=True)

        # Create a dictionary for side-by-side comparison
        metrics = {
            'Metric': ['Accuracy', 'Precision', 'Recall'],
            'Base Model': [base_accuracy, base_precision, base_recall],
            'Fine-Tuned Model': [ft_accuracy, ft_precision, ft_recall]
        }

        # Plotting
        x = range(len(metrics['Metric']))
        bar_width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        base_bars = ax.bar(x, metrics['Base Model'], bar_width, label='Base Model')
        ft_bars = ax.bar([i + bar_width for i in x], metrics['Fine-Tuned Model'], bar_width, label='Fine-Tuned Model')

        # Add metric values as labels on top of bars
        for bars in [base_bars, ft_bars]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.3f}', ha='center', va='bottom')

        ax.set_xticks([i + bar_width / 2 for i in x])
        ax.set_xticklabels(metrics['Metric'])
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.legend(loc='lower right')

        plt.tight_layout()
        plt.show()

    @staticmethod
    
    def plot_confusion_matrix(y_true, y_pred):

        # y_true = df['label'] 
        # y_pred = df['base_predicted_label']
        cm = confusion_matrix(y_true, y_pred)

        # Plot the confusion matrix
        plt.figure(figsize=(10, 10))  # Adjust the figure size as needed for 101 classes
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', square=True)

        # Configure axis labels and ticks
        plt.xlabel('Predicted Labels', fontsize=12)
        plt.ylabel('True Labels', fontsize=12)
        plt.xticks(ticks=range(len(cm)), labels=y_true.unique(), rotation=45, ha='right', fontsize=8)
        plt.yticks(ticks=range(len(cm)), labels=y_true.unique(), rotation=0, fontsize=8)
        plt.title('Confusion Matrix', fontsize=14)

        # Show the plot
        plt.tight_layout()
        plt.show()


# GENERAL FUNCTIONS
def date_sorted_df(details_dict):
    """ Create a pandas DataFrame from a dictionary and sort it by a 'created' or 'created_at' timestamp column for displaying OpenAI API tables. """
    df = pd.DataFrame(details_dict)
    
    if 'created' in df.columns:
        df.rename(columns={'created': 'created_at'}, inplace=True)
    
    # Convert 'created_at' from Unix timestamp to human-readable date/time format
    df['created_at'] = pd.to_datetime(df['created_at'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')

    if 'finished_at' in df.columns:
        # Convert 'finished_at' from Unix timestamp to human-readable date/time format, keeping null values as is
        df['finished_at'] = pd.to_datetime(df['finished_at'], unit='s', errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Sort DataFrame by 'created_at' in descending order
    df = df.sort_values(by='created_at', ascending=False)

    return df

def upload_frame_to_blob_as_jpeg(base64_frame, connection_string, container_name, blob_name):

    """
    Uploads a base64-encoded frame to Azure Blob Storage as a JPEG file, keeping operations in memory.

    Args:
        base64_frame (str): Base64-encoded image data.
        connection_string (str): Azure Blob Storage connection string.
        container_name (str): Name of the Azure Blob container.
        blob_name (str): Name of the blob (including virtual folder paths if applicable).

    Returns:
        str: Full URL of the uploaded blob.
    """
    try:
        # Decode Base64-encoded frame to binary data
        decoded_data = base64.b64decode(base64_frame)
        nparr = np.frombuffer(decoded_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Decoded Base64 data is not a valid image.")

        # Encode image as JPEG and store in an in-memory buffer
        is_success, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        if not is_success:
            raise ValueError("Failed to encode image as JPEG.")
        
        # Convert the buffer to BytesIO for in-memory upload
        in_memory_file = BytesIO(buffer.tobytes())

        # Initialize BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)

        # Ensure the container exists
        if not container_client.exists():
            container_client.create_container()

        # Upload the JPEG file from memory
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(
            in_memory_file.getvalue(),
            blob_type="BlockBlob",
            overwrite=True,
            content_settings=ContentSettings(content_type="image/jpeg")
        )

        # Construct full URL
        blob_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob_name}"
        return blob_url
    except Exception as e:
        raise RuntimeError(f"Failed to upload frame to blob: {e}")
