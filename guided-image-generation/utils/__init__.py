import requests
import base64
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.contentsafety.models import AnalyzeImageOptions, ImageData
from azure.core.exceptions import HttpResponseError
import io
import os

# Analize the image with GPT-4-turbo-Vision
def analyze_image_gpt4v(image_path, system_message, user_prompt, api_key, aoai_endpoint='', aoai_deployment='', temperature=0, seed=None, detail='auto', max_tokens=150):

    with open(image_path, "rb") as image_file:
        base64_image= base64.b64encode(image_file.read()).decode('utf-8')

    # Call the endpoint
    headers = {"Content-Type": "application/json", "api-key": api_key} 
    model = aoai_deployment
    url = f"{aoai_endpoint}openai/deployments/{aoai_deployment}/chat/completions?api-version=2023-12-01-preview" 
    
    payload = {
        "model": model,
        "messages": [ 
            { "role": "system", "content": system_message }, 
            { "role": "user", "content": [  
                { 
                    "type": "text", 
                    "text": user_prompt 
                },
                { 
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail" : detail
                    }
                }
            ] } 
        ], 
        "temperature" : temperature,
        "seed" : seed,
        "max_tokens": max_tokens
        }

    response = requests.post(url, headers=headers, json=payload)    

    return response

# Create the Stable Diffusion Image
def create_sdxl_image(input, url, api_key, deployment_name=None):

    headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + api_key}
    if deployment_name:
        headers['azureml-model-deployment'] = deployment_name

    try:
        response = requests.post(url, headers=headers, json=input)
        if response.status_code == 200:
            return response
        else:
            print(response.text)
    except Exception as e:
        print(e)

def display_moderation_results(results):
    display_text = ""
    for category, info in results.items():
        if category == "jailbreak":
            if info['filtered']:
                display_text += f"<span style='color:red;'>jailbreak attempt</span>, "
            else:
                display_text += f"jailbreak: not detected, "

        else:
            if info['filtered']:
                display_text += f"{category}: <span style='color:red;'>{info['severity']}</span>, "
            else:
                display_text += f"{category}: {info['severity']}, "

    # Remove trailing comma and space
    display_text = display_text.rstrip(", ")
    return display_text

def check_and_reduce_image_size(image_path, max_size=4194304):
    # Check the size of the image
    image_size = os.path.getsize(image_path)
    if image_size > max_size:
        print('Reducing image size ...')
        # Image size exceeds the limit and needs to be reduced
        with Image.open(image_path) as img:
            # Reduce the quality of the image while saving it until it's under the required size
            quality = 95
            while image_size > max_size and quality > 10:
                # Define the new image path with a suffix to indicate it's reduced
                base, ext = os.path.splitext(image_path)
                new_image_path = f"{base}_reduced{ext}"
                
                # Save the image with reduced quality
                img.save(new_image_path, quality=quality)
                
                # Check the size of the reduced image
                image_size = os.path.getsize(new_image_path)
                quality -= 5  # Decrease quality for the next iteration if needed
            
            return new_image_path
    else: # no resize needed
        return image_path
        
def analyze_content_safety(image_path, endpoint, key):

    # Create an Azure AI Content Safety client
    client = ContentSafetyClient(endpoint, AzureKeyCredential(key))

    # Build request
    with open(image_path, "rb") as file:
        request = AnalyzeImageOptions(image=ImageData(content=file.read()))

    # Analyze image
    try:
        response = client.analyze_image(request)
        return response
        
    except HttpResponseError as e:
        print("Analyze image failed.")
        if e.error:
            print(f"Error code: {e.error.code}")
            print(f"Error message: {e.error.message}")
            raise
        print(e)
        raise
