import requests
import base64
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from azure.ai.contentsafety import ContentSafetyClient
# from azure.core.credentials import AzureKeyCredential
# from azure.ai.contentsafety.models import AnalyzeImageOptions, ImageData
# from azure.core.exceptions import HttpResponseError
import io
import os

def dict_to_markdown_table(data):
    markdown = []
    
    for key, value in data.items():
        if isinstance(value, str):
            # Case 1: Top-level key with a string value
            markdown.append(f"| **{key}** | {value} |")
        elif isinstance(value, list):
            if all(isinstance(item, dict) for item in value):
                # Case 3: Top-level key with a list of dicts (brands/products)
                markdown.append(f"| **{key}** | |")
                for entry in value:
                    for subkey, subvalue in entry.items():
                        timestamps = ", ".join(subvalue)
                        markdown.append(f"| {subkey} | {timestamps} |")
            else:
                # Case 2: Top-level key with a list of timestamps
                timestamps = ", ".join(value)
                markdown.append(f"| **{key}** | {timestamps} |")
        else:
            # Handle any other unexpected cases
            markdown.append(f"| **{key}** | {str(value)} |")

    # Adding table header
    if markdown:
        markdown.insert(0, "| Category | Insights and time frames |")
        markdown.insert(1, "| --- | ----- |")
    
    return "\n".join(markdown)



def analyze_image_gpt4o(image_path, system_message, user_prompt, api_key, aoai_endpoint='', aoai_deployment='', temperature=0, seed=None, detail='auto', max_tokens=150, api='aoai'):

    with open(image_path, "rb") as image_file:
        base64_image= base64.b64encode(image_file.read()).decode('utf-8')

    if api == 'aoai':
        headers = {"Content-Type": "application/json", "api-key": api_key} 
        model = aoai_deployment
        url = f"{aoai_endpoint}openai/deployments/{aoai_deployment}/chat/completions?api-version=2024-02-01" 

    else: # OpenAI
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        model = 'gpt-4-vision-preview'
        url = "https://api.openai.com/v1/chat/completions"
    
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

def azure_image_analysis_predict(image_path, model_name, vision_endpoint, vision_key):

    url = f"{vision_endpoint}computervision/imageanalysis:analyze?model-name={model_name}&api-version=2023-02-01-preview"

    with open(image_path, 'rb') as file:
        data = file.read()

    headers = {
        'Ocp-Apim-Subscription-Key': vision_key,
        'Content-Type': 'application/octet-stream'
    }

    response = requests.post(url, headers=headers, data=data)

    if response.status_code == 200:
        return(response.json())
    else:
        print("Error:")
        print(response.text)


def azure_image_analysis_create_image(image_path, response_json, threshold=0.5, font_size=8):
    # Open the image file
    with open(image_path, 'rb') as file:
        image = Image.open(io.BytesIO(file.read()))

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    
    # No axis for the image
    ax.axis('off')

    # Process each detected object
    if response_json['customModelResult']:
        for obj in response_json['customModelResult']['objectsResult']['values']:
            confidence = obj['tags'][0]['confidence']
            if confidence >= threshold:
                # Extract bounding box coordinates
                box = obj['boundingBox']
                left, top, width, height = box['x'], box['y'], box['w'], box['h']

                # Create a Rectangle patch
                rect = patches.Rectangle((left, top), width, height, linewidth=2, edgecolor='red', facecolor='none')
                # Add the rectangle to the Axes
                ax.add_patch(rect)

                # Add the text label
                label = obj['tags'][0]['name']
                text = f"{label} ({confidence:.2f})"
                ax.text(left, top, text, color='white', fontsize=font_size, bbox=dict(facecolor='red', alpha=0.5))
    else:
        # no object found
        pass

    return fig

def create_sdxl_image(input, url, api_key, deployment_name=None):

    headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + api_key}
    if deployment_name:
        headers['azureml-model-deployment'] = deployment_name

    response = requests.post(url, headers=headers, json=input)
    if response.status_code == 200:
        return response
    else:
        print(response.text)

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