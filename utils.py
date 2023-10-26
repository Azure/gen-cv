import os
from dotenv import load_dotenv
import json
from PIL import Image
import requests
import math
from io import BytesIO
import matplotlib.pyplot as plt
import azure.ai.vision as sdk
import pickle
from tenacity import retry, stop_after_attempt, wait_random_exponential
import openai

# Central variables image search:
load_dotenv('../.env')

# Azure Computer Vision
key = os.getenv("azure_cv_key")
endpoint = os.getenv("azure_cv_endpoint")
# if endpoint.endswith('/'): endpoint = endpoint[:-1] # remove trailing slash if present


# Azure OpenAI
# api_key = os.getenv('AOAI_API_KEY') # key of your Azure OpenAI resource
# api_base = os.getenv('AOAI_ENDPOINT') # endpoint of your Azure OpenAI resource
# api_version = '2022-08-03-preview' # recommended to check for updates

# openai.api_type = "azure"
# openai.api_version = "2023-05-15" 
# openai.api_base = api_base
# openai.api_key = api_key





def show_images(images, cols=2, source='url', savedir='', show_title=False, titles=None):
    """
    Get images from URL and display them in a grid. Optionally save or retrieve images to/from local dir. 
    
    Parameters
    ----------
    images : list
        List of image urls or local file paths.
    cols : int
        Number of columns in the grid.
    source : str
        'url' or 'local'
    savedir : str
        Directory to save images to.
    show_title : bool
        Display filename as image title (local files only)
    """
    
    if savedir != '':
        os.makedirs(savedir, exist_ok=True)
        
    rows = int(math.ceil(len(images) / cols))

    fig = plt.figure(figsize=(cols * 5, rows * 5)) # specifying the overall grid size. TODO: 7,5 for landscape images

    for i, image_url in enumerate(images):
        plt.subplot(rows, cols,i+1)  
        
        if source == 'url':
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            
            # save images if savedir is specified
            if savedir != '':
                
                # get list of png files
                png_filenames = [image for image in os.listdir(savedir) if image.endswith('.png')]
                # get highest index from existing files
                if png_filenames == []:
                    max_index = 0
                else:
                    max_index = max([int(filename.strip('.png')) for filename in png_filenames])

                # save new file with index + 1
                new_filename = f'{max_index+1:03d}.png'
                fp = os.path.join(savedir, new_filename)
                img.save(fp, 'PNG')            
            
        else: 
            img = Image.open(image_url) # local file
            if show_title:
                if titles is None: plt.title(image_url)
                else: plt.title(titles[i])


        plt.imshow(img)
        plt.axis('off')

    fig.tight_layout()

    plt.show()


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6)) # automatic retry in case of a failing API call
def get_embedding(imagefile):
    """
    Get embedding from an image using Azure Computer Vision 4
    """
    # settings
    model = "?api-version=2023-02-01-preview&modelVersion=latest"
    url = endpoint + "/computervision/retrieval:vectorizeImage" + model
    headers = {
        "Content-type": "application/octet-stream",
        "Ocp-Apim-Subscription-Key": key,
    }

    # Read the image file
    with open(imagefile, "rb") as f:
        data = f.read()

    # Sending the requests
    r = requests.post(url, data=data, headers=headers)
    results = r.json()
    embeddings = results['vector']

    return embeddings


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6)) # automatic retry in case of a failing API call
def get_text_embedding(text):
    """
    Get embedding from text using Azure Computer Vision 4
    """

    # settings
    options = "&features=caption,tags"
    model = "?api-version=2023-02-01-preview&modelVersion=latest"
    url = endpoint + "/computervision/retrieval:vectorizeText" + model # + options
    headers = {
        "Content-type": "application/json",
        "Ocp-Apim-Subscription-Key": key,
    }

    data = {
        "text": text
    }

    # Sending the requests
    r = requests.post(url, data=json.dumps(data), headers=headers)
    results = r.json()
    embeddings = results['vector']

    return embeddings



def get_cosine_similarity(vector1, vector2):
    """
    Calculate cosine similarity of two embeddings vectors
    """
    dot_product = sum(a*b for a, b in zip(vector1, vector2))
    magnitude1 = math.sqrt(sum((val*val) for val in vector1))
    magnitude2 = math.sqrt(sum((val*val) for val in vector2))
    return dot_product / (magnitude1 * magnitude2)    



@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6)) # automatic retry in case of a failing API call
def analyze_image(imagefile, extended_analysis = False):

    service_options = sdk.VisionServiceOptions(endpoint, key)
    vision_source = sdk.VisionSource(filename=imagefile)
    analysis_options = sdk.ImageAnalysisOptions()

    options = sdk.ImageAnalysisFeature.CAPTION | sdk.ImageAnalysisFeature.TAGS 
    if extended_analysis: options = options | sdk.ImageAnalysisFeature.DENSE_CAPTIONS | sdk.ImageAnalysisFeature.OBJECTS
    analysis_options.features = (options)

    image_analyzer = sdk.ImageAnalyzer(service_options, vision_source, analysis_options)
    result = image_analyzer.analyze()

    caption = result.caption.content
    tags_str = ", ".join(tag.name for tag in result.tags)
    
    if extended_analysis:
        dense_captions = ", ".join(caption.content for caption in result.dense_captions)
        objects = ", ".join(obj.name for obj in result.objects)
        return ", ".join([dense_captions, tags_str, objects])
    else:
        return ", ".join([caption, tags_str])


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(30))
def get_openai_embedding(query, embedding_model = 'text-embedding-ada-002'):
    return openai.Embedding.create(input=query, engine=embedding_model)['data'][0]['embedding']



def save_obj_to_pkl(object, filename):
    with open(filename, 'wb') as pickle_out:
        pickle.dump(object, pickle_out)    



@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(7))
def chat_openai(prompt, completion_model, max_output_tokens = 500):


    if not isinstance(prompt, list):
        prompt = [{'role':'user', 'content': prompt}]

    resp = openai.ChatCompletion.create(
            messages=prompt,
            temperature=0.2,
            max_tokens=max_output_tokens,
            engine = completion_model
        )

    return resp["choices"][0]["message"]['content'].strip(" \n")    
