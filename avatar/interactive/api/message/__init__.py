import logging
import os
import json
import requests
from datetime import datetime, timedelta
# import pyodbc

import azure.functions as func
import openai
import re

search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_key = os.getenv("AZURE_SEARCH_API_KEY") 
search_api_version = '2023-07-01-Preview'
search_index_name = os.getenv("AZURE_SEARCH_INDEX")

AOAI_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
AOAI_key = os.getenv("AZURE_OPENAI_API_KEY")
AOAI_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
embeddings_deployment = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

sql_db_server = os.getenv("SQL_DB_SERVER")
sql_db_user = os.getenv("SQL_DB_USER")
sql_db_password = os.getenv("SQL_DB_PASSWORD")
sql_db_name = os.getenv("SQL_DB_NAME")

blob_sas_url = os.getenv("BLOB_SAS_URL")

server_connection_string = f"Driver={{ODBC Driver 17 for SQL Server}};Server=tcp:{sql_db_server},1433;Uid={sql_db_user};Pwd={sql_db_password};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
database_connection_string = server_connection_string + f"Database={sql_db_name};"

# font color adjustments
blue, end_blue = '\033[36m', '\033[0m'

place_orders = False

functions = [
    {
    "name": "get_product_information",
    "description": "Find information about a product based on a user question. Use only if the requested information if not already available in the conversation context.",
    "parameters": {
        "type": "object",
        "properties": {
            "user_question": {
                "type": "string",
                "description": "User question (i.e., do you have tennis shoes for men?, etc.)"
            },
        },
        "required": ["user_question"],
        }
    }
]

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    messages = json.loads(req.get_body())

    response = chat_complete_data(messages)

    products = []
    
    try:
        response_message = response.choices[0].message
    except:
        logging.info(response)

    response_message.content = re.sub(r'\s*\[doc\d+\]', '', response_message.content)

    messages.append({'role' : response_message.role, 'content' : response_message.content})

    logging.info(response_message.content)

    response_object = {
        "messages": messages,
        "products": products
    }

    return func.HttpResponse(
        json.dumps(response_object),
        status_code=200
    )


def display_product_info(product_info, display_size=40):
    """ Display product information """

    # Show image
    image_file = product_info['product_image_file']

    image_url = blob_sas_url.split("?")[0] + f"/{image_file}?" + blob_sas_url.split("?")[1]

    response = requests.get(image_url)
    print(image_url)

    # Check if the request was successful
    if response.status_code == 200:
        return {
            "tagline": product_info['tagline'],
            "original_price": product_info['original_price'],
            "special_offer": product_info['special_offer'],
            "image_url": image_url 
            }
    else:
        print(f"Failed to retrieve image. HTTP Status code: {response.status_code}")

    print(f"""
    {product_info['tagline']}
    Original price: ${product_info['original_price']} Special offer: ${product_info['special_offer']} 
    """)
    
def generate_embeddings(text):
    """ Generate embeddings for an input string using embeddings API """

    url = f"{AOAI_endpoint}/openai/deployments/{embeddings_deployment}/embeddings?api-version={AOAI_api_version}"

    headers = {
        "Content-Type": "application/json",
        "api-key": AOAI_key,
    }

    data = {"input": text}

    response = requests.post(url, headers=headers, data=json.dumps(data)).json()
    return response['data'][0]['embedding']

def get_product_information(user_question, categories='*', top_k=1):
    """ Vectorize user query to search Cognitive Search vector search on index_name. Optional filter on categories field. """
     
    url = f"{search_endpoint}/indexes/{search_index_name}/docs/search?api-version={search_api_version}"

    headers = {
        "Content-Type": "application/json",
        "api-key": f"{search_key}",
    }
    
    vector = generate_embeddings(user_question)

    data = {
        "vectors": [
            {
                "value": vector,
                "fields": "description_vector",
                "k": top_k
            },
        ],
        "select": "tagline, description, original_price, special_offer, product_image_file",
    }

    # optional filtered search
    if categories != '*':
        data["filter"] = f"category eq '{categories}'"

    results = requests.post(url, headers=headers, data=json.dumps(data))    
    results_json = results.json()
    
    # Extracting the required fields from the results JSON
    product_data = results_json['value'][0] # hard limit to top result for now

    response_data = {
        "tagline": product_data.get('tagline'),
        "description": product_data.get('description'),
        "original_price": product_data.get('original_price'),
        "special_offer": product_data.get('special_offer'),
        "product_image_file": product_data.get('product_image_file'),
    }
    return json.dumps(response_data)

def chat_complete_data(messages):
    client = openai.AzureOpenAI(
    azure_endpoint=AOAI_endpoint,
    api_key=AOAI_key,
    api_version="2024-02-01",
    )

    completion = client.chat.completions.create(
        model=chat_deployment,
        messages=messages,
        extra_body={
            "data_sources":[
                {
                    "type": "azure_search",
                    "parameters": {
                        "endpoint": search_endpoint,
                        "index_name": search_index_name,
                        "authentication": {
                            "type": "api_key",
                            "key": search_key,
                        }
                    }
                }
            ],
        }
    )

    return completion