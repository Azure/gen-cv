import logging
import os
import json
import requests
from datetime import datetime, timedelta
import pyodbc

import azure.functions as func

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
        "name": "get_bonus_points",
        "description": "Check the amount of customer bonus / loyalty points",
        "parameters": {
            "type": "object",
            "properties": {
                "account_id": {
                    "type": "number",
                    "description": "Four digit account number (i.e., 1005, 2345, etc.)"
                },
            },
            "required": ["account_id"],
        }
    },
    {
        "name": "get_order_details",
        "description": "Check customer account for expected delivery date of existing orders based on the provided parameters",
        "parameters": {
            "type": "object",
            "properties": {
                "account_id": {
                    "type": "number",
                    "description": "Four digit account number (i.e., 1005, 2345, etc.)"
                },
            },
            "required": ["account_id"],
        }
    },
    {
        "name": "order_product",
        "description": "Order a product based on the provided parameters",
        "parameters": {
            "type": "object",
            "properties": {
                "account_id": {
                    "type": "number",
                    "description": "Four digit account number (i.e., 1005, 2345, etc.)"
                },
                "product_name": {
                    "type": "string",
                    "description": "Name of the product to order (i.e., Elysian Voyager, Terra Roamer, AceMaster 3000, Server & Style)"
                },
                "quantity": {
                    "type": "number",
                    "description": "Quantity of the product to order (i.e., 1, 2, etc.)"
                }
            },
            "required": ["account_id", "product_name", "quantity"],
        }
    },
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

    response = chat_complete(messages, functions= functions, function_call= "auto")

    products = []
    
    try:
        response_message = response["choices"][0]["message"]
    except:
        logging.info(response)

    # if the model wants to call a function
    if response_message.get("function_call"):
        # Call the function. The JSON response may not always be valid so make sure to handle errors
        function_name = response_message["function_call"]["name"]

        available_functions = {
                "get_bonus_points": get_bonus_points,
                "get_order_details": get_order_details,
                "order_product": order_product,
                "get_product_information": get_product_information,
        }
        function_to_call = available_functions[function_name] 

        function_args = json.loads(response_message["function_call"]["arguments"])
        function_response = function_to_call(**function_args)
        # print(function_name, function_args)

        # Add the assistant response and function response to the messages
        messages.append({
            "role": response_message["role"],
            "function_call": {
                "name": function_name,
                "arguments": response_message["function_call"]["arguments"],
            },
            "content": None
        })

        if function_to_call == get_product_information:
            product_info = json.loads(function_response)
            # show product information after search for a different product that the current one
            # if product_info['product_image_file'] != current_product_image:
                
            products = [display_product_info(product_info)]
            current_product_image = product_info['product_image_file']
            
            # return only product description to LLM to avoid chatting about prices and image files 
            function_response = product_info['description']

        messages.append({
            "role": "function",
            "name": function_name,
            "content": function_response,
        })
     
        response = chat_complete(messages, functions= functions, function_call= "none")
        
        response_message = response["choices"][0]["message"]

    messages.append({'role' : response_message['role'], 'content' : response_message['content']})

    logging.info(json.dumps(response_message))

    response_object = {
        "messages": messages,
        "products": products
    }

    return func.HttpResponse(
        json.dumps(response_object),
        status_code=200
    )

def execute_sql_query(query, connection_string=database_connection_string, params=None):
    """Execute a SQL query and return the results."""
    results = []
    print('database_connection_string', database_connection_string)
    
    # Establish the connection
    with pyodbc.connect(connection_string) as conn:
        cursor = conn.cursor()
        
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        # If the query is a SELECT statement, fetch results
        if query.strip().upper().startswith('SELECT'):
            results = cursor.fetchall()
        
        conn.commit()

    return results

def get_bonus_points(account_id):
    """Retrieve bonus points and its cash value for a given account ID."""
     
    # Define the SQL query to retrieve loyalty_points for the given account_id
    query = "SELECT loyalty_points FROM Customers WHERE account_id = ?"

    # Execute the query with account_id as a parameter
    results = execute_sql_query(query, params=(account_id,))

    # If results are empty, return an error message in JSON format
    if not results:
        return json.dumps({"error": "Account not found"})

    # Get the loyalty_points value
    loyalty_points = results[0][0]

    # Convert loyalty_points to cash_value
    cash_value = loyalty_points / 9.5

    # Create a JSON object with the required keys and values
    response_json = json.dumps({
        "available_bonus_points": loyalty_points,
        "cash_value": cash_value
    })

    return response_json


def get_order_details(account_id):
     
    # Get orders and corresponding product names for the account_id
    query = '''
        SELECT o.order_id, p.name as product_name, o.days_to_delivery
        FROM Orders o
        JOIN Products p ON o.product_id = p.id
        WHERE o.account_id = ?
    '''
    orders = execute_sql_query(query, params=(account_id,))
    
    # Get today's date and calculate the expected delivery date for each order
    today = datetime.today()
    
    # Create a JSON object with the required details
    order_details = [
        {
            "product_name": order.product_name,
            "expected_delivery_date": (today + timedelta(days=order.days_to_delivery)).strftime('%Y-%m-%d')
        }
        for order in orders
    ]
    
    # Return the JSON object
    return json.dumps(order_details)

def order_product(account_id, product_name, quantity=1):
     
    # Step 1: Find the maximum existing order_id
    query = "SELECT MAX(order_id) FROM Orders"
    results = execute_sql_query(query)
    max_order_id = results[0][0] if results[0][0] is not None else 0

    # Step 2 & 3: Find product ID and check stock
    query = "SELECT id, name, stock FROM Products WHERE LOWER(name) LIKE LOWER(?)"
    params = (f'%{product_name}%',)
    results = execute_sql_query(query, params=params)
    
    # Handling no match found
    if not results:
        return json.dumps({"info": "No matching product found"})
    
    product_id, product_name_corrected, stock = results[0]
    
    # Check if the stock is sufficient
    if stock < quantity:
        return json.dumps({"info": "Insufficient stock"})
    
    # Step 4: Place the order
    # Deducting the ordered quantity from the stock
    query = "UPDATE Products SET stock = stock - ? WHERE id = ?"
    params = (quantity, product_id)
    if place_orders: execute_sql_query(query, params=params)

    # Adding the order details to the Orders table
    days_to_delivery = 5
    for i in range(quantity):
        max_order_id += 1
        query = "INSERT INTO Orders (order_id, product_id, days_to_delivery, account_id) VALUES (?, ?, ?, ?)"
        params = (max_order_id, product_id, days_to_delivery, account_id)
        if place_orders: execute_sql_query(query, params=params)
    
    # Step 5: Calculate the expected delivery date and return the JSON object
    today = datetime.now()
    expected_delivery_date = today + timedelta(days=days_to_delivery)
    
    return json.dumps({
        "info": "Order placed",
        "product_name": product_name_corrected,
        "expected_delivery_date": expected_delivery_date.strftime('%Y-%m-%d')
    })

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

def chat_complete(messages, functions, function_call='auto'):
    """  Return assistant chat response based on user query. Assumes existing list of messages """
    
    url = f"{AOAI_endpoint}/openai/deployments/{chat_deployment}/chat/completions?api-version={AOAI_api_version}"

    headers = {
        "Content-Type": "application/json",
        "api-key": AOAI_key
    }

    data = {
        "messages": messages,
        "functions": functions,
        "function_call": function_call,
        "temperature" : 0,
    }

    response = requests.post(url, headers=headers, data=json.dumps(data)).json()

    return response
