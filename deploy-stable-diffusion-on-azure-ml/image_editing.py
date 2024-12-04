"""
For instructions how to use the code, please reference this blog: 
https://techcommunity.microsoft.com/blog/machinelearningblog/image-editing-using-azure-openai-and-azure-machine-learning/4339077
"""

import os
import requests
import base64
import numpy as np
import cv2
import sys
import shutil
import urllib.request
import json
import ssl

gpt4o_API_key = "xxx"  #swedencentral
gpt4o_ENDPOINT = "https://xxx"
StableDiffusion_url = 'xxx'  # workspace-centralus
StableDiffusion_api_key = 'xxx'  # Replace with your API key

if len(sys.argv) < 4:
    print("Usage: python script.py <input_image_path> <mask_image_path> <final_image_path>")
    sys.exit(1)

def create_mask_using_prompt(input_image_path, output_image_path):
    # Configuration
    API_KEY = gpt4o_API_key
    ENDPOINT = gpt4o_ENDPOINT
    encoded_image = base64.b64encode(open(input_image_path, 'rb').read()).decode('ascii')
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY,
    }

    # User prompt for the first request
    user_prompt = input("Please enter your prompt, start with - image size x * y pixel, x axis left to right, y axis top down, and tell what you want: ")

    # Payload for the first request
    payload = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an AI assistant that helps people find information."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "\n"
                    }
                ]
            }
        ],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 4096
    }

    # Send the first request
    response = requests.post(ENDPOINT, headers=headers, json=payload)
    response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code

    # Handle the response from the first request
    result = response.json()
    first_message = result['choices'][0]['message']['content']

    # Display the first message to the user
    print(f"Here is the message: {first_message}\n")

    # Loop for the second prompt
    while True:
        # User prompt for the second request to convert the message to numpy array
        second_prompt = input("Please enter your prompt to provide the answer in numpy array format [[x,y],[x,y]...[x,y]]: ")

        # Payload for the second request
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are an AI assistant that helps people find information."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": second_prompt
                        },
                        {
                            "type": "text",
                            "text": first_message
                        }
                    ]
                }
            ],
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 4096
        }

        # Send the second request
        response = requests.post(ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code

        # Handle the response from the second request
        result = response.json()
        numpy_array_message = result['choices'][0]['message']['content']

        # Display the numpy array message
        print(numpy_array_message)

        # Ask the user if the response is correct
        user_feedback = input("Is this correct? (yes/no): ").strip().lower()
        if user_feedback == 'yes':
            break

    # Find the start and end of the square brackets
    start_idx = numpy_array_message.find('[')
    end_idx = numpy_array_message.rfind(']') + 1

    # Extract the part of the string that contains the array
    coordinates_str = numpy_array_message[start_idx:end_idx]

    coordinates_str = coordinates_str.replace(' ', '')
    coordinates = np.array([list(map(int, point.split(','))) for point in coordinates_str.strip('[]').split('],[')])

    # Display the numpy array
    print(coordinates)

    # Generate mask using the coordinates
    img = cv2.imread(input_image_path)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [coordinates], (255, 255, 255))
    cv2.imwrite(output_image_path, mask)
    print(f"Mask generated and saved as '{output_image_path}'.")

def create_mask_by_mouse_click(input_image_path, output_image_path):
    # Mouse callback function to draw contours and straight lines
    def draw_contour(event, x, y, flags, param):
        global drawing, points
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            points.append((x, y))
            print(f"Point added: {points[-1]}")
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                points.append((x, y))
                cv2.circle(img, (x, y), 2, (255, 255, 255), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            points.append((x, y))
            cv2.polylines(img, [np.array(points)], False, (255, 255, 255), 2)
            print("Contour drawn.")
        elif event == cv2.EVENT_RBUTTONDOWN:
            drawing = False
            if points:
                cv2.polylines(img, [np.array(points)], False, (255, 255, 255), 2)
                points = []
                print("Contour ended.")

    # Initialize drawing state and points list
    global drawing, points
    drawing = False
    points = []

    # Load an existing image
    img = cv2.imread(input_image_path)

    if img is None:
        print("Error: Could not load image.")
        return

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_contour)

    while True:
        cv2.imshow('image', img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Press 'ESC' to exit
            break
        elif key == ord('q'):  # Press 'q' to quit drawing mode
            break
        elif key == 13:  # Press 'Enter' to finalize the polygon
            if points:
                cv2.fillPoly(img, [np.array(points)], (255, 255, 255))
                mask = np.zeros_like(img)
                cv2.fillPoly(mask, [np.array(points)], (255, 255, 255))
                cv2.imwrite(output_image_path, mask)
                print(f"Polygon finalized and mask saved as '{output_image_path}'.")
                points = []
                cv2.destroyAllWindows()  # Close the image window automatically
            break

    cv2.destroyAllWindows()

def reverse_mask(input_image_path, output_image_path):
    # Read the mask image
    mask_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    if mask_image is None:
        print("Error: Could not load image.")
        return

    # Invert the mask image
    inverted_mask = cv2.bitwise_not(mask_image)

    # Save the inverted mask image
    cv2.imwrite(output_image_path, inverted_mask)

    # Display the original and inverted mask images
    cv2.imshow('Original Mask', mask_image)
    cv2.imshow('Inverted Mask', inverted_mask)
    cv2.waitKey(5000)  # Display for 5000 milliseconds (5 seconds)
    cv2.destroyAllWindows()

def copy_existing_mask(input_mask_path, output_image_path):
    try:
        if os.path.abspath(input_mask_path) == os.path.abspath(output_image_path):
            print(f"Mask already exists as '{output_image_path}'. No need to copy.")
        else:
            shutil.copy(input_mask_path, output_image_path)
            print(f"Mask copied to '{output_image_path}'.")
    except IOError as e:
        print(f"Unable to copy file. Use the file with matching name. {e}")

def create_mask_using_points(input_image_path, output_image_path, numpy_array_message):
    # Find the start and end of the square brackets
    start_idx = numpy_array_message.find('[')
    end_idx = numpy_array_message.rfind(']') + 1

    # Extract the part of the string that contains the array
    coordinates_str = numpy_array_message[start_idx:end_idx]

    # Remove extra spaces and split the string into coordinate pairs
    coordinates_str = coordinates_str.replace(' ', '')
    coordinates = np.array([list(map(int, point.split(','))) for point in coordinates_str.strip('[]').split('],[')])

    # Display the numpy array
    print(coordinates)

    # Generate mask using the coordinates
    img = cv2.imread(input_image_path)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [coordinates], (255, 255, 255))
    cv2.imwrite(output_image_path, mask)
    print(f"Mask generated and saved as '{output_image_path}'.")


# Main function to choose the method
def mask_generation():
    input_image_path = sys.argv[1]
    output_image_path = sys.argv[2]

    print("Choose an option:")
    print("1. Use prompt to generate mask")
    print("2. Use mouse click to generate mask")
    print("3. Reverse an existing foreground matting")
    print("4. Bring your own mask")
    print("5. Create mask with points coordinates")
    choice = input("Enter 1, 2, 3, 4 or 5: ")

    if choice == '1':
        create_mask_using_prompt(input_image_path, output_image_path)
    elif choice == '2':
        create_mask_by_mouse_click(input_image_path, output_image_path)
    elif choice == '3':
        mask_file_name = input("Please enter the mask file name: ")
        reverse_mask(mask_file_name, output_image_path)
    elif choice == '4':
        mask_file_name = input("Please enter the mask file name: ")
        copy_existing_mask(mask_file_name, output_image_path)
    elif choice == '5':
        input_array = input("please provide numpy array of all points coordinates in the format of [[x,y],[x,y] ...[x,y]]\n")
        create_mask_using_points(input_image_path, output_image_path, input_array)
    else:
        print("Invalid choice. Please enter 1, 2, 3, or 4.")


# Additional functionality to execute after calling mask_generation()
def allowSelfSignedHttps(allowed):
    # Bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True)  # This line is needed if you use self-signed certificate in your scoring service.

# Function to encode image to base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to edit the image
def edit_image(input_image, mask_image, output_image):
    # Check if the output image path exists, remove it if it does
    if os.path.exists(output_image):
        os.remove(output_image)

    # Print message and prompt user for input
    user_prompt = input("Please enter your prompt: ")

    # Read the original and mask images
    original_image = cv2.imread(input_image)
    mask_image = cv2.imread(mask_image)

    # Resize both images to the same size
    resized_size = (512, 512)  # Example size, adjust as needed
    original_resized = cv2.resize(original_image, resized_size)
    mask_resized = cv2.resize(mask_image, resized_size)

    # Save the resized images
    cv2.imwrite('original_resized.png', original_resized)
    cv2.imwrite('mask_resized.png', mask_resized)

    # Encode the images
    with open('original_resized.png', "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    with open('mask_resized.png', "rb") as image_file:
        mask_image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    # Request data
    data = {
        "input_data": {
            "data": {
                "prompt": user_prompt,
                "image": image_base64,
                "mask_image": mask_image_base64
            },
            "columns": [
                "prompt",
                "image",
                "mask_image"
            ],
            "index": [
                0
            ]
        }
    }

    body = str.encode(json.dumps(data))

    # Set the headers
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {StableDiffusion_api_key}'
    }

    # Make the request
    req = urllib.request.Request(StableDiffusion_url, body, headers)
    #print(req)

    try:
        response = urllib.request.urlopen(req)
        print("Response Status Code:", response.getcode())
        #print("Response Headers:", response.info())
        result = response.read()
        result_json = json.loads(result)

        fields = set()
        for item in result_json:
            fields.update(item.keys())

        #print(list(fields))
        
        # Assuming the response contains the base64-encoded image
        generated_image_base64 = result_json[0]['generated_image']  # Adjust this key based on your response structure
        generated_image = base64.b64decode(generated_image_base64)
        
        # Save the generated image
        with open(output_image, 'wb') as f:
            f.write(generated_image)
        print(f"Generated image saved as '{output_image}'")

        # Display the generated image
        generated_image_cv2 = cv2.imread(output_image)
        cv2.imshow('Generated Image', generated_image_cv2)
        cv2.waitKey(3000)  # Display for 3000 milliseconds (3 seconds)
        cv2.destroyAllWindows()

    except urllib.error.HTTPError as error:
        print(f"Request failed with status code: {error.code}")
        print(error.read().decode())
    except KeyError as e:
        print(f"KeyError: {e}. Please check the response structure.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Call the mask_generation() function
print("In order to edit an existing image, you need to have a mask image.")
mask_generation()

# Loop to ask user if they are satisfied with the image
while True:
    # Print message and prompt user for input
    print("Mask is generated, now input your prompt to edit the image.")
    edit_image(sys.argv[1], sys.argv[2], sys.argv[3])

    satisfied = input("Are you done with editing? (yes/no): ").strip().lower()
    if satisfied == 'yes':
        break

    use_previous_mask = input("Do you want to use the previous mask? (yes/no): ").strip().lower()
    if use_previous_mask == 'no':
        mask_generation()
