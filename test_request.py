from flask import Flask, request, jsonify
import requests
import base64 
import io 
from PIL import Image 


def main(image_path):
    url = "http://localhost:8080/upload"
    with open(image_path, 'rb') as image_file:
        # Read the image and encode it to base64
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    # Step 2: Create the payload with the base64-encoded image
    payload = {
        'image': encoded_image  # You can add more fields here if necessary
    }
    # Step 4: Send the POST request with the base64 image
    response = requests.post(url, json=payload)

    print(response.json())

if __name__ == "__main__":
    main('dione.jpg')