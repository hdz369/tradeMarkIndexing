import requests
import base64 

def health_check():
    url = "http://127.0.0.1:8080/ping"
    response = requests.get(url)

    print(response)    


def main(image_path):
    url = "http://127.0.0.1:8080/upload"
    with open(image_path, 'rb') as image_file:
        # Read the image and encode it to base64
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    headers={
        'User-Agent': 'python-requests/2.31.0',
        'Accept': '*/*',
        'Connection': 'keep-alive',
    }
        
    # Step 2: Create the payload with the base64-encoded image
    payload = {
        'image': encoded_image  # You can add more fields here if necessary
    }
    # Step 4: Send the POST request with the base64 image
    response = requests.post(url, headers=headers, json=payload)

    print(response.json())

if __name__ == "__main__":
    health_check()
    main('dione.jpg')