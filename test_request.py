import requests
import base64 

def health_check():
    url = "http://127.0.0.1:8080/ping"
    response = requests.get(url)

    print(response)    

def image_url_to_base64(url):
    response = requests.get(url)
    if response.status_code == 200:
        encoded_image = base64.b64encode(response.content).decode('utf-8')
        return encoded_image
    else:
        raise Exception(f"Failed to fetch image: {response.status_code}")

def image_path_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        # Read the image and encode it to base64
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_image
    
def main(encoded_image):
    url = "http://127.0.0.1:8080/invoke"

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


def retrieve_tradeMarks(lodgement_date):
    url = "https://api.data.gov.sg/v1/technology/ipos/trademarks?lodgement_date="
    response = requests.get(url+lodgement_date).json()
    print(response.keys())
    print("count", response['count'])
    items = response['items']
    for item in items:
        markindex = item['markIndex']
        # print(item['documents'][0].keys())
        url = item['documents'][0]['url']

        print(markindex)
        print(url)
        print(main(image_url_to_base64(url)))

     

if __name__ == "__main__":
    # health_check()
    # main(image_path_to_base64('siogoodChinese.jpg'))
    retrieve_tradeMarks("2018-11-23")

