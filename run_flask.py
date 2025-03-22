from flask import Flask, request, jsonify
import base64 
import io 
import time
from PIL import Image 
from ocr_describe_process import qwenvl
from ocr import easyocr_langdetect

# Flask constructor takes the nae of current module as argument
app = Flask(__name__)


# health check
@app.route('/ping')
def healthy():
    return "pong"

# upload base64 image for process
@app.route('/upload', methods=['POST'])  # Route for POST request
def upload():
    
    try:
        start_time = time.time()
        # Get JSON data from the request
        data = request.get_json()

        # Extract information from JSON (assuming it contains a "name" field)
        base64_str = data.get("image", "")

        if not base64_str:
            return jsonify({"error":"No image provided"}), 400
        
        image_byte = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_byte))

        chinese_text, english_text = easyocr_langdetect(image)
        description = qwenvl(image)


        result = {}
        result["chineseCharacter"] = chinese_text if chinese_text!="" else "null"
        result["wordsInMark"] = english_text if english_text!="" else "null"
        result["descrOfDevice"] = description
        
        print(time.time() - start_time)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)