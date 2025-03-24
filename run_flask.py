from flask import Flask, request, jsonify
import base64 
import io 
import time
from PIL import Image 
from ocr_describe import load_model, qwenvl
from ocr import easyocr_langdetect
from collections import OrderedDict

# Flask constructor takes the nae of current module as argument
app = Flask(__name__)

vlprovessor, vlmodel = load_model()

# health check
@app.route('/ping')
def healthy():
    return "pong\n"

# upload base64 image for process
@app.route('/invoke', methods=['POST'])  # Route for POST request
def invoke():
    
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
        description, output_text = qwenvl(image, processor=vlprovessor, model=vlmodel)

        chinese_text = chinese_text if chinese_text!="" else 'null'
        english_text = english_text if english_text!="" else 'null'

        result = OrderedDict({
            "wordsInMark":english_text, 
            "chineseCharacter": chinese_text,          
            "descrOfDevice":description
            })
        
        print(time.time() - start_time)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)