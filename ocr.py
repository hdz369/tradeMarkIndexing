from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import time
import re 
import easyocr 
from langdetect import detect
import torch 

ckpt_easyocr = "local_model_easyocr/model"

def trocr(image):
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-printed')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-printed')

    # image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    # prepare image
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # generate (no beam search)
    generated_ids = model.generate(pixel_values)

    # decode
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def split_compound_lowercase(word):

    # capitalize the first letter
    Word = word[0].upper() + word[1:] 

    # Use regex to split before each capital letter (except the first one)
    words = re.findall(r'[A-Z][a-z]+', Word)
    
    # If no split occurs (meaning it's a single word), return the original input
    if len(words) > 1:
        words.append(word)
    else:
        words = [word]

    words = [x.lower() for x in words]

    return " ".join(words)



def format_wordsInMark(text):
    """
    Format text in wordsInMark field with specified rules.

    Args:
        text: string, input string
    """
    words = text.split()
    words = [split_compound_lowercase(word) for word in words]
    # more formatting rules can be added here below

    return " ".join(words)

def separate_chinese_english(text):
    chinese_chars = re.findall(r'[\u4e00-\u9fff]+', text)  # Match Chinese characters
    english_chars = re.findall(r'[a-zA-Z0-9\s]+', text)    # Match English and numbers

    chinese_text = ''.join(chinese_chars)
    english_text = ''.join(english_chars)

    return chinese_text, english_text     

def detect_language(text):
    try:
        lang = detect(text)
        return lang
    except Exception as e:
        return str(e)
    
def easyocr_langdetect(image):
    # reader = easyocr.Reader(['ch_tra','ch_sim', 'en']) # this needs to run only once to load the model into memory
    reader = easyocr.Reader(['ch_sim','en'], download_enabled=False, model_storage_directory=ckpt_easyocr)

    # start_time = time.time()
    with torch.no_grad():
        result = reader.readtext(image, detail=0)
    # print("with torch no grad: ", time.time() - start_time)  

    # start_time = time.time()
    # result = reader.readtext(image, detail=0)
    # print(time.time() - start_time)
    print("ocr result: ", result)
    chinese_text = ""
    english_text = ""
    for text in result:
        ch, en = separate_chinese_english(text)
        chinese_text += ch
        english_text = english_text + " " + format_wordsInMark(en)

    return chinese_text, english_text



if __name__ == "__main__":
    image = "./dione.jpg"
    # reader = easyocr.Reader(['ch_tra','ch_sim', 'en']) # this needs to run only once to load the model into memory
    # generated_text = trocr(image)
    # print(generated_text)
    # start_time = time.time()
    # print(easyocr_langdetect(image))
    # print(time.time() - start_time)
    x = '习84旦召01) Peony Bright'
    print(format_wordsInMark(x))
    print(detect_language(x))
    print(separate_chinese_english(x))