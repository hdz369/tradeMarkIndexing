from transformers import AutoProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

ckpt_path = "./local_model_blip"


def blip_img_cap(image):
    # Load model and processor
    model_name = "Salesforce/blip-image-captioning-base"
    # processor = AutoProcessor.from_pretrained(model_name)
    # model = BlipForConditionalGeneration.from_pretrained(model_name)
    # # # Save model and tokenizer locally
    # model.save_pretrained(ckpt_path)
    # processor.save_pretrained(ckpt_path)

    processor = AutoProcessor.from_pretrained(ckpt_path)
    model = BlipForConditionalGeneration.from_pretrained(ckpt_path)
    # Process input
    # text = "This is a trademark: "
    text = ""
    inputs = processor(image, text, return_tensors="pt")

    # Generate caption
    
    caption_ids = model.generate(**inputs)

    # Decode caption
    caption = processor.batch_decode(caption_ids, skip_special_tokens=True)
    print(caption)  # Output: ['A cat sitting on a windowsill']

# Load image
image = Image.open("dione.jpg")
blip_img_cap(image)
