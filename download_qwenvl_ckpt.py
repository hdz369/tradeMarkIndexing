from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch 
import os

ckpt_path = "./local_model_qwenvl"
model_name = "prithivMLmods/Qwen2-VL-OCR-2B-Instruct"

def download_save_ckpt():

    # Check if GPU is available
    device = torch.device("cpu")
    device_map = 'cpu'
    if torch.cuda.is_available(): 
        device = torch.device("cuda")
        device_map = 'auto'

    # default: Load the model on the available device(s)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    # default processer
    processor = AutoProcessor.from_pretrained(model_name)
    # Save model and tokenizer locally
    model.save_pretrained(ckpt_path)
    processor.save_pretrained(ckpt_path)


if __name__ == "__main__":
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)  
        download_save_ckpt()