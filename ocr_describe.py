from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch 
import time

ckpt_path = "./local_model_qwenvl"


def qwenvl(image):
    # Check if GPU is available
    device = torch.device("cpu")
    device_map = 'cpu'
    if torch.cuda.is_available(): 
        device = torch.device("cuda")
        device_map = 'auto'

    # # default: Load the model on the available device(s)
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     "prithivMLmods/Qwen2-VL-OCR-2B-Instruct",
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",
    # )
    # # default processer
    # processor = AutoProcessor.from_pretrained("prithivMLmods/Qwen2-VL-OCR-2B-Instruct")
    # # Save model and tokenizer locally
    # model.save_pretrained("./local_models")
    # processor.save_pretrained("./local_models")


    # default: Load the model on the available device(s)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    # default processer
    processor = AutoProcessor.from_pretrained(ckpt_path)


    model.to(device)


    # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
    # prompt = """
    #             Please analyze the image and perform the following tasks:

    #             1. **Detect any Chinese text** in the image and return it. If there is no Chinese text, return `null`.
    #             2. **Detect any English text** in the image and return it. If there is no English text, return `null`.
    #             3. Provide a **description of the image**, including any notable features or objects present.
    #             Return the output in the following format:
    #             {
    #             "Chinese_Text": "Detected Chinese text or null if not detected",
    #             "English_Text": "Detected English text or null if not detected",
    #             "Description": "A description of the image."
    #             }
    #             """
    # prompt = """
    #             Please analyze the image and perform the following tasks:
    #             1. **Detect any Chinese text** in the image and return it. If there is no Chinese text, return `null`.
    #             2. **Detect any English text** in the image and return it. If there is no English text, return `null`.
    #             3. Provide a **description of the image**, including any notable features or objects present.
    #             """
    prompt = """ Provide a concise description of the image, including any notable features or objects present. 
    #             """
    # prompt = """ extract the key informatoin in the json format """
    # prompt = """extract the key informatoin in json format. and then describe this image """

    messages = [
        {
            "role": "system", "content": "You are a  assistant that detects text in the image.",  # add system role
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                    # "image": "dione.jpg",
                    "resized_height": 200,
                    "resized_width": 200,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # Inference: Generation of the output

    # start_time = time.time()
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    # print(time.time() - start_time)   

    # start_time = time.time()
    # with torch.no_grad():
    #     generated_ids = model.generate(**inputs, max_new_tokens=128)
    # print(time.time() - start_time)  


    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    # print(output_text)

    




    return output_text.strip("<|im_end|>")


if __name__ == "__main__":
    from PIL import Image
    image = Image.open('dione.jpg')
    start_time = time.time()
    print(qwenvl(image))
    print(time.time() - start_time)

