# tradeMarkIndexing Solution overview

## Objective
The goal of this solution is to depoly an trade mark indexing inference system, which will detect both English and Chinese text in a trade mark image and provide a meaningful description of the image using deep learning techniques. The output should be in json format with properties of 
```json
{
    "wordsInMark": "<english_indexing_results>",
    "chineseCharacter": "<chinese_indexing_results>",
    "descrOfDevice": "<trademark_description>"
}
```

## Design decisions and rationales 
During the solution development, two design approaches were considered, taking into account memory (16G) and time (30s) constraints.
1. **Generate target json output directly using vision language model (VLM )**
2. **Decouple the task into OCR for text detection and Image Description (Captioning) for description**

After extensive trials, a combination of approaches 1 and 2 was chosen as the final solution (OCR+VLM)


## 1. Generate target json output directly using VLM (vision language model)
### Pros
- VLM, as a multimode model, understands both image and NLP making it theoretically possible to fullfill this task with a single model
- Multimode model represents the future trend of AI development and may replace combinations of single-mode models in multi-modality scenarios

### Cons
- It remains chanllenging to precisely control how the model generates output, requiring extensive prompt engineering.
- Larger AI models generally perform better, but memory and time constaints limit the feasible size of VLM, reducing its capability.


## 2.Decouple the task: OCR for text detection and Image Description (Captioning) for description
### Pros
- Breaking a complex task into simpler sub-tasks improves control of output accuracy
- Modern OCR models use deep learning for text detection and recognition, achieving high accuracy
- Extracting Chinese and English Texts from OCR results using fixed rules is more stable than guding a VLM to output structured Chinese and English text in a image

### Cons
- By using multiple models instead of one VLM, we need to be more cautious about the memory usage and time consumption (speed)
- Surprisingly, the current SOTA OCR models and image captioning models underperform compared to VLM in this case
- A suitable image captioning model was not found to accomplish the description task.

## Other trials
Since the trade mark image description generated from qwen2vl were promising, a VLM+VLM approach was attempted: first generating a description, then extracting Chinese and English text from it. However, this approach also failed due to lack of control over the final output.

## Final Solution: OCR+VLM

**Libraries & Tools**
- **Flask** – Http API service 
- **EasyOCR** – Extracting text from trade mark images
- **Qwen2VLForConditionalGeneration** - QWen-VL 2B models to describe trade mark image 
- **PyTorch** – Implementing deep learning models.
- **Hugging Face Transformers** – Using pre-trained captioning models.

other OCR models like trocr was also tested but was abandon due to performance inferiority. 

Experiments on image size resizing has been done to balance between performance and time&memory constraints. 

## Conclusion
By combining OCR text detection and VLM description, this solution enables recognition of textual content and meaningful descriptions of trade mark images, making it useful for automated and consistent indexing. However, due to time and resource constraints, this solution still has much room for further improvement.

## Next Steps:
### Short term
- Evaluate different OCR models for text extraction. If necessary, finetune a model with current domain-specific data
- Analyze edge cases and failure scenarios related to text formatting and refine formatting rules
### Long term
- Further explore VLM model for better control (maybe some supervised finetuning for better instruction following), faster processing speed  and lower RAM consumption (model quantization, dedicated inference framework)


# Installation and Run

## Installation

### Clone this repository
1. Navigate to the home directory using:
```bash 
cd ~
```
If using a different directory, update the `WORKSPACE` variable when run `start_docker_xpu.sh`
2. Clone the repository
```bash 
git clone https://github.com/hdz369/tradeMarkIndexing.git
```
3. Enter the repository directory
```bash
cd tradeMarkIndexing
```

### Download QwenVL model (optional)

Download the file 'model.safetensors' from Hugging Face:[Qwen2-VL-OCR-2B-Instruct] (https://huggingface.co/prithivMLmods/Qwen2-VL-OCR-2B-Instruct/resolve/main) and save it in the following directory 

```bash
cd local_model_qwenvl
wget https://huggingface.co/prithivMLmods/Qwen2-VL-OCR-2B-Instruct/resolve/main/model.safetensors?download=true -O model.safetensors
```

Alternatively, you can bypass this step and download the [Qwen2-VL-OCR-2B-Instruct] model within a Docker container (where environment is ready) by running the script `download_qwenvl_ckpt.py`.

### Build from Dockerfile

Run the following command to build the Docker image:
```bash
bash build_docker.sh
```

## Run the application

Two scripts are available to start a Docker Container:  
- `start_docker_cpu.sh` - for CPU execution
- `start_docker_gpu.sh` - for GPU execution 
Update environment variables if necessary to correctly map the repository directory to the container and run it.

### Download QwenVL model (if not done previously)
Inside the container, run:
```bash
python3 download_qwenvl_ckpt.py
```` 
This download the model weight. Since the host directory is mapped to the container, you only need to download the weight once.

### Start the HTTP service
Run the following command inside the container:
```bash 
python3 run_flask.py
``` 
This starts the HTTP web inference service


# API examples
Please refer to the `health_check()` and `main()` functions in `test_request.py` for API example.