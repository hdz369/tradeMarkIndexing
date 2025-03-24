# tradeMarkIndexing

## Installation

### Clone this repo to local directory (preferably local home directory)
- Go to home directory by `cd ~`, otherwise, you will need to modify the WORKSPACE variable when run `start_docker.sh`
- Run `git clone https://github.com/hdz369/tradeMarkIndexing.git`

Once cloning is complete, change into the repository directory:
- Run `cd tradeMarkIndexing`

### Download QwenVL model (optional)

manually download the file 'model.safetensors' from Huggingface website: https://huggingface.co/prithivMLmods/Qwen2-VL-OCR-2B-Instruct/tree/main and save under directory `tradeMarkIndexing/local_model_qwenvl` 

If you do not do this step, you will need to download QwenVL model after you enter the docker container in later steps.

### Build from Dockerfile

- Run `bash build_docker.sh` to build docker. 

## Run

There are two shell script to initiate a docker container: `start_docker_cpu.sh` for running with CPU and `start_docker_gpu.sh` for running with GPU. Edit environment variables if necessary to map the correct repo directory to container and run it.
Once in the container:

### Download QwenVL model (if you have not done in previous step after cloning the repository)
- Run `python3 download_qwenvl_ckpt.py` to download the weight. Since you are mapping host directory to container, you only need to download the weight once.

### Start the http service
- Run `python3 run_flask.py` to start the http service