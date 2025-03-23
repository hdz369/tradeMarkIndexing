# tradeMarkIndexing

## Installation

### Clone this repo to local directory (preferably local home directory)
- Go to home directory by `cd ~`, otherwise, you will need to modify the WORKSPACE variable when run `start_docker.sh`
- Run `git clone https://github.com/hdz369/tradeMarkIndexing.git`

Once cloning is complete, change into the repository directory:
- Run `cd tradeMarkIndexing`

### Download QwenVL model from Huggingface
- Run `python3 download_qwenvl_ckpt.py` to download model weights to local directory

### Build from Dockerfile

- Run `bash build_docker.sh` to build docker. 

## Run

Edit environment variables in `start_docker.sh` and run it.
Once in the container
- Run `python3 run_flask.py` to start the http service