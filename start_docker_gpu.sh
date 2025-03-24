#!/bin/bash
export WORKSPACE=$HOME/tradeMarkIndexing

# export DATA=$HOME/huangdezhao/data

docker_image="huangdezhao/trademark:latest"

# docker run -it -w $WORKSPACE -v $(pwd):$WORKSPACE -p 8080:8080 $docker_image     # for cpu usage
docker run -it --gpus all -w $WORKSPACE -v $(pwd):$WORKSPACE -p 8080:8080 $docker_image   # for gpu usage