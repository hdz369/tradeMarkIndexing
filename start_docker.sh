#!/bin/bash
export WORKSPACE=$HOME/tradeMarkIndexing
# export DATA=$HOME/huangdezhao/data

docker_image="huangdezhao/trademark:latest"

# docker run -it -w $WORKSPACE -v $WORKSPACE:$WORKSPACE -v $DATA:$DATA -p 8080:8080 $docker_image
docker run -it -w $WORKSPACE -v $WORKSPACE:$WORKSPACE -p 8080:8080 $docker_image