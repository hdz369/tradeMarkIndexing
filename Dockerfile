FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN apt-get -y update && apt-get install -y \
    software-properties-common \
    build-essential \
    checkinstall \
    cmake \
    pkg-config \
    yasm \
    git \
    vim \
    curl \
    wget \
    gfortran \
    sudo \
    apt-transport-https \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    dbus-x11 \
    vlc \
    iputils-ping \
    python3-dev \
    python3-pip
    
RUN pip3 install --no-cache-dir --upgrade pip 

WORKDIR /home/TradeMarkIndexing

COPY ./requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY ./run_flask.py ./run_flask.py
COPY ./qwenvlm.py ./qwenvlm.py
COPY ./ocr.py ./ocr.py
COPY ocr_describe.py ./ocr_describe.py



