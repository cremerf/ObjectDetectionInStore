FROM nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu20.04 as base

# Install some packages
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.8-dev \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    vim \
    wget \
    curl

RUN apt-get -y update
RUN apt-get -y install git

# Add a non-root user
#RUN useradd -ms /bin/bash app
#USER app

# Setup some paths
# Setup some paths
ENV PYTHONPATH=/home/app/.local/lib/python3.8/site-packages:/home
ENV PATH=$PATH:/home/app/.local/bin

# Install the python packages for this new user
ADD requirements.txt .
RUN pip3 install -r requirements.txt

# PyTorch GPU 1.10
RUN pip3 install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

# Copy info inside 
ADD . /home

# Create working directory
WORKDIR /home

# Avoid problems with yolov5 clone
# RUN git config --global --add safe.directory /home/src/yolov5

# Clone yolov5
# RUN git clone https://github.com/ultralytics/yolov5 /src/yolov5

# FROM base as test
# RUN ["pytest", "-v", "/src/tests"]

FROM base as build
CMD ["python3", "/home/ml_service.py"]

## -------------------------------------------------------------------------------------------------------