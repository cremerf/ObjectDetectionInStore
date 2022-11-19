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
# RUN useradd -ms /bin/bash app
# USER app

# Setup some paths
ENV PYTHONPATH=/home/src/.local/lib/python3.8/site-packages:/home/src
ENV PATH=$PATH:/home/src/.local/bin

# Install the python packages for this new user
ADD requirements.txt .
RUN pip3 install -r requirements.txt

# PyTorch GPU 1.10
RUN pip3 install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

## ------
# Create working directory
WORKDIR /home/src

# Clone yolov5
RUN git clone https://github.com/ultralytics/yolov5 /home/src/yolov5

## -------------------------------------------------------------------------------------------------------

# # Start FROM NVIDIA PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
# FROM nvcr.io/nvidia/pytorch:22.04-py3
# RUN rm -rf /opt/pytorch  # remove 1.2GB dir

# # Downloads to user config dir
# ADD https://ultralytics.com/assets/Arial.ttf https://ultralytics.com/assets/Arial.Unicode.ttf /root/.config/Ultralytics/

# # Install linux packages
# RUN apt update && apt install --no-install-recommends -y zip htop screen libgl1-mesa-glx

# # Install pip packages
# COPY requirements.txt .
# RUN python -m pip install --upgrade pip wheel
# RUN pip uninstall -y Pillow torchtext  # torch torchvision
# RUN pip install --no-cache -r requirements.txt ultralytics albumentations comet gsutil notebook Pillow>=9.1.0 \
#     'opencv-python<4.6.0.66' \
#     --extra-index-url https://download.pytorch.org/whl/cu113

# # Create working directory
# RUN mkdir -p /home/src/app
# WORKDIR /home/src/app

# # Copy contents
# # COPY . /usr/src/app  (issues as not a .git directory)
# # git clone https://github.com/ultralytics/yolov5 /home/src/yolov5

# # Set environment variables
# ENV OMP_NUM_THREADS=8
# Update