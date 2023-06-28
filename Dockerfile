FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV NV_CUDNN_VERSION 8.6.0.163
ENV NV_CUDNN_PACKAGE_NAME "libcudnn8"
ENV NCCL_SOCKET_IFNAME "lo"
ENV KMP_DUPLICATE_LIB_OK True

ENV NV_CUDNN_PACKAGE "$NV_CUDNN_PACKAGE_NAME=$NV_CUDNN_VERSION-1+cuda11.8"
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ${NV_CUDNN_PACKAGE} \
    && apt-mark hold ${NV_CUDNN_PACKAGE_NAME} \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update -y \
    && apt-get install -y python3-pip \
    && apt-get install -y git-all \
    && apt-get install -y wget unzip p7zip-full libglib2.0-0 ffmpeg

RUN echo 'alias python=python3' >> ~/.bashrc

WORKDIR /app
COPY requirements.txt requirements.txt

# Activate conda environment for bash
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install ninja triton xformers
RUN pip install -r requirements.txt
RUN pip install git+https://github.com/elliottzheng/face-detection.git@master

ENTRYPOINT [ "bash" ]
