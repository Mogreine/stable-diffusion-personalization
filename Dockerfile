FROM python:3.10.6

WORKDIR /app

COPY requirements.txt requirements.txt

# Install other requirements
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install ffmpeg libsm6 libxext6 -y

# Activate conda environment for bash
RUN pip install -r requirements.txt
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 --upgrade
RUN pip install triton ninja
RUN pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers

ENTRYPOINT [ "bash" ]
