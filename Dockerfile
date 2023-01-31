ARG BASE_IMAGE=nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04
FROM ${BASE_IMAGE} AS base_image
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-key del 7fa2af80 2>&1 1>/dev/null && \
    . /etc/os-release && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/${ID}`echo $VERSION_ID | tr -d '.'`/`uname --machine`/3bf863cc.pub 2>&1 1>/dev/null  \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/${ID}`echo $VERSION_ID | tr -d '.'`/`uname --machine`/7fa2af80.pub 2>&1 1>/dev/null

RUN apt-get update && apt-get install -y curl \
        openssh-client \
        python3-dev \
        python3-distutils \
        python3-pip \
        python3.8-dev \
        libgl1 \
        ffmpeg \
        libsm6 \
        libxext6 \
        wget \
        sudo \
        nano \
        tmux \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

RUN ln -f -s /usr/bin/python3.8 /usr/bin/python3 && curl -sSL https://bootstrap.pypa.io/get-pip.py | python3 -

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

RUN pip install --no-cache-dir torch==1.7.1 \
        torchvision==0.8.2 \
        fire==0.4.0 \
        clearml==1.6.2 \
        tqdm==4.64.0 \
        yacs==0.1.8 \
        matplotlib>=3.2.2 \
        numpy>=1.18.5 \
        opencv-python>=4.1.2 \
        Pillow==9.2.0 \
        PyYAML>=5.3 \
        tensorboardX==2.5.1 \
        prefetch_generator==1.0.1 \
        scikit-learn==1.1.2 \ 
        segmentation-models-pytorch==0.3.2 \
        ipykernel

ARG USERNAME=lidar_fan
ENV USERNAME=${USERNAME}
ARG UID=6666
ARG GID=6666

# create user
RUN useradd -m ${USERNAME} --uid=${UID} && echo "${USERNAME}:${USERNAME}" | chpasswd && adduser ${USERNAME} sudo

USER ${UID}:${GUI}
WORKDIR /home/${USERNAME}
