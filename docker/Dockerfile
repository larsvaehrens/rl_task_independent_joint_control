FROM nvcr.io/nvidia/pytorch:20.10-py3
ENV DEBIAN_FRONTEND=noninteractive

# dependencies for gym
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
 libxcursor-dev \
 libxrandr-dev \
 libxinerama-dev \
 libxi-dev \
 mesa-common-dev \
 zip \
 unzip \
 make \
 g++ \
 wget \
 vulkan-utils \
 mesa-vulkan-drivers \
 doxygen \
 graphviz \
 fonts-roboto \
 python3-sphinx \
 pigz \
 git \
 git-lfs

# python dependencies for gym
RUN pip install -q -U \
numpy-quaternion click tqdm sphinx sphinx-rtd-theme breathe pylama

# copy gym repo to docker
COPY . /workspace/isaacgym

# install gym modules
RUN cd /workspace/isaacgym/python && pip install -q -e .
RUN cd /workspace/isaacgym/python/rlgpu/rl-pytorch && pip install -q -e .
