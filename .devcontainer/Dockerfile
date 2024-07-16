# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Prevent tzdata from asking for user input
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

# Install system dependencies
RUN apt-get update --fix-missing && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    vim \
    python3-pip \
    python3-dev \
    python3-opencv \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh \
    && $CONDA_DIR/bin/conda clean -a -y

# Initialize Conda for the shell
RUN conda init bash

# Create a new Conda environment
COPY .devcontainer/environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml \
    && conda clean -a -y

# Activate the Conda environment in all future commands
SHELL ["conda", "run", "-n", "data_science", "/bin/bash", "-c"]

# Ensure the Conda environment is activated by default in the container
RUN echo "conda activate data_science" >> ~/.bashrc

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch and torchvision with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Set working directory
WORKDIR /workspace

# Set the default command to bash
CMD ["bash"]
