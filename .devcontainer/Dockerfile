# .devcontainer/Dockerfile

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# System deps (no python3-opencv; we'll use conda's opencv)
RUN apt-get update --fix-missing && apt-get install -y \
    wget bzip2 ca-certificates curl git vim python3-pip python3-dev libglib2.0-0 \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
 && /bin/bash ~/miniconda.sh -b -p /opt/conda \
 && rm ~/miniconda.sh \
 && $CONDA_DIR/bin/conda clean -a -y

# --- Conda config: conda-forge only + strict; steer default_channels off repo.anaconda.com ---
RUN printf "channels:\n  - conda-forge\nchannel_priority: strict\ndefault_channels:\n  - https://conda.anaconda.org/conda-forge\n" > /root/.condarc \
 && conda config --remove-key default_channels || true \
 && conda config --add default_channels https://conda.anaconda.org/conda-forge

# Fast solver from conda-forge only, then set it
RUN conda install -n base -y conda-libmamba-solver -c conda-forge --override-channels \
 && conda config --set solver libmamba

# (Optional) init for interactive shells
RUN conda init bash

# Copy env spec
COPY .devcontainer/environment.yml /tmp/environment.yml

# Print config (debug) - verify only conda-forge is present
RUN conda config --show channels && conda config --show-sources && conda info

# Create the env (NOTE: env create does NOT accept -c/--override-channels)
RUN conda env create -f /tmp/environment.yml \
 && conda clean -a -y

# Use env for subsequent RUNs
SHELL ["conda", "run", "-n", "data_science", "/bin/bash", "-c"]

# Auto-activate when attaching
RUN echo "conda activate data_science" >> ~/.bashrc

# Build-time smoke test
RUN python - <<'PY'
import sys
print("Python:", sys.version)
for mod in ["numpy", "pandas", "sklearn", "matplotlib", "jupyter", "cv2"]:
    __import__(mod)
    print(mod, "OK")
PY

# Torch CUDA 11.8
RUN pip install --upgrade pip \
 && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

WORKDIR /workspace
CMD ["bash"]
