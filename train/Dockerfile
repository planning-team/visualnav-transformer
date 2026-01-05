FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install Python dependencies
COPY train_environment.yml /workspace/
RUN pip install --upgrade pip && \
    pip install \
    torch==2.1.0 \
    torchvision==0.16.0 \
    numpy \
    matplotlib \
    pandas \
    pillow \
    pyyaml \
    tqdm \
    wandb \
    prettytable \
    efficientnet_pytorch \
    warmup_scheduler \
    lmdb \
    av \
    pyarrow

# Install diffusers with compatible huggingface_hub
RUN pip install diffusers==0.21.0 huggingface_hub==0.17.3

# Install transformers for SigLIP2
RUN pip install transformers==4.36.0 accelerate

# Install diffusion_policy (needed for NoMaD, but good to have)
RUN pip install einops hydra-core

# Set environment variables
ENV PYTHONPATH=/workspace/train:/workspace/egowalk-dataset:$PYTHONPATH
ENV HF_HOME=/workspace/cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/cache/transformers

# Default command
CMD ["bash"]
