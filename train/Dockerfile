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
    "numpy<2.0" \
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
    pyarrow \
    opencv-python \
    scikit-image

# Install compatible versions: use huggingface_hub < 0.26.0 to avoid cached_download removal
# See: https://github.com/easydiffusion/easydiffusion/issues/1851
RUN pip install huggingface_hub==0.25.2 diffusers==0.27.0

# Install transformers for SigLIP2 (need 4.41+ for SigLIP2 support)
RUN pip install transformers==4.45.0 accelerate

# Install diffusion_policy (needed for NoMaD, but good to have)
RUN pip install einops hydra-core

# Install vit-pytorch for ViT model
RUN pip install vit-pytorch

# Clone and install diffusion_policy (needed for NoMaD)
RUN cd /workspace && \
    git clone https://github.com/real-stanford/diffusion_policy.git && \
    cd diffusion_policy && \
    pip install -e .

# Set environment variables
ENV PYTHONPATH=/workspace/train:/workspace/egowalk-dataset:/workspace/diffusion_policy:$PYTHONPATH
ENV HF_HOME=/workspace/cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/cache/transformers

# Default command
CMD ["bash"]
