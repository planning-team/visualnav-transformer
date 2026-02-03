# ViNT Text Training on Server

This guide explains how to train ViNT with text goals on a server with a good GPU, we use (A100 80GB GPU, 256GB RAM).

## Setup Steps

### 1. Clone Repository on your machine

```bash
cd /path/to/your/workspace
git clone <your-repo-url>
cd visualnav-transformer/train
```

### 2. Generate Trajectory Split (One-time Setup)

This creates a reproducible 90/10 train/test split.

**IMPORTANT**: Run this OUTSIDE the Docker container on the host machine:

```bash
python generate_trajectory_split.py \
  --data_path /mnt/vol0/hf_cache/egowalk/ \
  --annotations_path /mnt/vol0/datasets/egowalk_lang/release/annotations \
  --annotations_subset end2end \
  --train_fraction 0.9 \
  --seed 42 \
  --output config/trajectory_split.yaml
```

**Output**: This will create `config/trajectory_split.yaml` with:
- 90% of trajectories for training
- 10% of trajectories for testing
- Random seed 42 for reproducibility

You can check the split:
```bash
cat config/trajectory_split.yaml
```

### 3. Docker Compose Configuration

The `docker-compose.yml` is configured for our server:

```yaml
version: '3.8'

services:
  vint-text-train:
    build:
      context: .
      dockerfile: Dockerfile
    image: vint-text-train:latest
    container_name: vint-text-train

    # GPU support (A100)
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

    stdin_open: true
    tty: true

    # Volume mounts for the paths on our machine
    volumes:
      # Training code
      - ./:/workspace/train
      # EgoWalk dataset library (adjust path as needed)
      - /path/to/egowalk-dataset:/workspace/egowalk-dataset
      # EgoWalk data
      - /mnt/vol0/hf_cache/egowalk:/workspace/data
      # Cache directory
      - ./cache:/workspace/cache
      # Logs directory
      - ./logs:/workspace/train/logs
      # W&B config
      - ./wandb_config:/root/.config/wandb

    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - HF_EGOWALK_HOME=/workspace/data
      - HF_HOME=/workspace/cache/huggingface
      - TRANSFORMERS_CACHE=/workspace/cache/transformers
      - PYTHONPATH=/workspace/train:/workspace/egowalk-dataset:/workspace/diffusion_policy

    working_dir: /workspace/train
    network_mode: host
    shm_size: '32gb'  # Large shared memory for A100
```

### 4. Build Docker Image

```bash
docker build -t vint-text-train:latest .
```

### 5. Start Container

```bash
docker compose up -d
```

### 6. Log in to Weights & Biases (One-time)

```bash
docker exec -it vint-text-train bash
wandb login
# Enter your W&B API key
exit
```

### 7. Start Training

```bash
docker exec -it vint-text-train bash
python train.py --config config/vint_text.yaml
```

Or run in detached mode:
```bash
docker exec -d vint-text-train python train.py --config config/vint_text.yaml
```

## Configuration Details

### Hardware Optimization (A100 80GB)

The `vint_text.yaml` config is optimized for A100:

- **Batch size**: 32 (vs 4 on RTX 2060)
- **Workers**: 16 (vs 2 on local)
- **No gradient accumulation**: Large batch fits in memory
- **Warmup epochs**: 5 (longer warmup for stability)
- **Total epochs**: 100

### Dataset Configuration

**Host Paths (server)**:
- **Data**: `/mnt/vol0/hf_cache/egowalk/` → Container: `/workspace/data`
- **Annotations**: `/mnt/vol0/datasets/egowalk_lang/release/annotations` → Container: `/workspace/annotations`

**Config Settings**:
- **Caption type**: `brief` (as in the example notebook)
- **Annotations subset**: `end2end`
- **Window step**: 1 (more samples vs 2 on local)
- **Train/test split**: 90/10 (vs 80/20 on local)
- **Trajectories**: Loaded from `trajectory_split.yaml`

### Model Configuration

- **Text encoder**: SigLIP2-base-patch16-224
- **Vision encoder**: EfficientNet-b0
- **Frozen**: SigLIP2 and ViNT backbone (only train projection + decoder)
- **Parameters**: ~24M trainable, ~375M frozen

## Monitoring Training

### View W&B Dashboard

Visit: https://wandb.ai/medfa/egowalk_vint_text

### Check Logs

```bash
# Follow training logs
docker logs -f vint-text-train

# Check specific log file
tail -f logs/egowalk_vint_text/<run_name>/train.log
```

### Check GPU Usage

```bash
# Inside container
docker exec vint-text-train nvidia-smi

# Or from host
nvidia-smi
```

## Testing Trained Model

After training completes, evaluate on test set:

```bash
docker exec -it vint-text-train bash

python test_vint_text.py \
  --config config/vint_text.yaml \
  --checkpoint logs/egowalk_vint_text/<run_name>/latest.pth \
  --output logs/test_results.json \
  --use_wandb
```

## Checkpoints

Checkpoints are saved to:
```
logs/egowalk_vint_text/<run_name>/
├── 0.pth         # Epoch 0
├── 1.pth         # Epoch 1
├── ...
├── 99.pth        # Epoch 99
└── latest.pth    # Latest checkpoint
```

Each checkpoint includes:
- Model weights
- Optimizer state
- Epoch number
- Training config

## Resuming Training

To resume from a checkpoint:

1. Update config:
```yaml
load_run: egowalk_vint_text/<previous_run_name>
```

2. Start training:
```bash
python train.py --config config/vint_text.yaml
```

## Expected Training Time

With A100 80GB and batch size 32:
- **Per epoch**: ~10-15 minutes (depends on number of trajectories)
- **100 epochs**: ~20-25 hours
- **With hundreds of trajectories**: May take longer, monitor first epoch

## Troubleshooting

### Out of Memory

If OOM occurs:
1. Reduce `batch_size` in config (try 16 or 8)
2. Reduce `num_workers` (try 8)
3. Check `nvidia-smi` for other processes using GPU

### Data Loading Slow

If data loading is bottleneck:
1. Increase `num_workers` (try 32 with 256GB RAM)
2. Check disk I/O with `iostat -x 1`
3. Consider caching annotations

### Container Can't Find Data

Check volume mounts:
```bash
docker exec vint-text-train ls /workspace/data/EgoWalk/trajectories
docker exec vint-text-train ls /workspace/data/EgoWalk/trajectories/annotations/end2end
```

## Reproducing Results

The trajectory split is saved in `config/trajectory_split.yaml` with:
- Random seed: 42
- Train/test split: 90/10
- All trajectory names

To reproduce results:
1. Use the same `trajectory_split.yaml` file
2. Use the same config `vint_text.yaml`
3. Set `seed: 42` (or your chosen seed) in config
