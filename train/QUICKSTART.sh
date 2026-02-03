#!/bin/bash
# Quick Start Commands for ViNT Text Training on Our Server
# Run each section step by step

set -e  # Exit on error

echo "=========================================="
echo "ViNT Text Training - Quick Start"
echo "=========================================="

# STEP 1: Generate trajectory split (ONE-TIME SETUP)
echo ""
echo "STEP 1: Generating trajectory split..."
echo "This creates a reproducible 90/10 train/test split"
python generate_trajectory_split.py \
  --data_path /mnt/vol0/hf_cache/egowalk/ \
  --annotations_path /mnt/vol0/datasets/egowalk_lang/release/annotations \
  --annotations_subset end2end \
  --train_fraction 0.9 \
  --seed 42 \
  --output config/trajectory_split.yaml

echo ""
echo "✓ Trajectory split created!"
echo "  Check: cat config/trajectory_split.yaml"

# STEP 2: Build Docker image
echo ""
echo "STEP 2: Building Docker image..."
docker build -t vint-text-train:latest .

echo ""
echo "✓ Docker image built!"

# STEP 3: Start container
echo ""
echo "STEP 3: Starting Docker container..."
docker compose up -d

echo ""
echo "✓ Container started!"

# STEP 4: W&B login (interactive - will prompt for API key)
echo ""
echo "STEP 4: Log in to Weights & Biases..."
echo "You'll be prompted to enter your W&B API key"
docker exec -it vint-text-train bash -c "wandb login"

echo ""
echo "✓ W&B configured!"

# STEP 5: Start training
echo ""
echo "STEP 5: Starting training..."
echo "Training will run in the background."
echo "Check logs with: docker logs -f vint-text-train"
docker exec -d vint-text-train python train.py --config config/vint_text.yaml

echo ""
echo "=========================================="
echo "✓ Training started!"
echo "=========================================="
echo ""
echo "Useful commands:"
echo "  - View logs:        docker logs -f vint-text-train"
echo "  - Check GPU:        docker exec vint-text-train nvidia-smi"
echo "  - Enter container:  docker exec -it vint-text-train bash"
echo "  - Stop training:    docker stop vint-text-train"
echo "  - W&B dashboard:    https://wandb.ai/medfa/egowalk_vint_text"
echo ""
echo "Checkpoints saved to: logs/egowalk_vint_text/<run_name>/"
echo ""
