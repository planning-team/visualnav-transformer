# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements General Navigation Models (GNM, ViNT, NoMaD) - goal-conditioned visual navigation policies that can control diverse robots in zero-shot. The codebase has two main components:
- **train/**: Model training and data processing
- **deployment/**: Robot deployment (ROS-based, primarily LoCoBot/TurtleBot2)

## Common Commands

### Environment Setup
```bash
# Training environment
conda env create -f train/train_environment.yml
conda activate vint_train
pip install -e train/

# Deployment environment (on robot)
conda env create -f deployment/deployment_environment.yaml
conda activate vint_deployment
pip install -e train/

# Both require diffusion_policy package
git clone git@github.com:real-stanford/diffusion_policy.git
pip install -e diffusion_policy/
```

### Training
```bash
cd train
python train.py -c config/vint.yaml        # Train ViNT (image goals)
python train.py -c config/vint_text.yaml   # Train ViNT_Text (text goals)
python train.py -c config/nomad.yaml       # Train NoMaD (diffusion)
python train.py -c config/gnm.yaml         # Train GNM (lightweight)
```

### Docker Training (GPU Server)
```bash
cd train
docker compose up -d                                    # Start container
docker exec -it vint-text-train wandb login             # Authenticate W&B
docker exec -it vint-text-train python train.py -c config/vint_text.yaml
```

### Data Processing
```bash
cd train
# Process RECON HDF5 dataset
python process_recon.py -i <recon_dataset_path> -o datasets/recon/

# Process ROS bags
python process_bags.py -d <dataset_name> -i <bags_dir> -o datasets/<name>/

# Create train/test splits (after processing)
python data_split.py -i <processed_data_dir> -d <dataset_name> -s 0.8

# Generate trajectory split for EgoWalk text dataset
python generate_trajectory_split.py \
  --data_path /path/to/egowalk \
  --annotations_path /path/to/annotations \
  --train_fraction 0.9 \
  --output config/trajectory_split.yaml
```

### Model Evaluation
```bash
cd train
python test_vint_text.py \
  --config config/vint_text.yaml \
  --checkpoint logs/<project>/<run>/latest.pth \
  --output results.json \
  --use_wandb
```

### Deployment (run from deployment/src/)
```bash
./record_bag.sh <bag_name>                           # Record demo trajectory
./create_topomap.sh <topomap_name> <bag_filename>    # Create topological map
./navigate.sh "--model vint --dir <topomap_dir>"     # Navigate with model
./explore.sh "--model nomad"                         # Explore (NoMaD only)
```

## Architecture

### Model Hierarchy
- **GNM**: MobileNetV2-based encoder, predicts distance + waypoints
- **ViNT**: EfficientNet encoder + Transformer decoder, adds temporal context (5 frames default)
- **ViNT_Text**: ViNT with SigLIP2 text encoder for natural language goal specification
- **NoMaD**: ViNT-style encoder + diffusion policy for action generation, supports goal masking for exploration

### ViNT_Text Architecture
Text-goal navigation uses SigLIP2 to encode natural language goals:
- `vint_train/models/vint/vint_text.py`: Main model combining ViNT encoder with text goals
- `vint_train/models/vint/text_encoder.py`: SigLIP2TextEncoder with MLP projection (768 → 512)
- Supports freezing ViNT backbone (`freeze_vint: True`) and/or SigLIP encoder (`freeze_siglip: True`)
- Load pretrained ViNT: set `load_vint_checkpoint` in config

### Training Pipeline
1. **ViNT_Dataset** (`vint_train/data/vint_dataset.py`): Loads trajectory data, samples observation-goal pairs, computes normalized actions. Uses LMDB caching for images.
2. **ViNT_Text_Dataset** (`vint_train/data/vint_text_dataset.py`): Loads EgoWalk trajectories with text annotations.
3. **train_eval_loop** (`vint_train/training/train_eval_loop.py`): Standard loop for GNM/ViNT; `train_eval_loop_nomad` handles diffusion; `train_eval_loop_text` handles text goals.
4. **Config merging**: `config/defaults.yaml` provides base config, model-specific YAML overrides it.

### Data Format
Each processed trajectory folder contains:
- `0.jpg, 1.jpg, ...`: Temporally ordered RGB images
- `traj_data.pkl`: Dictionary with `position` (np.ndarray [T, 2]) and `yaw` (np.ndarray [T,])

Dataset config in `train/vint_train/data/data_config.yaml` specifies `metric_waypoint_spacing` for each dataset.

### Deployment Flow
1. `navigate.py`: Loads model and topomap, subscribes to camera images, publishes waypoints
2. `pd_controller.py`: Converts waypoints to velocity commands
3. `joy_teleop.py`: Manual override via joystick

## Key Configuration

### Important Training Parameters
```yaml
model_type: vint_text          # gnm, vint, vint_text, nomad
context_size: 5                # Temporal context frames
len_traj_pred: 5               # Output waypoints
obs_encoding_size: 512         # Embedding dimension
mha_num_attention_heads: 4     # Transformer heads
mha_num_attention_layers: 4    # Transformer layers
alpha: 0.5                     # action vs distance loss tradeoff
```

### Adding a Custom Dataset
1. Process data into the required folder structure (images + traj_data.pkl)
2. Add `metric_waypoint_spacing` to `train/vint_train/data/data_config.yaml`
3. Add dataset entry to training config YAML under `datasets:`
4. Run `data_split.py` to create train/test splits

### Model Checkpoints
- Training: logs saved to `train/logs/<project_name>/<run_name>/`
- Checkpoint files: `latest.pth` (most recent), `epoch_X.pth` (per-epoch)
- Deployment: weights go in `deployment/model_weights/`, register in `deployment/config/models.yaml`
- Resume training: add `load_run: <project_name>/<log_run_name>` to config

### Robot Configuration
`deployment/config/robot.yaml`: max velocities, frame rate, ROS topics
