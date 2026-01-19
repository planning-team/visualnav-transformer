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
python train.py -c config/vint.yaml   # Train ViNT model
python train.py -c config/nomad.yaml  # Train NoMaD model
python train.py -c config/gnm.yaml    # Train GNM model
```

### Data Processing
```bash
cd train
# Process RECON HDF5 dataset
python process_recon.py -i <recon_dataset_path> -o datasets/recon/

# Process ROS bags
python process_bags.py <args>

# Create train/test splits (after processing)
python data_split.py -i <processed_data_dir> -d <dataset_name> -s 0.8
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
- **ViNT**: EfficientNet encoder + Transformer decoder, adds temporal context
- **NoMaD**: ViNT-style encoder + diffusion policy for action generation, supports goal masking for exploration

### Training Pipeline
1. **ViNT_Dataset** (`vint_train/data/vint_dataset.py`): Loads trajectory data, samples observation-goal pairs, computes normalized actions. Uses LMDB caching for images.
2. **train_eval_loop** (`vint_train/training/train_eval_loop.py`): Standard loop for GNM/ViNT; `train_eval_loop_nomad` handles diffusion training.
3. **Config merging**: `config/defaults.yaml` provides base config, model-specific YAML overrides it.

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

### Adding a Custom Dataset
1. Process data into the required folder structure (images + traj_data.pkl)
2. Add `metric_waypoint_spacing` to `train/vint_train/data/data_config.yaml`
3. Add dataset entry to training config YAML under `datasets:`
4. Run `data_split.py` to create train/test splits

### Model Checkpoints
- Training: logs saved to `train/logs/<project_name>/<run_name>/`
- Deployment: weights go in `deployment/model_weights/`, register in `deployment/config/models.yaml`
- Resume training: add `load_run: <project_name>/<log_run_name>` to config

### Robot Configuration
`deployment/config/robot.yaml`: max velocities, frame rate, ROS topics
