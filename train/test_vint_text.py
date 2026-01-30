#!/usr/bin/env python3
"""
Test script for ViNT_Text model with text-based navigation.
Evaluates trained model on EgoWalk trajectories with text goals.
"""

import os
import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from pathlib import Path
import json

from vint_train.models.vint.vint_text import ViNT_Text
from vint_train.data.vint_text_dataset import ViNT_Text_Dataset, ViNT_Text_DataLoader


def compute_waypoint_error(pred_actions, gt_actions):
    """
    Compute L2 distance error between predicted and ground truth waypoints.

    Args:
        pred_actions: [batch, num_waypoints, 4] (x, y, sin_yaw, cos_yaw)
        gt_actions: [batch, num_waypoints, 4]

    Returns:
        Dictionary of metrics
    """
    # Extract x, y positions
    pred_xy = pred_actions[:, :, :2]  # [batch, num_waypoints, 2]
    gt_xy = gt_actions[:, :, :2]

    # Compute L2 distance per waypoint
    waypoint_errors = torch.norm(pred_xy - gt_xy, dim=2)  # [batch, num_waypoints]

    # Average across waypoints for each sample
    sample_errors = waypoint_errors.mean(dim=1)  # [batch]

    # Final waypoint error (important for goal-reaching)
    final_waypoint_errors = waypoint_errors[:, -1]  # [batch]

    return {
        'mean_waypoint_error': sample_errors.mean().item(),
        'std_waypoint_error': sample_errors.std().item(),
        'final_waypoint_error': final_waypoint_errors.mean().item(),
        'median_waypoint_error': sample_errors.median().item(),
    }


def compute_orientation_error(pred_actions, gt_actions):
    """
    Compute orientation error using cosine similarity.

    Args:
        pred_actions: [batch, num_waypoints, 4] (x, y, sin_yaw, cos_yaw)
        gt_actions: [batch, num_waypoints, 4]

    Returns:
        Dictionary of metrics
    """
    # Extract sin/cos orientation
    pred_sincos = pred_actions[:, :, 2:]  # [batch, num_waypoints, 2]
    gt_sincos = gt_actions[:, :, 2:]

    # Normalize to unit vectors
    pred_sincos = torch.nn.functional.normalize(pred_sincos, dim=2)
    gt_sincos = torch.nn.functional.normalize(gt_sincos, dim=2)

    # Cosine similarity (dot product of normalized vectors)
    cos_sim = (pred_sincos * gt_sincos).sum(dim=2)  # [batch, num_waypoints]

    # Convert to angle error in degrees
    # cos_sim in [-1, 1], acos gives angle in [0, pi]
    angle_errors = torch.acos(torch.clamp(cos_sim, -1, 1)) * 180 / np.pi

    # Average across waypoints
    sample_angle_errors = angle_errors.mean(dim=1)  # [batch]

    return {
        'mean_orientation_error_deg': sample_angle_errors.mean().item(),
        'std_orientation_error_deg': sample_angle_errors.std().item(),
        'mean_cosine_similarity': cos_sim.mean().item(),
    }


def compute_multi_step_accuracy(pred_actions, gt_actions, thresholds=[0.5, 1.0, 2.0]):
    """
    Compute success rate for reaching waypoints within distance thresholds.

    Args:
        pred_actions: [batch, num_waypoints, 4]
        gt_actions: [batch, num_waypoints, 4]
        thresholds: List of distance thresholds in meters

    Returns:
        Dictionary of accuracy metrics
    """
    pred_xy = pred_actions[:, :, :2]
    gt_xy = gt_actions[:, :, :2]

    waypoint_errors = torch.norm(pred_xy - gt_xy, dim=2)  # [batch, num_waypoints]

    metrics = {}
    for thresh in thresholds:
        # Success if waypoint is within threshold
        success = waypoint_errors < thresh  # [batch, num_waypoints]

        # Success rate per waypoint step
        per_step_success = success.float().mean(dim=0)  # [num_waypoints]

        # Overall success rate
        overall_success = success.float().mean().item()

        metrics[f'success_rate_at_{thresh}m'] = overall_success
        metrics[f'final_waypoint_success_at_{thresh}m'] = per_step_success[-1].item()

    return metrics


@torch.no_grad()
def evaluate_model(model, dataloader, device, metric_waypoint_spacing, use_tqdm=True):
    """
    Evaluate model on a dataset.

    Args:
        model: ViNT_Text model
        dataloader: Test dataloader
        device: torch device
        metric_waypoint_spacing: Normalization factor to convert to meters
        use_tqdm: Whether to show progress bar

    Returns:
        Dictionary of aggregated metrics
    """
    model.eval()

    all_waypoint_errors = []
    all_orientation_errors = []
    all_cosine_similarities = []
    all_success_metrics = {0.5: [], 1.0: [], 2.0: []}

    iterator = tqdm(dataloader, desc="Evaluating") if use_tqdm else dataloader

    for data in iterator:
        obs_image, goal_text, action_label = data

        # Move to device
        obs_image = obs_image.to(device)
        action_label = action_label.to(device)

        # Forward pass
        _, action_pred = model(obs_image, goal_text)

        # Denormalize x,y coordinates to meters for metric computation
        # Both predictions and labels are normalized, so multiply by metric_waypoint_spacing
        action_pred_denorm = action_pred.clone()
        action_label_denorm = action_label.clone()
        action_pred_denorm[:, :, :2] = action_pred[:, :, :2] * metric_waypoint_spacing
        action_label_denorm[:, :, :2] = action_label[:, :, :2] * metric_waypoint_spacing

        # Compute metrics in actual meters
        waypoint_metrics = compute_waypoint_error(action_pred_denorm, action_label_denorm)
        orientation_metrics = compute_orientation_error(action_pred_denorm, action_label_denorm)
        success_metrics = compute_multi_step_accuracy(action_pred_denorm, action_label_denorm)

        # Store for aggregation
        all_waypoint_errors.append(waypoint_metrics['mean_waypoint_error'])
        all_orientation_errors.append(orientation_metrics['mean_orientation_error_deg'])
        all_cosine_similarities.append(orientation_metrics['mean_cosine_similarity'])

        for thresh in [0.5, 1.0, 2.0]:
            all_success_metrics[thresh].append(success_metrics[f'success_rate_at_{thresh}m'])

    # Aggregate metrics
    results = {
        'mean_waypoint_error_m': np.mean(all_waypoint_errors),
        'std_waypoint_error_m': np.std(all_waypoint_errors),
        'mean_orientation_error_deg': np.mean(all_orientation_errors),
        'std_orientation_error_deg': np.std(all_orientation_errors),
        'mean_cosine_similarity': np.mean(all_cosine_similarities),
    }

    for thresh in [0.5, 1.0, 2.0]:
        results[f'success_rate_at_{thresh}m'] = np.mean(all_success_metrics[thresh])

    return results


def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Update with command line args
    if args.checkpoint:
        config['checkpoint'] = args.checkpoint
    if args.batch_size:
        config['eval_batch_size'] = args.batch_size

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create test dataset
    print("Creating test dataset...")
    egowalk_config = config["egowalk_text"]

    # Load trajectory split file (required)
    trajectory_split_file = egowalk_config.get("trajectory_split_file")
    if not trajectory_split_file:
        raise ValueError(
            "trajectory_split_file must be specified in config under egowalk_text.\n"
            "This ensures testing uses the exact same split as training."
        )

    print(f"Loading trajectory split from {trajectory_split_file}...")
    with open(trajectory_split_file, 'r') as f:
        split_data = yaml.safe_load(f)

    test_trajectories = split_data.get('test')
    if not test_trajectories:
        raise ValueError(f"No 'test' key found in {trajectory_split_file}")

    print(f"Test trajectories from split file: {len(test_trajectories)}")

    test_dataset = ViNT_Text_Dataset(
        trajectories=test_trajectories,
        context_size=config["context_size"],
        len_traj_pred=config["len_traj_pred"],
        image_size=tuple(config["image_size"]),
        normalize=True,
        caption_type=egowalk_config.get("caption_type", "caption"),
        window_step=egowalk_config.get("window_step", 2),
        context_step=egowalk_config.get("context_step", 1),
        action_step=egowalk_config.get("action_step", 1),
        data_path=egowalk_config["data_path"],
        annotations_path=egowalk_config.get("annotations_path"),
        annotations_subset=egowalk_config.get("annotations_subset", "end2end"),
        n_workers=0,  # Sequential for testing
    )

    print(f"Test dataset size: {len(test_dataset)}")

    # Load metric_waypoint_spacing for denormalization
    data_config_path = os.path.join(os.path.dirname(__file__), "vint_train", "data", "data_config.yaml")
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)
    metric_waypoint_spacing = data_config["egowalk"]["metric_waypoint_spacing"]
    print(f"Metric waypoint spacing: {metric_waypoint_spacing} m")

    # Create dataloader
    test_loader = ViNT_Text_DataLoader(
        test_dataset,
        batch_size=config.get("eval_batch_size", 4),
        shuffle=False,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=True,
        drop_last=False,
    )

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Create model
    model = ViNT_Text(
        context_size=config["context_size"],
        len_traj_pred=config["len_traj_pred"],
        learn_angle=config.get("learn_angle", True),
        obs_encoder=config.get("obs_encoder", "efficientnet-b0"),
        obs_encoding_size=config.get("obs_encoding_size", 512),
        mha_num_attention_heads=config.get("mha_num_attention_heads", 4),
        mha_num_attention_layers=config.get("mha_num_attention_layers", 4),
        mha_ff_dim_factor=config.get("mha_ff_dim_factor", 4),
        siglip_model_name=config.get("siglip_model_name", "google/siglip2-base-patch16-224"),
        siglip_cache_dir=config.get("siglip_cache_dir"),
        freeze_siglip=True,  # Frozen during inference anyway
    )

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)

    print(f"\nModel loaded successfully!")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")

    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(
            project=config.get("project_name", "egowalk_vint_text"),
            entity=config.get("wandb_entity"),
            name=f"test_{Path(args.checkpoint).stem}",
            config=config,
            tags=["test", "evaluation"],
        )

    # Evaluate
    print("\n" + "="*50)
    print("Starting evaluation...")
    print("="*50 + "\n")

    results = evaluate_model(model, test_loader, device, metric_waypoint_spacing, use_tqdm=True)

    # Add metric_waypoint_spacing to results for reference
    results['metric_waypoint_spacing'] = metric_waypoint_spacing

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print(f"(metric_waypoint_spacing: {metric_waypoint_spacing} m)")
    print("="*50)
    print(f"\nWaypoint Position Error:")
    print(f"  Mean: {results['mean_waypoint_error_m']:.4f} m")
    print(f"  Std:  {results['std_waypoint_error_m']:.4f} m")

    print(f"\nOrientation Error:")
    print(f"  Mean: {results['mean_orientation_error_deg']:.2f}°")
    print(f"  Std:  {results['std_orientation_error_deg']:.2f}°")
    print(f"  Cosine Similarity: {results['mean_cosine_similarity']:.4f}")

    print(f"\nSuccess Rates (waypoint within threshold):")
    for thresh in [0.5, 1.0, 2.0]:
        success_rate = results[f'success_rate_at_{thresh}m'] * 100
        print(f"  Within {thresh}m: {success_rate:.2f}%")

    print("\n" + "="*50 + "\n")

    # Log to wandb
    if args.use_wandb:
        wandb.log(results)
        wandb.finish()

    # Save results to file
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ViNT_Text model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/vint_text.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for evaluation (default: use config)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON (default: print only)"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Log results to wandb"
    )

    args = parser.parse_args()
    main(args)
