#!/usr/bin/env python3
"""
Generate reproducible train/test trajectory split for EgoWalk dataset.
Run this once to create trajectory_split.yaml file.
"""

import os
import yaml
import random
from pathlib import Path
import argparse


def get_all_trajectories(data_path, annotations_path, annotations_subset="end2end"):
    """
    Get list of all available trajectories that have annotations.

    Args:
        data_path: Path to EgoWalk data directory
        annotations_path: Path to annotations directory
        annotations_subset: Annotation subset (e.g., "end2end")

    Returns:
        List of trajectory names
    """
    # Get trajectories from annotations (these have text goals)
    annotations_dir = Path(annotations_path) / annotations_subset

    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")

    # List all .parquet files
    annotation_files = list(annotations_dir.glob("*.parquet"))

    # Extract trajectory names (remove .parquet extension)
    trajectories = [f.stem for f in annotation_files]

    print(f"Found {len(trajectories)} trajectories with annotations in {annotations_dir}")

    return sorted(trajectories)


def split_trajectories(trajectories, train_fraction=0.9, seed=42):
    """
    Split trajectories into train/test sets.

    Args:
        trajectories: List of trajectory names
        train_fraction: Fraction for training (default 0.9 = 90%)
        seed: Random seed for reproducibility

    Returns:
        dict with 'train' and 'test' trajectory lists
    """
    random.seed(seed)

    # Shuffle trajectories
    shuffled = trajectories.copy()
    random.shuffle(shuffled)

    # Split
    n_train = int(len(trajectories) * train_fraction)
    train_trajectories = sorted(shuffled[:n_train])
    test_trajectories = sorted(shuffled[n_train:])

    print(f"\nSplit: {len(train_trajectories)} train / {len(test_trajectories)} test")
    print(f"Train: {train_fraction*100:.1f}% / Test: {(1-train_fraction)*100:.1f}%")

    return {
        'train': train_trajectories,
        'test': test_trajectories,
        'seed': seed,
        'train_fraction': train_fraction,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate reproducible train/test trajectory split"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/mnt/vol0/hf_cache/egowalk/",
        help="Path to EgoWalk data"
    )
    parser.add_argument(
        "--annotations_path",
        type=str,
        default="/mnt/vol0/datasets/egowalk_lang/release/annotations",
        help="Path to annotations directory"
    )
    parser.add_argument(
        "--annotations_subset",
        type=str,
        default="end2end",
        help="Annotation subset to use"
    )
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=0.9,
        help="Fraction of data for training (default: 0.9 = 90%%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="config/trajectory_split.yaml",
        help="Output YAML file"
    )

    args = parser.parse_args()

    print("="*60)
    print("Generating EgoWalk Trajectory Split")
    print("="*60)
    print(f"Data path: {args.data_path}")
    print(f"Annotations: {args.annotations_path}/{args.annotations_subset}")
    print(f"Train fraction: {args.train_fraction}")
    print(f"Random seed: {args.seed}")
    print("="*60)

    # Get all trajectories
    trajectories = get_all_trajectories(
        args.data_path,
        args.annotations_path,
        args.annotations_subset
    )

    if len(trajectories) == 0:
        print("\nERROR: No trajectories found!")
        return

    # Split trajectories
    split = split_trajectories(
        trajectories,
        train_fraction=args.train_fraction,
        seed=args.seed
    )

    # Save to YAML
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(split, f, default_flow_style=False, sort_keys=False)

    print(f"\n✓ Trajectory split saved to: {output_path}")
    print(f"\nFirst 5 train trajectories:")
    for traj in split['train'][:5]:
        print(f"  - {traj}")
    print(f"\nFirst 5 test trajectories:")
    for traj in split['test'][:5]:
        print(f"  - {traj}")

    print("\n" + "="*60)
    print("To use this split in training, update your config:")
    print(f"  trajectory_split_file: \"{output_path}\"")
    print("="*60)


if __name__ == "__main__":
    main()
