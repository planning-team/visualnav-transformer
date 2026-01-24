"""
ViNT Text Dataset: Dataset wrapper for training ViNT with text goals.

Uses the egowalk-dataset library to load EgoWalk trajectories with text captions.

Environment setup:
    - EGOWALK_LIB_PATH: Path to egowalk-dataset library (or add to PYTHONPATH)
    - HF_EGOWALK_HOME: Path to EgoWalk data (or pass data_path in config)
"""

import os
import sys
import yaml
import torch
import numpy as np
from typing import List, Optional, Tuple, Union, Callable
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from vint_train.data.data_utils import IMAGE_ASPECT_RATIO

# Add egowalk-dataset to path if EGOWALK_LIB_PATH is set
_egowalk_lib_path = os.environ.get("EGOWALK_LIB_PATH")
if _egowalk_lib_path and _egowalk_lib_path not in sys.path:
    sys.path.insert(0, _egowalk_lib_path)

try:
    from egowalk_dataset.datasets.gnm.gnm_indexing import index_gnm_text
    from egowalk_dataset.datasets.gnm.gnm_dataset import (
        GNMDataset,
        GNMRGBFeature,
        GNMCaptionFeature,
        GNMWaypointFeature,
    )
    from egowalk_dataset.datasets.gnm.cutters import (
        SpikesCutter,
        StuckCutter,
        BackwardCutter,
    )
except ImportError as e:
    raise ImportError(
        f"egowalk-dataset library not found: {e}\n"
        "Please set EGOWALK_LIB_PATH env var or add egowalk-dataset to PYTHONPATH.\n"
        "Example: export EGOWALK_LIB_PATH=/path/to/egowalk-dataset"
    )


class ViNTImageTransform:
    """
    Image transform for ViNT format that can be pickled for multiprocessing.

    Matches original ViNT transform from data_utils.py:resize_and_aspect_crop
    - Center crop to 4:3 aspect ratio
    - Resize to target size
    - Convert to tensor and normalize to [0, 1]

    Args:
        image_size: Target size (width, height)
    """
    def __init__(self, image_size: Tuple[int, int] = (85, 64)):
        self.image_size = image_size

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        # img is [H, W, 3] numpy array from egowalk-dataset
        # Convert to PIL for transforms
        pil_img = Image.fromarray(img)

        # Center crop to 4:3 aspect ratio (same as original ViNT)
        w, h = pil_img.size
        if w > h:
            pil_img = TF.center_crop(pil_img, (h, int(h * IMAGE_ASPECT_RATIO)))
        else:
            pil_img = TF.center_crop(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))

        # Resize to ViNT size (width, height)
        pil_img = pil_img.resize(self.image_size, Image.BILINEAR)

        # Convert to tensor and normalize to [0, 1] (same as TF.to_tensor)
        img_tensor = TF.to_tensor(pil_img)

        return img_tensor


class ViNT_Text_Dataset(Dataset):
    """
    Dataset for training ViNT with text goals using EgoWalk data.

    Wraps the egowalk-dataset GNMDataset to provide data in ViNT format.

    Args:
        trajectories: List of trajectory names to include
        context_size: Number of context frames (excluding current)
        len_traj_pred: Number of waypoints to predict
        image_size: Target image size (width, height)
        normalize: Whether to normalize images to [0, 1] and actions by metric_waypoint_spacing
        caption_type: Type of captions ("normal" or "brief")
        window_step: Step size for sliding window
        n_window_steps: Number of window steps per annotation
        context_step: Step between context frames
        action_step: Step between action waypoints
        data_path: Path to EgoWalk data directory
        n_workers: Number of workers for indexing
    """

    def __init__(
        self,
        trajectories: Optional[List[str]] = None,
        context_size: int = 5,
        len_traj_pred: int = 5,
        image_size: Tuple[int, int] = (85, 64),
        normalize: bool = True,
        caption_type: str = "normal",
        window_step: int = 2,
        n_window_steps: int = 4,
        context_step: int = 1,
        action_step: int = 1,
        data_path: Optional[str] = None,
        annotations_path: Optional[str] = None,
        annotations_subset: str = "end2end",
        n_workers: int = 0,
    ):
        super().__init__()

        self.context_size = context_size
        self.len_traj_pred = len_traj_pred
        self.image_size = image_size
        self.normalize = normalize
        self.caption_type = caption_type

        # Load data_config.yaml for metric_waypoint_spacing (same as original ViNT)
        with open(os.path.join(os.path.dirname(__file__), "data_config.yaml"), "r") as f:
            all_data_config = yaml.safe_load(f)
        self.data_config = all_data_config["egowalk"]

        # Set data path - from config or HF_EGOWALK_HOME env var
        if data_path is None:
            data_path = os.environ.get("HF_EGOWALK_HOME")
            if data_path is None:
                raise ValueError(
                    "data_path must be provided in config or set HF_EGOWALK_HOME env var.\n"
                    "Example: export HF_EGOWALK_HOME=/path/to/egowalk/data"
                )
        self.data_path = os.path.join(data_path, "EgoWalk", "trajectories")

        # Create image transform (picklable class for multiprocessing)
        self.image_transform = ViNTImageTransform(image_size)

        # Create index using egowalk library
        # Note: context_length in egowalk includes the current frame,
        # but ViNT's context_size is previous frames only
        # So we use context_length = context_size (egowalk will give us context_size+1 frames total

        # Create cutters for trajectory segmentation
        cutters = [
            SpikesCutter(spike_threshold=2.0),
            BackwardCutter(backwards_eps=1e-2, stuck_eps=1e-2, ignore_stuck=True),
            SpikesCutter(spike_threshold=2.0),
        ]

        # Annotations path - can be custom or default
        if annotations_path is None:
            # Default: annotations inside data directory
            annotations_path = os.path.join(self.data_path, "annotations")

        print(f"Data path: {self.data_path}")
        print(f"Annotations path: {annotations_path}")

        print(f"Indexing trajectories for text goals...")
        self.gnm_index = index_gnm_text(
            cutters=cutters,
            annotations_path=annotations_path,
            annotations_subset=annotations_subset,
            caption_type=caption_type,
            context_length=context_size,  # This gives us context_size+1 frames total
            action_length=len_traj_pred,
            window_step=window_step,
            context_step=context_step,
            action_step=action_step,
            data_path=self.data_path,
            trajectories=trajectories,
            n_workers=n_workers,
            use_tqdm=True,
        )
        print(f"Indexed {len(self.gnm_index['trajectory'])} samples")

        # Create GNMDataset with appropriate features
        self.gnm_dataset = GNMDataset(
            index=self.gnm_index,
            features=[
                GNMRGBFeature(
                    name="obs",
                    field="obs",
                    transform=self.image_transform,
                ),
                GNMCaptionFeature(name="goal_text"),
                GNMWaypointFeature(
                    name="action",
                    angle_format="sincos",  # ViNT uses sin, cos for angles
                    return_tensors="pt",
                ),
            ],
            data_path=self.data_path,
        )

    def __len__(self) -> int:
        return len(self.gnm_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            obs_img: Observation images [3*(context_size+1), H, W]
            goal_text: Text description of goal
            action: Waypoints [len_traj_pred, 4] (x, y, sin, cos)
        """
        item = self.gnm_dataset[idx]

        # obs is [context_size+1, C, H, W] tensor
        obs = item["obs"]

        # Flatten to ViNT format: [3*(context_size+1), H, W]
        if len(obs.shape) == 4:
            # [N, C, H, W] -> [N*C, H, W]
            obs_img = obs.reshape(-1, obs.shape[-2], obs.shape[-1])
        else:
            obs_img = obs

        # Goal text
        goal_text = item["goal_text"]

        # Actions [len_traj_pred, 4] - (x, y, sin, cos)
        action = item["action"]

        # Normalize action waypoints (critical for generalization)
        # Same as original ViNT: divide x,y by metric_waypoint_spacing
        if self.normalize:
            action = action.clone()  # Don't modify original data
            action[:, :2] = action[:, :2] / self.data_config["metric_waypoint_spacing"]

        return obs_img, goal_text, action


class ViNT_Text_DataLoader:
    """
    DataLoader wrapper that handles text goal batching.

    Standard DataLoader can't batch strings, so this wrapper
    collects text goals as a list while batching tensors normally.
    """

    def __init__(
        self,
        dataset: ViNT_Text_Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        # Create standard DataLoader with custom collate
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch):
        """Custom collate function that handles text goals."""
        obs_imgs = []
        goal_texts = []
        actions = []

        for obs_img, goal_text, action in batch:
            obs_imgs.append(obs_img)
            goal_texts.append(goal_text)
            actions.append(action)

        # Stack tensors
        obs_imgs = torch.stack(obs_imgs, dim=0)
        actions = torch.stack(actions, dim=0)

        # goal_texts remains a list of strings
        return obs_imgs, goal_texts, actions

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)


def get_egowalk_text_dataset(
    trajectories: Optional[List[str]] = None,
    context_size: int = 5,
    len_traj_pred: int = 5,
    image_size: Tuple[int, int] = (85, 64),
    caption_type: str = "normal",
    **kwargs,
) -> ViNT_Text_Dataset:
    """
    Convenience function to create EgoWalk text dataset.

    Args:
        trajectories: List of trajectory names (None for all)
        context_size: Number of context frames
        len_traj_pred: Number of waypoints to predict
        image_size: Target image size
        caption_type: "normal" or "brief"
        **kwargs: Additional arguments for ViNT_Text_Dataset

    Returns:
        ViNT_Text_Dataset instance
    """
    return ViNT_Text_Dataset(
        trajectories=trajectories,
        context_size=context_size,
        len_traj_pred=len_traj_pred,
        image_size=image_size,
        caption_type=caption_type,
        **kwargs,
    )
