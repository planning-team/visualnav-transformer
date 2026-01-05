"""
ViNT Text Dataset: Dataset wrapper for training ViNT with text goals.

Uses the egowalk-dataset library to load EgoWalk trajectories with text captions.
"""

import os
import sys
import torch
import numpy as np
from typing import List, Optional, Tuple, Union, Callable
from PIL import Image
from torch.utils.data import Dataset

# Add egowalk-dataset to path
EGOWALK_DATASET_PATH = "/media/mohamad/Transcend/Skoltech-PhD/egowalk-dataset"
if EGOWALK_DATASET_PATH not in sys.path:
    sys.path.insert(0, EGOWALK_DATASET_PATH)

# Set HF_EGOWALK_HOME environment variable if not set
if "HF_EGOWALK_HOME" not in os.environ:
    os.environ["HF_EGOWALK_HOME"] = "/media/mohamad/Transcend/Skoltech-PhD/egowalk-dataset/hf_data_dir"

from egowalk_dataset.datasets.gnm.gnm_indexing import index_gnm_text
from egowalk_dataset.datasets.gnm.gnm_dataset import (
    GNMDataset,
    GNMRGBFeature,
    GNMCaptionFeature,
    GNMWaypointFeature,
)


def create_vint_image_transform(
    image_size: Tuple[int, int] = (85, 64),
    normalize: bool = True,
) -> Callable:
    """
    Create image transform for ViNT format.

    Args:
        image_size: Target size (width, height)
        normalize: Whether to normalize to [0, 1]

    Returns:
        Transform function
    """
    def transform(img: np.ndarray) -> torch.Tensor:
        # img is [H, W, 3] numpy array from egowalk-dataset
        # Convert to PIL for resizing
        pil_img = Image.fromarray(img)

        # Resize to ViNT size (width, height)
        pil_img = pil_img.resize(image_size, Image.BILINEAR)

        # Convert to numpy and then tensor
        img_array = np.array(pil_img)

        # Convert to [C, H, W] format
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()

        # Normalize to [0, 1]
        if normalize:
            img_tensor = img_tensor / 255.0

        return img_tensor

    return transform


class ViNT_Text_Dataset(Dataset):
    """
    Dataset for training ViNT with text goals using EgoWalk data.

    Wraps the egowalk-dataset GNMDataset to provide data in ViNT format.

    Args:
        trajectories: List of trajectory names to include
        context_size: Number of context frames (excluding current)
        len_traj_pred: Number of waypoints to predict
        image_size: Target image size (width, height)
        normalize: Whether to normalize images to [0, 1]
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
        n_workers: int = 0,
    ):
        super().__init__()

        self.context_size = context_size
        self.len_traj_pred = len_traj_pred
        self.image_size = image_size
        self.normalize = normalize
        self.caption_type = caption_type

        # Set data path
        if data_path is None:
            data_path = os.environ.get(
                "HF_EGOWALK_HOME",
                "/media/mohamad/Transcend/Skoltech-PhD/egowalk-dataset/hf_data_dir"
            )
        self.data_path = os.path.join(data_path, "EgoWalk", "trajectories")

        # Create image transform
        self.image_transform = create_vint_image_transform(image_size, normalize)

        # Create index using egowalk library
        # Note: context_length in egowalk includes the current frame,
        # but ViNT's context_size is previous frames only
        # So we use context_length = context_size (egowalk will give us context_size+1 frames)
        print(f"Indexing trajectories for text goals...")
        self.gnm_index = index_gnm_text(
            context_length=context_size,  # This gives us context_size+1 frames total
            action_length=len_traj_pred,
            caption_type=caption_type,
            window_step=window_step,
            n_window_steps=n_window_steps,
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

        # Actions [len_traj_pred, 4]
        action = item["action"]

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
