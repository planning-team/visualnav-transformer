"""
SigLIP2 Text Encoder for ViNT with text goals.

This module provides a text encoder that uses SigLIP2 to encode text goals
and project them to the same embedding space as ViNT's image goal encoder.
"""

from typing import List, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class SigLIP2TextEncoder(nn.Module):
    """
    SigLIP2-based text encoder for ViNT text goals.

    Encodes text descriptions and projects them to the goal embedding space
    that ViNT's transformer decoder expects.

    Args:
        model_name: HuggingFace model name (e.g., "google/siglip2-base-patch16-224")
        goal_encoding_size: Output dimension for goal embeddings (default: 512 to match ViNT)
        freeze_siglip: Whether to freeze SigLIP2 weights (default: True)
        cache_dir: Directory to cache model weights (default: None uses HF default)
    """

    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-224",
        goal_encoding_size: int = 512,
        freeze_siglip: bool = True,
        cache_dir: Optional[str] = None,
    ):
        super(SigLIP2TextEncoder, self).__init__()

        self.model_name = model_name
        self.goal_encoding_size = goal_encoding_size
        self.freeze_siglip = freeze_siglip

        # Load SigLIP2 model and tokenizer
        self.siglip_model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )

        # Get SigLIP2 text embedding dimension
        # For siglip2-base-patch16-224, this is 768
        self.siglip_embed_dim = self.siglip_model.config.text_config.hidden_size

        # MLP projection: SigLIP2 dim -> ViNT goal encoding size
        # Design rationale:
        # - hidden_dim = input_dim to avoid premature compression before non-linearity (standard practice in CLIP-like models)
        # - LayerNorm stabilizes gradients when adapting frozen encoder to new space
        # - GELU activation (standard in vision-language projectors like LLaVA)
        # - Final layer compresses to goal encoding size
        hidden_dim = self.siglip_embed_dim  # 768 - maintain full width through first transform
        self.projection = nn.Sequential(
            nn.Linear(self.siglip_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, goal_encoding_size),
        )

        # Freeze SigLIP2 weights if specified
        if freeze_siglip:
            self._freeze_siglip()

    def _freeze_siglip(self):
        """Freeze all SigLIP2 parameters."""
        for param in self.siglip_model.parameters():
            param.requires_grad = False

    def forward(
        self,
        text: Union[str, List[str]],
    ) -> torch.Tensor:
        """
        Encode text goals into embeddings.

        Args:
            text: Single string or list of strings describing goals

        Returns:
            Goal embeddings of shape [batch_size, 1, goal_encoding_size]
        """
        # Handle single string input
        if isinstance(text, str):
            text = [text]

        # Lowercase text (SigLIP2 was trained with lowercased text)
        text = [t.lower() for t in text]

        # Tokenize with padding (SigLIP2 requires padding="max_length", max_length=64)
        inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        )

        # Move inputs to same device as model
        device = next(self.siglip_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get text features from SigLIP2
        with torch.set_grad_enabled(not self.freeze_siglip):
            # get_text_features returns [batch_size, siglip_embed_dim]
            text_features = self.siglip_model.get_text_features(**inputs)

        # Project to goal encoding size
        # [batch_size, siglip_embed_dim] -> [batch_size, goal_encoding_size]
        goal_encoding = self.projection(text_features)

        # Add sequence dimension to match ViNT's expected format
        # [batch_size, goal_encoding_size] -> [batch_size, 1, goal_encoding_size]
        goal_encoding = goal_encoding.unsqueeze(1)

        return goal_encoding

    def get_trainable_parameters(self):
        """Return only trainable parameters (projection layer if SigLIP2 is frozen)."""
        if self.freeze_siglip:
            return self.projection.parameters()
        else:
            return self.parameters()


class TextEncoderConfig:
    """Configuration for SigLIP2TextEncoder."""

    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-224",
        goal_encoding_size: int = 512,
        freeze_siglip: bool = True,
        cache_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.goal_encoding_size = goal_encoding_size
        self.freeze_siglip = freeze_siglip
        self.cache_dir = cache_dir

    def to_dict(self):
        return {
            "model_name": self.model_name,
            "goal_encoding_size": self.goal_encoding_size,
            "freeze_siglip": self.freeze_siglip,
            "cache_dir": self.cache_dir,
        }

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)
