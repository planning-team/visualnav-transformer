"""
ViNT_Text: ViNT model with text-based goals using SigLIP2.

This model replaces ViNT's image goal encoder with a SigLIP2 text encoder,
allowing navigation based on natural language descriptions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union
from efficientnet_pytorch import EfficientNet

from vint_train.models.base_model import BaseModel
from vint_train.models.vint.self_attention import MultiLayerDecoder
from vint_train.models.vint.text_encoder import SigLIP2TextEncoder


class ViNT_Text(BaseModel):
    """
    ViNT with text-based goals.

    Uses the same observation encoder and transformer decoder as ViNT,
    but replaces the image goal encoder with SigLIP2 text encoder.

    Args:
        context_size: Number of previous observations for context
        len_traj_pred: Number of waypoints to predict
        learn_angle: Whether to predict yaw angle
        obs_encoder: EfficientNet architecture name
        obs_encoding_size: Observation embedding dimension
        mha_num_attention_heads: Number of attention heads in decoder
        mha_num_attention_layers: Number of transformer layers
        mha_ff_dim_factor: Feed-forward expansion factor
        siglip_model_name: SigLIP2 model name from HuggingFace
        siglip_cache_dir: Directory to cache SigLIP2 weights
        freeze_siglip: Whether to freeze SigLIP2 weights
        freeze_vint: Whether to freeze ViNT components (obs encoder, decoder, predictors)
    """

    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        mha_num_attention_heads: Optional[int] = 4,
        mha_num_attention_layers: Optional[int] = 4,
        mha_ff_dim_factor: Optional[int] = 4,
        siglip_model_name: str = "google/siglip2-base-patch16-224",
        siglip_cache_dir: Optional[str] = None,
        freeze_siglip: bool = True,
        freeze_vint: bool = False,
    ) -> None:
        super(ViNT_Text, self).__init__(context_size, len_traj_pred, learn_angle)

        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size  # Must match for concatenation

        # Observation encoder (same as ViNT)
        if obs_encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3)
            self.num_obs_features = self.obs_encoder._fc.in_features
        else:
            raise NotImplementedError(f"Observation encoder {obs_encoder} not supported")

        # Compression layer for observation features
        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()

        # Text goal encoder (replaces image goal encoder)
        self.text_encoder = SigLIP2TextEncoder(
            model_name=siglip_model_name,
            goal_encoding_size=self.goal_encoding_size,
            freeze_siglip=freeze_siglip,
            cache_dir=siglip_cache_dir,
        )

        # Transformer decoder (same as ViNT)
        self.decoder = MultiLayerDecoder(
            embed_dim=self.obs_encoding_size,
            seq_len=self.context_size + 2,  # context + current obs + goal
            output_layers=[256, 128, 64, 32],
            nhead=mha_num_attention_heads,
            num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor,
        )

        # Prediction heads (same as ViNT)
        self.dist_predictor = nn.Sequential(
            nn.Linear(32, 1),
        )
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_trajectory_pred * self.num_action_params),
        )

        # Optionally freeze ViNT components (when training only text encoder)
        if freeze_vint:
            self.freeze_vint_components()

    def forward(
        self,
        obs_img: torch.Tensor,
        goal_text: Union[str, List[str]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with text goals.

        Args:
            obs_img: Observation images [batch, 3*(context_size+1), H, W]
            goal_text: Text descriptions of goals (string or list of strings)

        Returns:
            dist_pred: Predicted distance to goal [batch, 1]
            action_pred: Predicted waypoints [batch, len_traj_pred, num_action_params]
        """
        batch_size = obs_img.shape[0]

        # Encode text goal
        # [batch_size, 1, goal_encoding_size]
        goal_encoding = self.text_encoder(goal_text)

        # Split observation into context frames
        # [batch_size, 3*(context_size+1), H, W] -> list of [batch_size, 3, H, W]
        obs_img = torch.split(obs_img, 3, dim=1)

        # Stack along batch dimension for efficient encoding
        # [batch_size*(context_size+1), 3, H, W]
        obs_img = torch.cat(obs_img, dim=0)

        # Encode observations
        obs_encoding = self.obs_encoder.extract_features(obs_img)
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)

        # Compress to target dimension
        obs_encoding = self.compress_obs_enc(obs_encoding)

        # Reshape: [batch*(context+1), encoding] -> [batch, context+1, encoding]
        obs_encoding = obs_encoding.reshape((self.context_size + 1, batch_size, self.obs_encoding_size))
        obs_encoding = torch.transpose(obs_encoding, 0, 1)

        # Concatenate observations and goal
        # [batch, context+1, encoding] + [batch, 1, encoding] -> [batch, context+2, encoding]
        tokens = torch.cat((obs_encoding, goal_encoding), dim=1)

        # Decode
        final_repr = self.decoder(tokens)

        # Predict distance and actions
        dist_pred = self.dist_predictor(final_repr)
        action_pred = self.action_predictor(final_repr)

        # Reshape action predictions
        action_pred = action_pred.reshape(
            (batch_size, self.len_trajectory_pred, self.num_action_params)
        )

        # Convert deltas to cumulative waypoints
        action_pred[:, :, :2] = torch.cumsum(action_pred[:, :, :2], dim=1)

        # Normalize angle predictions if learning angles
        if self.learn_angle:
            action_pred[:, :, 2:] = F.normalize(action_pred[:, :, 2:].clone(), dim=-1)

        return dist_pred, action_pred

    @classmethod
    def from_pretrained_vint(
        cls,
        vint_checkpoint_path: str,
        siglip_model_name: str = "google/siglip2-base-patch16-224",
        siglip_cache_dir: Optional[str] = None,
        freeze_vint: bool = True,
        freeze_siglip: bool = True,
        device: Optional[torch.device] = None,
    ) -> "ViNT_Text":
        """
        Create ViNT_Text from a pre-trained ViNT checkpoint.

        Loads observation encoder, decoder, and prediction heads from ViNT,
        initializes a new SigLIP2 text encoder for goals.

        Args:
            vint_checkpoint_path: Path to ViNT checkpoint (.pth file)
            siglip_model_name: SigLIP2 model name
            siglip_cache_dir: Directory to cache SigLIP2 weights
            freeze_vint: Whether to freeze loaded ViNT weights
            freeze_siglip: Whether to freeze SigLIP2 weights
            device: Device to load model on

        Returns:
            ViNT_Text model with loaded weights
        """
        # Load checkpoint
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(vint_checkpoint_path, map_location=device)

        # Extract model state dict (handle different checkpoint formats)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present (from DataParallel)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # Infer model configuration from checkpoint
        # Get context_size from decoder positional encoding
        if "decoder.positional_encoding.pe" in state_dict:
            seq_len = state_dict["decoder.positional_encoding.pe"].shape[1]
            context_size = seq_len - 2  # seq_len = context + current + goal
        else:
            context_size = 5  # default

        # Get encoding size from compress_obs_enc
        if "compress_obs_enc.weight" in state_dict:
            obs_encoding_size = state_dict["compress_obs_enc.weight"].shape[0]
        else:
            obs_encoding_size = 512  # default

        # Get len_traj_pred and learn_angle from action_predictor
        if "action_predictor.0.weight" in state_dict:
            action_out_dim = state_dict["action_predictor.0.weight"].shape[0]
            # action_out_dim = len_traj_pred * num_action_params
            # If learn_angle: num_action_params = 4, else 2
            if action_out_dim % 4 == 0:
                len_traj_pred = action_out_dim // 4
                learn_angle = True
            else:
                len_traj_pred = action_out_dim // 2
                learn_angle = False
        else:
            len_traj_pred = 5
            learn_angle = True

        # Get MHA params from decoder
        if "decoder.sa_layer.self_attn.in_proj_weight" in state_dict:
            # in_proj_weight shape is [3*embed_dim, embed_dim]
            embed_dim = state_dict["decoder.sa_layer.self_attn.in_proj_weight"].shape[1]
            mha_num_attention_heads = 4  # Can't easily infer, use default
        else:
            mha_num_attention_heads = 4

        # Count number of transformer layers
        mha_num_attention_layers = 0
        for key in state_dict.keys():
            if "decoder.sa_decoder.layers" in key:
                layer_idx = int(key.split(".")[3])
                mha_num_attention_layers = max(mha_num_attention_layers, layer_idx + 1)
        if mha_num_attention_layers == 0:
            mha_num_attention_layers = 4  # default

        # Create model
        model = cls(
            context_size=context_size,
            len_traj_pred=len_traj_pred,
            learn_angle=learn_angle,
            obs_encoder="efficientnet-b0",
            obs_encoding_size=obs_encoding_size,
            mha_num_attention_heads=mha_num_attention_heads,
            mha_num_attention_layers=mha_num_attention_layers,
            mha_ff_dim_factor=4,
            siglip_model_name=siglip_model_name,
            siglip_cache_dir=siglip_cache_dir,
            freeze_siglip=freeze_siglip,
        )

        # Load weights for observation encoder
        obs_encoder_dict = {
            k.replace("obs_encoder.", ""): v
            for k, v in state_dict.items()
            if k.startswith("obs_encoder.")
        }
        if obs_encoder_dict:
            model.obs_encoder.load_state_dict(obs_encoder_dict, strict=False)

        # Load weights for compression layer
        compress_dict = {
            k.replace("compress_obs_enc.", ""): v
            for k, v in state_dict.items()
            if k.startswith("compress_obs_enc.")
        }
        if compress_dict:
            model.compress_obs_enc.load_state_dict(compress_dict, strict=True)

        # Load weights for decoder
        decoder_dict = {
            k.replace("decoder.", ""): v
            for k, v in state_dict.items()
            if k.startswith("decoder.")
        }
        if decoder_dict:
            model.decoder.load_state_dict(decoder_dict, strict=True)

        # Load weights for prediction heads
        dist_dict = {
            k.replace("dist_predictor.", ""): v
            for k, v in state_dict.items()
            if k.startswith("dist_predictor.")
        }
        if dist_dict:
            model.dist_predictor.load_state_dict(dist_dict, strict=True)

        action_dict = {
            k.replace("action_predictor.", ""): v
            for k, v in state_dict.items()
            if k.startswith("action_predictor.")
        }
        if action_dict:
            model.action_predictor.load_state_dict(action_dict, strict=True)

        # Freeze ViNT components if requested
        if freeze_vint:
            model.freeze_vint_components()

        return model

    def freeze_vint_components(self):
        """Freeze all ViNT components (observation encoder, decoder, predictors)."""
        # Freeze observation encoder
        for param in self.obs_encoder.parameters():
            param.requires_grad = False

        # Freeze compression layer
        for param in self.compress_obs_enc.parameters():
            param.requires_grad = False

        # Freeze decoder
        for param in self.decoder.parameters():
            param.requires_grad = False

        # Freeze prediction heads
        for param in self.dist_predictor.parameters():
            param.requires_grad = False
        for param in self.action_predictor.parameters():
            param.requires_grad = False

    def get_trainable_parameters(self):
        """Return only trainable parameters."""
        trainable_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params

    def print_trainable_parameters(self):
        """Print summary of trainable vs frozen parameters."""
        trainable_params = 0
        total_params = 0
        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"  Trainable: {name} ({param.numel():,} params)")

        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
        print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
