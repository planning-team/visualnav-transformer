import os
import wandb
import argparse
import numpy as np
import yaml
import time
import pdb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam, AdamW
from torchvision import transforms
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler

"""
IMPORT YOUR MODEL HERE
"""
from vint_train.models.gnm.gnm import GNM
from vint_train.models.vint.vint import ViNT
from vint_train.models.vint.vit import ViT
from vint_train.models.vint.vint_text import ViNT_Text
from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D


from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.data.vint_text_dataset import ViNT_Text_Dataset, ViNT_Text_DataLoader
from vint_train.training.train_eval_loop import (
    train_eval_loop,
    train_eval_loop_nomad,
    train_eval_loop_text,
    load_model,
)


def main_text(config):
    """Main training function for ViNT_Text model with text goals."""
    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "gpu_ids" not in config:
            config["gpu_ids"] = [0]
        elif type(config["gpu_ids"]) == int:
            config["gpu_ids"] = [config["gpu_ids"]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(x) for x in config["gpu_ids"]]
        )
        print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("Using cpu")

    first_gpu_id = config["gpu_ids"][0]
    device = torch.device(
        f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True

    cudnn.benchmark = True
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load EgoWalk text dataset
    egowalk_config = config["egowalk_text"]

    # Add egowalk library to path
    import sys
    if egowalk_config["egowalk_lib_path"] not in sys.path:
        sys.path.insert(0, egowalk_config["egowalk_lib_path"])

    # Set environment variable
    os.environ["HF_EGOWALK_HOME"] = egowalk_config["data_path"]

    # Check if using trajectory split file
    trajectory_split_file = egowalk_config.get("trajectory_split_file")

    if trajectory_split_file:
        # Load pre-defined train/test split from file
        print(f"Loading trajectory split from {trajectory_split_file}...")
        with open(trajectory_split_file, 'r') as f:
            split_data = yaml.safe_load(f)

        train_trajectories = split_data['train']
        test_trajectories = split_data['test']

        print(f"Train trajectories: {len(train_trajectories)}")
        print(f"Test trajectories: {len(test_trajectories)}")

        # Create separate datasets
        print("Creating train dataset...")
        train_dataset = ViNT_Text_Dataset(
            trajectories=train_trajectories,
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            image_size=tuple(config["image_size"]),
            normalize=config.get("normalize", True),
            caption_type=egowalk_config.get("caption_type", "caption"),
            window_step=egowalk_config.get("window_step", 2),
            context_step=egowalk_config.get("context_step", 1),
            action_step=egowalk_config.get("action_step", 1),
            data_path=egowalk_config["data_path"],
            annotations_path=egowalk_config.get("annotations_path"),
            annotations_subset=egowalk_config.get("annotations_subset", "end2end"),
            n_workers=egowalk_config.get("n_index_workers", 0),
        )

        print("Creating test dataset...")
        test_dataset = ViNT_Text_Dataset(
            trajectories=test_trajectories,
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            image_size=tuple(config["image_size"]),
            normalize=config.get("normalize", True),
            caption_type=egowalk_config.get("caption_type", "caption"),
            window_step=egowalk_config.get("window_step", 2),
            context_step=egowalk_config.get("context_step", 1),
            action_step=egowalk_config.get("action_step", 1),
            data_path=egowalk_config["data_path"],
            annotations_path=egowalk_config.get("annotations_path"),
            annotations_subset=egowalk_config.get("annotations_subset", "end2end"),
            n_workers=egowalk_config.get("n_index_workers", 0),
        )

        print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

    else:
        # Use old approach: single dataset with random split
        print("Creating EgoWalk text dataset...")
        full_dataset = ViNT_Text_Dataset(
            trajectories=egowalk_config.get("trajectories"),
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            image_size=tuple(config["image_size"]),
            normalize=config.get("normalize", True),
            caption_type=egowalk_config.get("caption_type", "caption"),
            window_step=egowalk_config.get("window_step", 2),
            context_step=egowalk_config.get("context_step", 1),
            action_step=egowalk_config.get("action_step", 1),
            data_path=egowalk_config["data_path"],
            annotations_path=egowalk_config.get("annotations_path"),
            annotations_subset=egowalk_config.get("annotations_subset", "end2end"),
            n_workers=egowalk_config.get("n_index_workers", 0),
        )

        # Split into train/test
        train_fraction = config.get("train_fraction", 0.8)
        total_size = len(full_dataset)
        train_size = int(total_size * train_fraction)
        test_size = total_size - train_size

        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size],
            generator=torch.Generator().manual_seed(config.get("seed", 0))
        )
        print(f"Train size: {train_size}, Test size: {test_size}")

    # Create data loaders with custom collate for text
    train_loader = ViNT_Text_DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )

    test_dataloaders = {
        "egowalk_test": ViNT_Text_DataLoader(
            test_dataset,
            batch_size=config.get("eval_batch_size", config["batch_size"]),
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
    }

    # Create model
    load_vint_checkpoint = config.get("load_vint_checkpoint", "")
    if load_vint_checkpoint and os.path.exists(load_vint_checkpoint):
        print(f"Loading pre-trained ViNT from: {load_vint_checkpoint}")
        model = ViNT_Text.from_pretrained_vint(
            vint_checkpoint_path=load_vint_checkpoint,
            siglip_model_name=config.get("siglip_model_name", "google/siglip2-base-patch16-224"),
            siglip_cache_dir=config.get("siglip_cache_dir"),
            freeze_vint=config.get("freeze_vint", True),
            freeze_siglip=config.get("freeze_siglip", True),
            device=device,
        )
    else:
        print("Creating ViNT_Text model from scratch")
        model = ViNT_Text(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config.get("obs_encoder", "efficientnet-b0"),
            obs_encoding_size=config.get("obs_encoding_size", 512),
            mha_num_attention_heads=config.get("mha_num_attention_heads", 4),
            mha_num_attention_layers=config.get("mha_num_attention_layers", 4),
            mha_ff_dim_factor=config.get("mha_ff_dim_factor", 4),
            siglip_model_name=config.get("siglip_model_name", "google/siglip2-base-patch16-224"),
            siglip_cache_dir=config.get("siglip_cache_dir"),
            freeze_siglip=config.get("freeze_siglip", True),
            freeze_vint=config.get("freeze_vint", False),  # Default False when training from scratch
        )

    # Print trainable parameters
    model.print_trainable_parameters()

    # Gradient clipping
    if config.get("clipping", False):
        print("Clipping gradients to", config["max_norm"])
        for p in model.parameters():
            if not p.requires_grad:
                continue
            p.register_hook(
                lambda grad: torch.clamp(
                    grad, -1 * config["max_norm"], config["max_norm"]
                )
            )

    # Optimizer (only trainable params)
    lr = float(config["lr"])
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_name = config.get("optimizer", "adamw").lower()
    if optimizer_name == "adam":
        optimizer = Adam(trainable_params, lr=lr, betas=(0.9, 0.98))
    elif optimizer_name == "adamw":
        optimizer = AdamW(trainable_params, lr=lr)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported")

    # Scheduler
    scheduler = None
    if config.get("scheduler"):
        scheduler_name = config["scheduler"].lower()
        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["epochs"]
            )
        elif scheduler_name == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=config.get("plateau_factor", 0.5),
                patience=config.get("plateau_patience", 3),
                verbose=True,
            )

        if config.get("warmup", False):
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=config.get("warmup_epochs", 2),
                after_scheduler=scheduler,
            )

    # Move to device
    model = model.to(device)

    # Training loop
    train_eval_loop_text(
        train_model=config.get("train", True),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=train_loader,
        test_dataloaders=test_dataloaders,
        transform=transform,
        epochs=config["epochs"],
        device=device,
        project_folder=config["project_folder"],
        wandb_log_freq=config.get("wandb_log_freq", 10),
        print_log_freq=config.get("print_log_freq", 50),
        image_log_freq=config.get("image_log_freq", 500),
        num_images_log=config.get("num_images_log", 4),
        current_epoch=0,
        learn_angle=config.get("learn_angle", True),
        use_wandb=config.get("use_wandb", True),
        eval_fraction=config.get("eval_fraction", 0.5),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
    )

    print("FINISHED TRAINING ViNT_Text")


def main(config):
    # Check if this is a vint_text model
    if config.get("model_type") == "vint_text":
        return main_text(config)

    assert config["distance"]["min_dist_cat"] < config["distance"]["max_dist_cat"]
    assert config["action"]["min_dist_cat"] < config["action"]["max_dist_cat"]

    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "gpu_ids" not in config:
            config["gpu_ids"] = [0]
        elif type(config["gpu_ids"]) == int:
            config["gpu_ids"] = [config["gpu_ids"]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(x) for x in config["gpu_ids"]]
        )
        print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("Using cpu")

    first_gpu_id = config["gpu_ids"][0]
    device = torch.device(
        f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True

    cudnn.benchmark = True  # good if input sizes don't vary
    transform = ([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform = transforms.Compose(transform)

    # Load the data
    train_dataset = []
    test_dataloaders = {}

    if "context_type" not in config:
        config["context_type"] = "temporal"

    if "clip_goals" not in config:
        config["clip_goals"] = False

    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]
        if "negative_mining" not in data_config:
            data_config["negative_mining"] = True
        if "goals_per_obs" not in data_config:
            data_config["goals_per_obs"] = 1
        if "end_slack" not in data_config:
            data_config["end_slack"] = 0
        if "waypoint_spacing" not in data_config:
            data_config["waypoint_spacing"] = 1

        for data_split_type in ["train", "test"]:
            if data_split_type in data_config:
                    dataset = ViNT_Dataset(
                        data_folder=data_config["data_folder"],
                        data_split_folder=data_config[data_split_type],
                        dataset_name=dataset_name,
                        image_size=config["image_size"],
                        waypoint_spacing=data_config["waypoint_spacing"],
                        min_dist_cat=config["distance"]["min_dist_cat"],
                        max_dist_cat=config["distance"]["max_dist_cat"],
                        min_action_distance=config["action"]["min_dist_cat"],
                        max_action_distance=config["action"]["max_dist_cat"],
                        negative_mining=data_config["negative_mining"],
                        len_traj_pred=config["len_traj_pred"],
                        learn_angle=config["learn_angle"],
                        context_size=config["context_size"],
                        context_type=config["context_type"],
                        end_slack=data_config["end_slack"],
                        goals_per_obs=data_config["goals_per_obs"],
                        normalize=config["normalize"],
                        goal_type=config["goal_type"],
                    )
                    if data_split_type == "train":
                        train_dataset.append(dataset)
                    else:
                        dataset_type = f"{dataset_name}_{data_split_type}"
                        if dataset_type not in test_dataloaders:
                            test_dataloaders[dataset_type] = {}
                        test_dataloaders[dataset_type] = dataset

    # combine all the datasets from different robots
    train_dataset = ConcatDataset(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=False,
        persistent_workers=True,
    )

    if "eval_batch_size" not in config:
        config["eval_batch_size"] = config["batch_size"]

    for dataset_type, dataset in test_dataloaders.items():
        test_dataloaders[dataset_type] = DataLoader(
            dataset,
            batch_size=config["eval_batch_size"],
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

    # Create the model
    if config["model_type"] == "gnm":
        model = GNM(
            config["context_size"],
            config["len_traj_pred"],
            config["learn_angle"],
            config["obs_encoding_size"],
            config["goal_encoding_size"],
        )
    elif config["model_type"] == "vint":
        model = ViNT(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
    elif config["model_type"] == "nomad":
        if config["vision_encoder"] == "nomad_vint":
            vision_encoder = NoMaD_ViNT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        elif config["vision_encoder"] == "vib": 
            vision_encoder = ViB(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        elif config["vision_encoder"] == "vit": 
            vision_encoder = ViT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                image_size=config["image_size"],
                patch_size=config["patch_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        else: 
            raise ValueError(f"Vision encoder {config['vision_encoder']} not supported")
            
        noise_pred_net = ConditionalUnet1D(
                input_dim=2,
                global_cond_dim=config["encoding_size"],
                down_dims=config["down_dims"],
                cond_predict_scale=config["cond_predict_scale"],
            )
        dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
        
        model = NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )

        noise_scheduler = DDPMScheduler(
            num_train_timesteps=config["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
    else:
        raise ValueError(f"Model {config['model']} not supported")

    if config["clipping"]:
        print("Clipping gradients to", config["max_norm"])
        for p in model.parameters():
            if not p.requires_grad:
                continue
            p.register_hook(
                lambda grad: torch.clamp(
                    grad, -1 * config["max_norm"], config["max_norm"]
                )
            )

    lr = float(config["lr"])
    config["optimizer"] = config["optimizer"].lower()
    if config["optimizer"] == "adam":
        optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    elif config["optimizer"] == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr)
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported")

    scheduler = None
    if config["scheduler"] is not None:
        config["scheduler"] = config["scheduler"].lower()
        if config["scheduler"] == "cosine":
            print("Using cosine annealing with T_max", config["epochs"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["epochs"]
            )
        elif config["scheduler"] == "cyclic":
            print("Using cyclic LR with cycle", config["cyclic_period"])
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=lr / 10.,
                max_lr=lr,
                step_size_up=config["cyclic_period"] // 2,
                cycle_momentum=False,
            )
        elif config["scheduler"] == "plateau":
            print("Using ReduceLROnPlateau")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=config["plateau_factor"],
                patience=config["plateau_patience"],
                verbose=True,
            )
        else:
            raise ValueError(f"Scheduler {config['scheduler']} not supported")

        if config["warmup"]:
            print("Using warmup scheduler")
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=config["warmup_epochs"],
                after_scheduler=scheduler,
            )

    current_epoch = 0
    if "load_run" in config:
        load_project_folder = os.path.join("logs", config["load_run"])
        print("Loading model from ", load_project_folder)
        latest_path = os.path.join(load_project_folder, "latest.pth")
        latest_checkpoint = torch.load(latest_path) #f"cuda:{}" if torch.cuda.is_available() else "cpu")
        load_model(model, config["model_type"], latest_checkpoint)
        if "epoch" in latest_checkpoint:
            current_epoch = latest_checkpoint["epoch"] + 1

    # Multi-GPU
    if len(config["gpu_ids"]) > 1:
        model = nn.DataParallel(model, device_ids=config["gpu_ids"])
    model = model.to(device)

    if "load_run" in config:  # load optimizer and scheduler after data parallel
        if "optimizer" in latest_checkpoint:
            optimizer.load_state_dict(latest_checkpoint["optimizer"].state_dict())
        if scheduler is not None and "scheduler" in latest_checkpoint:
            scheduler.load_state_dict(latest_checkpoint["scheduler"].state_dict())

    if config["model_type"] == "vint" or config["model_type"] == "gnm": 
        train_eval_loop(
            train_model=config["train"],
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=train_loader,
            test_dataloaders=test_dataloaders,
            transform=transform,
            epochs=config["epochs"],
            device=device,
            project_folder=config["project_folder"],
            normalized=config["normalize"],
            print_log_freq=config["print_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            learn_angle=config["learn_angle"],
            alpha=config["alpha"],
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
        )
    else:
        train_eval_loop_nomad(
            train_model=config["train"],
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            noise_scheduler=noise_scheduler,
            train_loader=train_loader,
            test_dataloaders=test_dataloaders,
            transform=transform,
            goal_mask_prob=config["goal_mask_prob"],
            epochs=config["epochs"],
            device=device,
            project_folder=config["project_folder"],
            print_log_freq=config["print_log_freq"],
            wandb_log_freq=config["wandb_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            alpha=float(config["alpha"]),
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
            eval_freq=config["eval_freq"],
        )

    print("FINISHED TRAINING")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Visual Navigation Transformer")

    # project setup
    parser.add_argument(
        "--config",
        "-c",
        default="config/vint.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    args = parser.parse_args()

    with open("config/defaults.yaml", "r") as f:
        default_config = yaml.safe_load(f)

    config = default_config

    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)

    config.update(user_config)

    config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    config["project_folder"] = os.path.join(
        "logs", config["project_name"], config["run_name"]
    )
    os.makedirs(
        config[
            "project_folder"
        ],  # should error if dir already exists to avoid overwriting and old project
    )

    if config["use_wandb"]:
        wandb.login()
        wandb_entity = config.get("wandb_entity", None)
        wandb.init(
            project=config["project_name"],
            settings=wandb.Settings(start_method="fork"),
            entity=wandb_entity,
        )
        wandb.save(args.config, policy="now")  # save the config file
        wandb.run.name = config["run_name"]
        # update the wandb args with the training configurations
        if wandb.run:
            wandb.config.update(config)

    print(config)
    main(config)
