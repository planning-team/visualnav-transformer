"""
Text-based navigation using ViNT_Text model.

This script navigates the robot toward a goal specified by natural language
(e.g., "red table", "doorway", "stairs"). Unlike the original ViNT navigation
that uses a topological map with image goals, this uses SigLIP2-encoded text
embeddings as goal representations.

Usage:
    python navigate_text.py --goal-text "red table" --waypoint 2
"""

import os
from typing import List
import numpy as np
import torch
import yaml
import argparse
import time

# ROS
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray

# Local utils
from utils import msg_to_pil, to_numpy, transform_images, load_model
from topic_names import IMAGE_TOPIC, WAYPOINT_TOPIC


# CONSTANTS
ROBOT_CONFIG_PATH = "/home/captain/visualnav-transformer/deployment/config/robot.yaml"
MODEL_CONFIG_PATH = "/home/captain/visualnav-transformer/deployment/config/models.yaml"

with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"]

# GLOBALS
context_queue: List = []
context_size: int = None

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def callback_obs(msg):
    """ROS callback for image observations."""
    obs_img = msg_to_pil(msg)
    if context_size is not None:
        if len(context_queue) < context_size + 1:
            context_queue.append(obs_img)
        else:
            context_queue.pop(0)
            context_queue.append(obs_img)


def main(args: argparse.Namespace):
    global context_size

    # Load model configuration
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    model_config_path = model_paths[args.model]["config_path"]
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

    # Validate model type
    if model_params["model_type"] != "vint_text":
        raise ValueError(
            f"This script requires a vint_text model, but got: {model_params['model_type']}"
        )

    context_size = model_params["context_size"]

    # Load model weights
    ckpt_path = model_paths[args.model]["ckpt_path"]
    if os.path.exists(ckpt_path):
        print(f"Loading model from {ckpt_path}")
    else:
        raise FileNotFoundError(f"Model weights not found at {ckpt_path}")

    model = load_model(ckpt_path, model_params, device)
    model = model.to(device)
    model.eval()

    # Pre-compute goal embedding once (efficient for deployment)
    print(f"Goal text: '{args.goal_text}'")
    print("Encoding goal text...")
    with torch.no_grad():
        goal_encoding = model.encode_goal(args.goal_text)
        goal_encoding = goal_encoding.to(device)
    print(f"Goal encoding shape: {goal_encoding.shape}")

    # ROS setup
    rospy.init_node("TEXT_NAVIGATION", anonymous=False)
    rate = rospy.Rate(RATE)
    
    # Subscribers
    rospy.Subscriber(IMAGE_TOPIC, Image, callback_obs, queue_size=1)
    
    # Publishers
    waypoint_pub = rospy.Publisher(WAYPOINT_TOPIC, Float32MultiArray, queue_size=1)
    goal_pub = rospy.Publisher("/text_nav/goal_distance", Float32MultiArray, queue_size=1)

    print("Registered with master node. Waiting for image observations...")
    print(f"Navigating toward: '{args.goal_text}'")

    # Navigation loop
    while not rospy.is_shutdown():
        # Initialize waypoint (zero motion if not enough context)
        chosen_waypoint = np.zeros(4)

        # Only predict when we have enough context frames
        if len(context_queue) > model_params["context_size"]:
            # Transform observations
            obs_images = transform_images(
                context_queue, 
                model_params["image_size"]
            )
            obs_images = obs_images.to(device)

            # Run inference with pre-computed goal encoding
            with torch.no_grad():
                dist_pred, action_pred = model.forward_with_goal_encoding(
                    obs_images, 
                    goal_encoding
                )

            # Extract predictions
            dist = to_numpy(dist_pred).flatten()[0]
            waypoints = to_numpy(action_pred)[0]  # [len_traj_pred, num_action_params]

            # Select the desired waypoint
            chosen_waypoint = waypoints[args.waypoint]

            # Publish goal distance for monitoring
            goal_dist_msg = Float32MultiArray()
            goal_dist_msg.data = [dist]
            goal_pub.publish(goal_dist_msg)

            # Log periodically
            if args.verbose:
                print(f"Distance to goal: {dist:.2f}, Waypoint: {chosen_waypoint[:2]}")

        # Apply normalization if model was trained with normalized outputs
        if model_params.get("normalize", False):
            chosen_waypoint[:2] *= (MAX_V / RATE)

        # Publish waypoint command
        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = chosen_waypoint.tolist()
        waypoint_pub.publish(waypoint_msg)

        rate.sleep()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Text-based navigation using ViNT_Text model"
    )
    parser.add_argument(
        "--model",
        "-m",
        default="vint_text",
        type=str,
        help="Model name from ../config/models.yaml (default: vint_text)",
    )
    parser.add_argument(
        "--goal-text",
        "-g",
        required=True,
        type=str,
        help="Natural language description of the goal (e.g., 'red table', 'doorway')",
    )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2,
        type=int,
        help="Index of waypoint to use for navigation (0 to len_traj_pred-1, default: 2)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print distance and waypoint at each step",
    )
    args = parser.parse_args()
    
    print(f"Using {device}")
    main(args)
