import VisionNode
from Franka_Client_Controller import FrankaClientController
import gymnasium as gym
import numpy as np
import torch
import tkinter as tk
import threading

import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from stable_baselines3 import RLPD_SAC
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

import wandb
from wandb.integration.sb3 import WandbCallback

import numpy as np
import torch
import torch.nn as nn
import Franka_Gym_Environment
import RL_Models
from Reward_Inference import VIP_Inference


import os
import time
from tqdm import tqdm
from typing import Union, List, Dict, Tuple, Optional, Any
import json
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

import sys
sys.path.append("/home/mverghese/ego_env/Franka_Kitchen_Env")
import BehaviorCloning
sys.path.remove("/home/mverghese/ego_env/Franka_Kitchen_Env")


def check_quaternion_normalized(quat):
    norm = np.linalg.norm(quat)
    return np.isclose(norm, 1.0, atol=1e-3)

def normalize_quaternion(quat):
    norm = np.linalg.norm(quat)
    if norm < 1e-6:
        raise ValueError("Cannot normalize a zero-length quaternion.")
    return quat / norm

def scale_action(action, alpha, rotation_factor = .1):
    scaled_action = action.copy()
    scaled_action[:3] *= alpha
    scaled_angle = R.from_quat(action[3:7]).as_rotvec() * alpha * rotation_factor
    scaled_action[3:7] = R.from_rotvec(scaled_angle).as_quat()
    scaled_action[7] *= alpha
    return scaled_action

def unscale_action(scaled_action, alpha, rotation_factor = .1):
    unscaled_action = scaled_action.copy()
    unscaled_action[:3] /= alpha
    unscaled_angle = (R.from_quat(scaled_action[3:7]).as_rotvec() / (alpha * rotation_factor))
    unscaled_action[3:7] = R.from_rotvec(unscaled_angle).as_quat()
    unscaled_action[7] /= alpha
    return unscaled_action

def add_actions(act1, act2):
	result = np.zeros_like(act1)
	result[:3] = act1[:3] + act2[:3]
	result[3:7] = (R.from_quat(act1[3:7]) * R.from_quat(act2[3:7])).as_quat()
	result[7] = act1[7] + act2[7]
	return result

def subtract_actions(act1, act2):
	result = np.zeros_like(act1)
	result[:3] = act1[:3] - act2[:3]
	result[3:7] = (R.from_quat(act1[3:7]) * R.from_quat(act2[3:7]).inv()).as_quat()
	result[7] = act1[7] - act2[7]
	return result

class DummyVisionNode:
    @staticmethod
    def create_vision_node():
        return DummyVisionNode()
    
    def get_camera_images(self):
        # Return dummy images (480x640x3)
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        return [dummy_image, dummy_image], None
    
    def stop_cameras(self):
        pass

class EnvGUI(tk.Tk):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.title("Franka Gym Environment")
        self.geometry("300x400")
        self.reset_button = tk.Button(self, text="Reset Environment", command=self.reset_environment)
        self.reset_button.pack(pady=20)
        self.success_button = tk.Button(self, text="Episode Success", command=self.success_episode)
        self.success_button.pack(pady=20)
        self.fail_button = tk.Button(self, text="Episode Fail", command=self.fail_episode)
        self.fail_button.pack(pady=20)
        self.ready_button = tk.Button(self, text="Reset Ready", command=self.reset_ready)
        self.ready_button.pack(pady=20)
        self.quit_button = tk.Button(self, text="Quit", command=self.quit)
        self.quit_button.pack(pady=20)
    
    def reset_environment(self):
        self.env.manual_reset()
        print("Environment reset via GUI.")

    def success_episode(self):
        self.env.set_terminated(True, success=True)
        print("Episode marked as successful via GUI.")

    def fail_episode(self):
        self.env.set_truncated(True)
        print("Episode marked as failed via GUI.")

    def reset_ready(self):
        self.env.set_reset_ready(True)
        print("Environment reset ready via GUI.")

def run_gui(env):
    gui = EnvGUI(env)
    gui.mainloop()

class DummyFrankaGymEnvironment(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Dict({
            "world_image": gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
            "wrist_image": gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
            "pose": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        })
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        self.max_episode_steps = 200
        self.current_step = 0
    
    def reset(self, seed=0):
        self.current_step = 0
        self.pbar = tqdm(total=self.max_episode_steps)
        return self.observation_space.sample(), {}

    def step(self, action):
        self.current_step += 1
        truncated = self.current_step >= self.max_episode_steps
        self.pbar.update(1)
        return self.observation_space.sample(), 0, False, truncated, {}
    
class ReplayFrankaGymEnvironment(gym.Env):
    def __init__(self, replay_data):
        self.replay_data = replay_data
        self.observation_space = gym.spaces.Dict({
            "world_image": gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
            "wrist_image": gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
            "pose": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        })
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        self.current_index = 0
    
    def reset(self, seed=0):
        self.current_index = 0
        return self.replay_data['observations'][self.current_index], {}

    def step(self, action):
        self.current_index += 1
        done = self.current_index >= len(self.replay_data['observations']) - 1
        obs = self.replay_data['observations'][self.current_index]
        reward = self.replay_data['rewards'][self.current_index - 1]
        return obs, reward, done, False, {}

class FrankaGymEnvironment(gym.Env):
    def __init__(self, profile = "cartesian", render_mode=None, load_vision_node=True, use_gui=False, episode_timeout=200, reset_wait=0, rotation_max = 0.25*np.pi, wait_for_gui_reset=False):
        super(FrankaGymEnvironment, self).__init__()
        self.client_controller = FrankaClientController()
        assert profile in ["cartesian", "droid"], f"Invalid profile: {profile}"
        self.profile = profile
        # Profile is either "cartesion" for relative cartesian end effector control or "droid" for relative joint control from the droid dataset environment.
        if self.profile == "cartesian":
            # Observation space is a dict consisting of two 480x640 RGB images and an 8D end-effector pose plus gripper width
            self.observation_space = gym.spaces.Dict({
                "world_image": gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
                "wrist_image": gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
                "pose": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
            })
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)  # 3 for position, 4 for orientation (quaternion), 1 for gripper
            self.action_scale = np.array([0.05, 0.05, 0.05, 1, 1, 1, 1, 0.04])  # Scale for position (m), orientation (quat), gripper width (m)
            self.action_shift = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04])  # Shift for gripper width to be in [0, 0.08]
        elif self.profile == "droid":
            # Observation space is a dict consisting of two 480x640 RGB images and a 15D pose consisiting of joint positions, joint velocities, and gripper width
            self.observation_space = gym.spaces.Dict({
                "world_image": gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
                "wrist_image": gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
                "pose": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
            })
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)  # 7 for joint positions, 1 for gripper width
            self.action_scale = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.04])  # Scale for joint positions (rad), gripper width (m) (The droid dataset environment scales joint deltas to be in the range -.2 to .2)
            self.action_shift = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04])  # Shift for joint positions to be in [-1, 1], shift for gripper width to be in [0, 0.08]
        if load_vision_node:
            self.vision_node = VisionNode.create_vision_node()
            time.sleep(1)
        else:
            self.vision_node = DummyVisionNode.create_vision_node()
        
        self.reset_wait = reset_wait
        self.max_episode_steps = episode_timeout
        self.rotation_max = rotation_max
        self.wait_for_gui_reset = wait_for_gui_reset
        self.current_step = 0
        self.truncated = False
        self.terminated = False
        self.terminated_success = False
        self.reset_ready = False
        if use_gui:
            self.gui_thread = threading.Thread(target=run_gui, args=(self,))
            self.gui_thread.start()

    def _get_obs(self):
        pos, quat, joint_positions, joint_velocities, gripper_width = self.client_controller.get_state()
        pos, quat = np.array(pos), np.array(quat)
        rgb_images, _ = self.vision_node.get_camera_images()
        rgb_images = [img.astype(np.uint8) for img in rgb_images]
        # rgb_images = (np.zeros((480,640,3),dtype=np.uint8), np.zeros((480,640,3),dtype=np.uint8))  # Dummy images for faster testing
        if self.profile == "cartesian":
            return {
            "world_image": rgb_images[0],
            "wrist_image": rgb_images[1],
            "pose": np.concatenate([pos, quat, np.array([gripper_width])])
            }
        elif self.profile == "droid":
            # Flip the wrist image vertically
            # rgb_images[1] = cv2.flip(rgb_images[1], 0)
            return {
                "world_image": rgb_images[0],
                "wrist_image": rgb_images[1],
                "pose": np.concatenate([joint_positions, joint_velocities, np.array([gripper_width])])
            }
    def clip_action_for_safety(self, action):
        init_action = action.copy()
        action_rotvec = R.from_quat(action[3:7]).as_rotvec()
        rot_vec_magnitude = np.linalg.norm(action_rotvec)
        if rot_vec_magnitude > self.rotation_max:
            # print("Clipping rotation")
            action_rotvec = action_rotvec / rot_vec_magnitude * self.rotation_max
        action[3:7] = R.from_rotvec(action_rotvec).as_quat()
        action = np.clip(action, np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.08]))
        # print(f"Actions are close: {np.allclose(init_action, action)}")
        return action
    
    def manual_reset(self):
        self.client_controller.open_gripper()
        _ = self._get_obs()
        self.client_controller.go_home()
        _ = self._get_obs()

    def reset(self, seed=0):
        print(f"Resetting environment")
        self.current_step = 0
        self.truncated = False
        self.terminated = False
        self.terminated_success = False
        self.reset_ready = False
        self.client_controller.open_gripper()
        _ = self._get_obs()
        self.client_controller.go_home(droid_home=self.profile == "droid")
        _ = self._get_obs()
        if not self.wait_for_gui_reset:
            time.sleep(self.reset_wait)
        else:
            while not self.reset_ready:
                time.sleep(0.1)
        self.client_controller.null_command()
        observation = self._get_obs()
        # print(f"Obs pose shape: {observation['pose'].shape}")
        return observation, {}

    def step(self, action):
        # print(f"Received base action: {action}")
        pose_delta = action * self.action_scale + self.action_shift
        # pose_delta = np.zeros_like(pose_delta)
        # pose_delta[6:] = 1 #Send a valid quaternion and keep the gripper open
        if self.profile == "cartesian":
            if not check_quaternion_normalized(pose_delta[3:7]):
                pose_delta[3:7] = normalize_quaternion(pose_delta[3:7])
            pose_delta = self.clip_action_for_safety(pose_delta)

        # print(f"executed base action: {pose_delta}")

        if self.profile == "cartesian":
            self.client_controller.move_to_relative_pose(torch.from_numpy(pose_delta))
        elif self.profile == "droid":
            self.client_controller.move_to_relative_joint_pose(torch.from_numpy(pose_delta))
        observation = self._get_obs()
        reward = 0.0  # Define your reward function here
        reward += 1.0 if self.terminated_success else 0.0
        info = {"success": self.terminated_success}
        self.current_step += 1
        if self.current_step >= self.max_episode_steps:
            self.set_truncated(True)
        # print(f"Step: {self.current_step}, Truncated: {truncated} internal truncated: {self.truncated}, Reward: {reward}")
        return observation, reward, self.terminated, self.truncated, info

    def set_terminated(self, terminated, success=False):
        self.terminated = terminated
        self.terminated_success = success
        print(f"Set terminated to {terminated} with success={success}")

    def set_truncated(self, truncated):
        self.truncated = truncated
        print(f"Set truncated to {truncated}")

    def set_reset_ready(self, reset_ready):
        self.reset_ready = reset_ready
        print(f"Set reset_ready to {reset_ready}")

    def render(self, mode='human'):
        pass

    def close(self):
        # self.client_controller.shutdown()
        self.vision_node.stop_cameras()

def safety_test():
    env = FrankaGymEnvironment(load_vision_node=False, use_gui=True)
    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # Replace with your action selection logic
        scaled_action = scale_action(action, 0.1)
        obs, reward, terminated, truncated, info = env.step(scaled_action)
        done = terminated or truncated
        print(obs["pose"], done)

if __name__ == "__main__":
    # safety_test()
    # import sys; sys.exit()
    env = FrankaGymEnvironment(profile="droid", load_vision_node=True, use_gui=True)
    obs, info = env.reset()
    done = False
    try:
        while not done:
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # Replace with your action selection logic
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(obs["pose"], done)
    except KeyboardInterrupt:
        print("Shutting down Franka gym environment.")
    finally:
        env.close()