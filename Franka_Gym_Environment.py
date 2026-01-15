import VisionNode
from Franka_Client_Controller import FrankaClientController
import gymnasium as gym
import numpy as np
import torch
import tkinter as tk
import threading

def check_quaternion_normalized(quat):
    norm = np.linalg.norm(quat)
    return np.isclose(norm, 1.0, atol=1e-3)

def normalize_quaternion(quat):
    norm = np.linalg.norm(quat)
    if norm < 1e-6:
        raise ValueError("Cannot normalize a zero-length quaternion.")
    return quat / norm

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
        self.geometry("300x300")
        self.reset_button = tk.Button(self, text="Reset Environment", command=self.reset_environment)
        self.reset_button.pack(pady=20)
        self.success_button = tk.Button(self, text="Episode Success", command=self.success_episode)
        self.success_button.pack(pady=20)
        self.fail_button = tk.Button(self, text="Episode Fail", command=lambda: self.env.set_truncated(True, success=False))
        self.fail_button.pack(pady=20)
        self.quit_button = tk.Button(self, text="Quit", command=self.quit)
        self.quit_button.pack(pady=20)
    
    def reset_environment(self):
        self.env.reset()
        print("Environment reset via GUI.")

    def success_episode(self):
        self.env.set_truncated(True, success=True)
        print("Episode marked as successful via GUI.")

    def fail_episode(self):
        self.env.set_truncated(True, success=False)
        print("Episode marked as failed via GUI.")

def run_gui(env):
    gui = EnvGUI(env)
    gui.mainloop()

class FrankaGymEnvironment(gym.Env):
    def __init__(self, render_mode=None, load_vision_node=True, use_gui=False):
        super(FrankaGymEnvironment, self).__init__()
        self.client_controller = FrankaClientController()
        if load_vision_node:
            self.vision_node = VisionNode.create_vision_node()
        else:
            self.vision_node = DummyVisionNode.create_vision_node()
        # Observation space is a dict consisting of a 480x640 RGB image and a 7D end-effector pose
        self.observation_space = gym.spaces.Dict({
            "world_image": gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
            "wrist_image": gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
            "pose": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        })
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)  # 3 for position, 4 for orientation (quaternion), 1 for gripper
        self.action_scale = np.array([0.05, 0.05, 0.05, 1, 1, 1, 1, 0.04])  # Scale for position (m), orientation (quat), gripper width (m)
        self.action_shift = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04])  # Shift for gripper width to be in [0, 0.08]
        self.max_episode_steps = 300
        self.current_step = 0
        self.truncated = False
        self.truncated_success = False
        if use_gui:
            self.gui_thread = threading.Thread(target=run_gui, args=(self,))
            self.gui_thread.start()

    def _get_obs(self):
        pos, quat = self.client_controller.get_state()
        pos, quat = np.array(pos), np.array(quat)
        rgb_images, _ = self.vision_node.get_camera_images()
        rgb_images = [img.astype(np.uint8) for img in rgb_images]
        # rgb_images = (np.zeros((480,640,3),dtype=np.uint8), np.zeros((480,640,3),dtype=np.uint8))  # Dummy images for faster testing
        return {
            "world_image": rgb_images[0],
            "wrist_image": rgb_images[1],
            "pose": np.concatenate([pos, quat])
        }

    def reset(self):
        self.current_step = 0
        self.truncated = False
        self.truncated_success = False
        self.client_controller.go_home()
        observation = self._get_obs()
        return observation, {}

    def step(self, action):
        pose_delta = action * self.action_scale + self.action_shift
        # pose_delta = np.zeros_like(pose_delta)
        # pose_delta[6:] = 1 #Send a valid quaternion and keep the gripper open
        if not check_quaternion_normalized(pose_delta[3:7]):
            pose_delta[3:7] = normalize_quaternion(pose_delta[3:7])

        self.client_controller.move_to_relative_pose(torch.from_numpy(pose_delta))
        observation = self._get_obs()
        reward = 0.0  # Define your reward function here
        terminated = False  # Define your termination condition here
        truncated = self.truncated
        reward += 1.0 if self.truncated_success else 0.0
        info = {}
        self.current_step += 1
        if self.current_step >= self.max_episode_steps:
            terminated = True
        # print(f"Step: {self.current_step}, Truncated: {truncated} internal truncated: {self.truncated}, Reward: {reward}")
        return observation, reward, terminated, truncated, info
    
    def set_truncated(self, truncated, success=False):
        self.truncated = truncated
        self.truncated_success = success
        print(f"Set truncated to {truncated} with success={success}")

    def render(self, mode='human'):
        pass

    def close(self):
        # self.client_controller.shutdown()
        self.vision_node.stop_cameras()

if __name__ == "__main__":
    env = FrankaGymEnvironment(load_vision_node=False, use_gui=True)
    obs, info = env.reset()
    done = False
    try:
        while not done:
            action = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0])  # Replace with your action selection logic
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(obs["pose"], done)
    except KeyboardInterrupt:
        print("Shutting down Franka gym environment.")
    finally:
        env.close()