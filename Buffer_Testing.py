import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.preprocessing import get_obs_shape
from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer




def main():
    obs_space = spaces.Dict({
        "world_image": spaces.Box(low=0, high=255, shape=(3, 240, 320), dtype=np.uint8),
        "wrist_image": spaces.Box(low=0, high=255, shape=(3, 240, 320), dtype=np.uint8),
        "pose": spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
    })
    action_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)

    replay_buffer = DictReplayBuffer(
        buffer_size=100000,
        observation_space=obs_space,
        action_space=action_space,
        device="cuda:0"
    )

if __name__ == "__main__":
    main()