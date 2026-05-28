import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from stable_baselines3 import RLPD_SAC
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer
from Callbacks import EpisodeCheckpointCallback

import wandb
from wandb.integration.sb3 import WandbCallback

import numpy as np
import torch
import torch.nn as nn
import Franka_Gym_Environment
from Franka_Gym_Environment import scale_action, add_actions, normalize_quaternion
import RL_Models
from Reward_Inference import VIP_Inference


import os
import time
from tqdm import tqdm
from typing import Union, List, Dict, Tuple, Optional, Any
import json
import cv2


import sys
sys.path.append("/home/mverghese/ego_env/Franka_Kitchen_Env")
import BehaviorCloning
sys.path.remove("/home/mverghese/ego_env/Franka_Kitchen_Env")
sys.path.append("/home/mverghese/ego_env")
from Inference import MPRInference
sys.path.remove("/home/mverghese/ego_env")

def unwrap_env_to_specific_wrapper(env, wrapper_class):
	assert issubclass(wrapper_class, gym.Wrapper), "wrapper_class must be a subclass of gym.Wrapper"
	while not isinstance(env, wrapper_class):
		env = env.env
	return env


class BasePolicyWrapper(gym.Wrapper):
	def __init__(self, env: gym.Env, policy: BehaviorCloning.DP_Inference , alpha: float = .1, position_control: bool = False):
		super().__init__(env)
		self.policy = policy
		# self.observation_space["base_policy_actions"] = Box(low=-1, high=1, shape=(self.env.action_space.shape[0],), dtype=np.float64)
		self.observation_space = gym.spaces.Dict({
			**self.env.observation_space,
			"base_policy_actions": gym.spaces.Box(low=-1, high=1, shape=(self.env.action_space.shape[0],), dtype=np.float64)
		})
		self.prev_obs = None
		self.alpha = alpha
		self.position_control = position_control
		self.next_action = None

	def get_base_action(self):
		# start_time = time.time()
		base_action = self.policy.act(self.prev_obs)
		base_action = base_action.astype(np.float64)
		if self.position_control:
			current_robot_pos = self.prev_obs[:8]
			joint_deltas = base_action - current_robot_pos
			# print(f"Joint deltas: {joint_deltas}")
			joint_vels = joint_deltas / self.env.unwrapped.robot_env.dt
			base_action = joint_vels
		# print(f"Compute base action took {time.time() - start_time:.4f} seconds")
		# print(f"Computed Base Action: {base_action}")
		return base_action

	def reset(self, **kwargs):
		obs, info = self.env.reset(**kwargs)
		# print(f"Obs pose shape before base policy: {obs['pose'].shape}")

		# Copy the observation to a new dictionary
		self.prev_obs =  obs.copy()
		self.policy.reset(self.prev_obs)
		self.next_action = self.get_base_action()
		obs["base_policy_actions"] = self.next_action
		info["next_base_action"] = self.next_action
		info["base_policy_alpha"] = self.alpha
		# print(f"Obs pose shape after base policy: {obs['pose'].shape}")

		return obs, info

	def step(self, action):
		# print(f"Base policy action: {self.next_action}, input action: {action}, scaled input: {scale_action(action, self.alpha)}")
		# print(f"Wrapper base action: {add_actions(scale_action(action, self.alpha), self.next_action)}")
		action[3:7] = normalize_quaternion(action[3:7])
		obs, reward, terminated, truncated, info = self.env.step(add_actions(scale_action(action, self.alpha), self.next_action))
		self.prev_obs = obs.copy()
		self.next_action = self.get_base_action()
		obs["base_policy_actions"] = self.next_action
		info["next_base_action"] = self.next_action
		info["base_policy_alpha"] = self.alpha
		return obs, reward, terminated, truncated, info

class SuccessLoggingWrapper(gym.Wrapper):
	def __init__(self, env, save_path):
		super().__init__(env)
		self.successes = []
		self.episode_timesteps = []
		self.current_ep_success = False
		self.save_path = save_path
		self.episode_steps = 0

	def reset(self, **kwargs):
		obs, info = self.env.reset(**kwargs)
		if self.episode_steps > 0:
			self.successes.append(self.current_ep_success)
			self.episode_timesteps.append(self.episode_steps)
			print(f"Episode success: {self.current_ep_success}. Avg success over last 20 runs: {np.mean(self.successes[-20:])}")
			# Save the running success array
			with open(self.save_path, "w") as f:
				json.dump({"successes": self.successes, "timesteps": self.episode_timesteps}, f)
		self.current_ep_success = False
		self.episode_steps = 0
		return obs, info

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action)
		if info.get("success", False):
			self.current_ep_success = True
		self.episode_steps += 1
		return obs, reward, terminated, truncated, info

class ImageCachingWrapper(gym.Wrapper):
	def __init__(self, env):
		super().__init__(env)
		self.episode_cache = []
		self.image_cache = []
		self.episode_steps = 0
		self.epsiode_steps_list = []

	def reset(self, **kwargs):
		obs, info = self.env.reset(**kwargs)
		if len(self.image_cache) > 0:
			self.episode_cache.append(self.image_cache)
			self.image_cache = []
		self.image_cache.append(obs['world_image'].copy())
		self.epsiode_steps_list.append(self.episode_steps)
		self.episode_steps = 0
		return obs, info

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action)
		self.image_cache.append(obs['world_image'].copy())
		self.episode_steps += 1
		return obs, reward, terminated, truncated, info
	
	def get_episode_caches(self, add_current: bool = True):
		if add_current and len(self.image_cache) > 0:
			return self.episode_cache + [self.image_cache], self.epsiode_steps_list + [self.episode_steps]
		return self.episode_cache, self.epsiode_steps_list

	def clear_episode_caches(self, clear_current: bool = True):
		self.episode_cache = []
		self.epsiode_steps_list = []
		if clear_current:
			self.image_cache = [self.image_cache[-1]] if self.image_cache else []
			self.episode_steps = 0

class ImageEncoderWrapper(gym.ObservationWrapper):
	def __init__(self, env, device = "cuda:1", arch = "ResNet-10"):
		super().__init__(env)
		self.device = device
		# self.image_encoder = RL_Models.get_pretrained_resnet10().to(self.device)
		self.image_encoder, self.preprocess, embed_dim = RL_Models.get_image_encoder(arch)
		self.image_encoder = self.image_encoder.to(device)
		self.image_encoder.eval()
		space_size = 0
		self.obs_keys = self.env.observation_space.keys()
		print(f"Observation keys before encoding: {self.obs_keys}")
		self.obs_keys = sorted(self.obs_keys)
		for k in self.obs_keys:
			if "image" in k:
				space_size += embed_dim 
			else:
				space_size += self.env.observation_space[k].shape[0]
		print(f"Observation keys after encoding: {self.obs_keys}, total encoded space size: {space_size}")
		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(space_size,), dtype=np.float32)

	def encode_obs(self, obs):
		# Pass the image through the encoder
		# start_time = time.time()
		encoded_features = []
		image_stack = []
		for k in self.obs_keys:
			with torch.no_grad():
				if "image" in k:
					with torch.no_grad():
						preprocessed = self.preprocess(torch.from_numpy(obs[k]).float().permute(2,0,1).unsqueeze(0).to(self.device))
						encoded = self.image_encoder(preprocessed)
					encoded_features.append(encoded.cpu().numpy().flatten())
				else:
					encoded_features.append(obs[k])
		out =  np.concatenate(encoded_features)
		# print(f"Encoded observation shape: {out.shape}")
		end_time = time.time()
		# print(f"Encoding time: {end_time - start_time:.4f} seconds")
		return out

	def reset(self, **kwargs):
		obs, info = self.env.reset(**kwargs)
		obs = self.encode_obs(obs)
		return obs, info

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action)
		obs = self.encode_obs(obs)
		return obs, reward, terminated, truncated, info
	
class ImageEncodingCallback(BaseCallback):
	def __init__(self, env_obs_space, env_act_space, buffer_size = 1000000, device = "cuda:0", arch = "ResNet-10"):
		super(ImageEncodingCallback, self).__init__(verbose)
		self.image_encoder, self.preprocess = RL_Models.get_image_encoder(arch)
		self.image_encoder = self.image_encoder.to(device)
		self.image_encoder.eval()
		self.flat_space_size = 0
		for k in env_obs_space.keys():
			if "image" in k:
				self.flat_space_size += 512  # Assuming the encoder outputs 512-dim features
			else:
				self.flat_space_size += env_obs_space[k].shape[0]
		flat_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.flat_space_size,), dtype=np.float32)
		self.replay_buffer = ReplayBuffer(
			buffer_size,
			flat_space,
			env_act_space,
			device = device,
			n_envs = 1,
			optimize_memory_usage = False
		)
		self.env_buffer = None


	def _on_training_start(self) -> None:
		"""
		This method is called before the first rollout starts.
		"""
		self.env_buffer = self.locals["replay_buffer"]

	def _on_rollout_start(self) -> None:
		"""
		A rollout is the collection of environment interaction
		using the current policy.
		This event is triggered before collecting new samples.
		"""
		self.locals["replay_buffer"] = self.env_buffer

	def _on_step(self) -> bool:
		"""
		This method will be called by the model after each call to `env.step()`.

		For child callback (of an `EventCallback`), this will be called
		when the event is triggered.

		:return: (bool) If the callback returns False, training is aborted early.
		"""
		return True

	def _on_rollout_end(self) -> None:
		"""
		This event is triggered before updating the policy.
		"""
		self.env_buffer = self.locals["replay_buffer"]

		observations = []
		next_observations = []
		for key in self.env_buffer.observations.sorted().keys():
			if "image" in key:
				image_obs = self.env_buffer.observations[key]
				torch_obs = torch.from_numpy(image_obs).float().permute(0, 3, 1, 2)
				encoded = self.image_encoder(self.preprocess(torch_obs)).cpu().numpy()
				observations.append(encoded)
				next_image_obs = self.env_buffer.next_observations[key]
				torch_next_obs = torch.from_numpy(next_image_obs).float().permute(0, 3, 1, 2)
				encoded_next = self.image_encoder(self.preprocess(torch_next_obs)).cpu().numpy()
				next_observations.append(encoded_next)
			else:
				observations.append(self.env_buffer.observations[key])
				next_observations.append(self.env_buffer.next_observations[key])

		observations = np.hstack(observations)
		next_observations = np.hstack(next_observations)
		assert self.replay_buffer.size + self.env_buffer.size <= self.replay_buffer.buffer_size, "ImageEncodingCallback does not support full replay buffers."
		pos = self.replay_buffer.pos # This is encoded buffer position
		size = self.env_buffer.size # This is the number of new samples
		self.replay_buffer.observations[pos:pos+size] = observations
		self.replay_buffer.next_observations[pos:pos+size] = next_observations
		self.replay_buffer.actions[pos:pos+size] = self.env_buffer.actions[:size]
		self.replay_buffer.rewards[pos:pos+size] = self.env_buffer.rewards[:size]
		self.replay_buffer.dones[pos:pos+size] = self.env_buffer.dones[:size]
		self.replay_buffer.timeouts[pos:pos+size] = self.env_buffer.timeouts[:size]
		self.replay_buffer.size += size

		self.env_buffer.reset()

	def _on_training_end(self) -> None:
		"""
		This event is triggered before exiting the `learn()` method.
		"""
		pass

class BatchedRewardCallback(BaseCallback):
	def __init__(self, inference, verbose=0, reward_offset=-1):
		super(BatchedRewardCallback, self).__init__(verbose)
		self.inference = inference
		self.reward_offset = reward_offset

	def _on_training_start(self) -> None:
		pass

	def _on_rollout_start(self) -> None:
		pass

	def _on_step(self) -> bool:
		return True

	def _on_rollout_end(self) -> None:
		"""
		This event is triggered before updating the policy.
		"""
		replay_buffer = self.locals["replay_buffer"]
		if replay_buffer.full:
			raise ValueError("BatchedMPRRewardCallback does not support full replay buffers.")
		
		assert self.training_env.num_envs == 1, "BatchedMPRRewardCallback only supports single environment training."

		caching_wrapper = unwrap_env_to_specific_wrapper(self.training_env.envs[0].env, ImageCachingWrapper)
		episode_caches, episode_steps = caching_wrapper.get_episode_caches()
		reward = self.inference.video_inference(episode_caches[-2]) # reset is called before this callback so the last cache is empty
		reward = np.array(reward) + self.reward_offset
		reward = reward[:episode_steps[-2]]
		# 1/0
		replay_buffer.rewards[replay_buffer.pos - episode_steps[-2]:replay_buffer.pos] += reward.reshape((-1,1))

		# Clear the episode caches
		caching_wrapper.clear_episode_caches(clear_current=True)


	def _on_training_end(self) -> None:
		pass
	

def make_env(base_policy_checkpoint: str, reward_mode: str, alpha: float = 0.1, save_path: str = "successes.json", obs_encoder: str = "", episode_timeout: int = 400, reset_wait: int = 2, replay_env_data_path="", gui_reset = False) -> gym.Env:
	# Note, non sparse rewards are handled by the batched reward callback. This
	# function only sets up the ImageCachingWrapper to enable the callback
	if replay_env_data_path:
		replay_data = np.load(replay_env_data_path, allow_pickle=True)
		env = Franka_Gym_Environment.ReplayFrankaGymEnvironment(replay_data)
	else:
		env = Franka_Gym_Environment.FrankaGymEnvironment(load_vision_node=True, use_gui=True, episode_timeout=episode_timeout, reset_wait=reset_wait, wait_for_gui_reset=gui_reset)
	# env = Franka_Gym_Environment.DummyFrankaGymEnvironment()
	if base_policy_checkpoint is not None:
		action_dim = env.action_space.shape[0]
		inference = BehaviorCloning.DP_Inference(
			model_path=base_policy_checkpoint,
			obs_dim=8,
			action_dim=action_dim,
			device="cuda:0",
			num_diffusion_iters=50,
			is_transformer=False,
			vision_model = True
		)
		env = BasePolicyWrapper(env, inference, alpha=alpha, position_control=False)
		print(f"Using base policy from {base_policy_checkpoint} with alpha {alpha}")

	env = SuccessLoggingWrapper(env, save_path=save_path)

	if reward_mode == "sparse":
		pass  # use the default sparse reward
	elif reward_mode == "VIP" or reward_mode == "MPR":
		env = ImageCachingWrapper(env)

	if obs_encoder:
		env = ImageEncoderWrapper(env, arch=obs_encoder)
		print("Using image encoder wrapper to encode observations")

	return env

def get_default_RLPD_SAC_params() -> dict:
	return {
		"buffer_size" : int(1e6),
		"batch_size" : 1024,
		"learning_starts" : 1024,
		"gradient_steps" : 1,
		"learning_rate" : 3e-4,
		"gamma" : .99,
		"offline_ratio" : 0.5,
		"offline_buffer_path" : None,
		"policy_kwargs" : {"net_arch" : [256, 256, 256]},

	}

def run_experiment(task: str, timesteps: int, run_args: dict, env_args: dict, device: str = "",  log_wandb : bool = True, folder: str = "runs/Franka_Kitchen_Env/default_folder"):

	# Pretrain iterations is a manually handled run arg and cannot be passed to stable_baselines3
	pretrain_iterations = 0
	if "pretrain_iterations" in run_args:
		pretrain_iterations = run_args["pretrain_iterations"]
		del run_args["pretrain_iterations"]

	# use diffusion policy encoders is a manually handled run arg and cannot be passed to stable_baselines3
	use_dp_encoders = False
	if "use_dp_encoders" in run_args.keys():
		use_dp_encoders = run_args["use_dp_encoders"]
		del run_args["use_dp_encoders"]
		

	# Alpha is a run arg but it is implemented as an env arg, so move it from run args to env args and then delete it from run args
	if "alpha" in run_args.keys():
		env_args["alpha"] = run_args["alpha"]
		del run_args["alpha"]

	model_checkpoint = ""
	if "model_checkpoint" in run_args.keys():
		model_checkpoint = run_args["model_checkpoint"]
		del run_args["model_checkpoint"]
		steps = int(model_checkpoint.split("_")[-2])
		episodes = int(model_checkpoint.split("_")[-4])

	buffer_checkpoint = ""
	if "buffer_checkpoint" in run_args.keys():
		buffer_checkpoint = run_args["buffer_checkpoint"]
		del run_args["buffer_checkpoint"]
		steps = int(buffer_checkpoint.split("_")[-2])
		episodes = int(buffer_checkpoint.split("_")[-4])

	# Get the default args for the specified rl_algorithm
	args = get_default_RLPD_SAC_params()

	# Update the default args with the run args
	if "policy_kwargs" in run_args.keys():
		args["policy_kwargs"].update(run_args["policy_kwargs"])
		del run_args["policy_kwargs"]
	
	args.update(run_args)
	

	# If using diffusion policy encoders, we need to modify the environment args
	if use_dp_encoders:
		args["policy_kwargs"].update({"features_extractor_kwargs": {"DP_Checkpoint": env_args["base_policy_checkpoint"]}})

	experiment_name = f"{task}__reward_mode_{env_args['reward_mode']}_offline_ratio_{run_args['offline_ratio']}"
	print(f"Experiment name: {experiment_name}")
	print(f"Starting experiment with parameters: {args}")

	if device == "":
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print(f"automatically selected device: {device}")
	
	args["device"] = device

	print(f"Path to save results: {os.path.join(folder, experiment_name, 'checkpoints')}")
	if not os.path.exists(os.path.join(folder, experiment_name, "checkpoints")):
		os.makedirs(os.path.join(folder, experiment_name, "checkpoints"))

	checkpoint_callback = EpisodeCheckpointCallback(
		save_freq=10,
		save_path=os.path.join(folder, experiment_name, "checkpoints"),
		name_prefix="model_checkpoint",
		save_replay_buffer=True,
	)
	callbacks = [checkpoint_callback]
	if env_args["reward_mode"] == "MPR":
		MPR_Checkpoint_Path = env_args["MPR_Checkpoint_Path"]
		obj_language_tag = env_args["obj_language_tag"]
		del env_args["MPR_Checkpoint_Path"]
		del env_args["obj_language_tag"]
		mpr_inference = MPRInference(MPR_Checkpoint_Path, obj_language_tag, device="cuda:1")
		callbacks.append(BatchedRewardCallback(mpr_inference))
	elif env_args["reward_mode"] == "VIP":
		vip_goal_path = env_args["vip_goal_path"]
		del env_args["vip_goal_path"]
		goal_image = cv2.imread(vip_goal_path)
		goal_image = cv2.cvtColor(goal_image, cv2.COLOR_BGR2RGB)
		vip_inference = VIP_Inference(device="cuda:0")
		vip_inference.set_goal_image(goal_image)
		callbacks.append(BatchedRewardCallback(vip_inference, reward_offset = 0))

	# Get the policy type depending on whether the environment encodes the observation
	obs_encoder = env_args.get("obs_encoder", "")
	policy_type = "MlpPolicy" if obs_encoder else "MultiInputPolicy"


	if log_wandb:
		wandb_config = {
			"policy_type": "MlpPolicy",
			"total_timesteps": timesteps,
			"env_name": "Franka_Kitchen",
		}
		run = wandb.init(
			project="Franka Real",
			name=f"{experiment_name}",
			config=wandb_config,
			sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
			monitor_gym=False,  # auto-upload the videos of agents playing the game
			save_code=True,  # optional
		)
		wandb_callback = WandbCallback(
			gradient_save_freq=100,
			verbose=2,
		)
		callbacks.append(wandb_callback)
	
	env_args["save_path"] = os.path.join(folder, experiment_name, "successes.json")

	env = make_env(**env_args)



	model = RLPD_SAC(policy_type,env,verbose=0, tensorboard_log=os.path.join(folder, experiment_name, "tensorboard_logs"), **args)

	pretrain = True
	if model_checkpoint:
		model.load(model_checkpoint)
		pretrain = False
	if buffer_checkpoint:
		model.load_replay_buffer(buffer_checkpoint)
		pretrain = False

	if pretrain_iterations > 0 and pretrain:
		model.learn(total_timesteps = 0, progress_bar=False, )
		model.train(pretrain_iterations, batch_size = args["batch_size"], only_critic=True)
		print("Completed Pretraining")
		model.save(os.path.join(folder, experiment_name, "checkpoints", "model_checkpoint_0_steps.zip"))
	else:
		checkpoint_callback.episode_counter = episodes
		model.num_timesteps = steps
		checkpoint_callback.num_timesteps = steps

	try:
		model.learn(total_timesteps = timesteps, progress_bar=True, callback=callbacks)
	except KeyboardInterrupt:
		print("Training interrupted")
	finally:
		model.save(os.path.join(folder, experiment_name, "checkpoints", "model_checkpoint_final.zip"))
	print("Completed Training")


	env.close()

class SB3AgentWrapper:
	def __init__(self, agent):
		self.agent = agent

	def act(self, obs):
		action, _ = self.agent.predict(obs)
		return action

class DummyWrapper:
	def __init__(self):
		pass

	def act(self, obs):
		return np.array([0., 0., 0., 0., 0., 0., 1., 0.])

	
def eval_policy(env: gym.Env, agent_checkpoint: str = "", runs: int = 10) -> float:
	if agent_checkpoint:
		agent_model = RLPD_SAC.load(agent_checkpoint, device="cuda:0")
		agent = SB3AgentWrapper(agent_model)
	else:
		agent = DummyWrapper()
	successes = 0
	for _ in tqdm(range(runs)):
		obs, _ = env.reset()
		print("Reset environment")
		terminated = False
		truncated = False
		step_count = 0
		while not terminated and not truncated:
			with torch.no_grad():
				action = agent.act(obs)
			obs, reward, terminated, truncated, info = env.step(action)
			step_count += 1
			if terminated:
				successes += 1
		print(f"Episode finished in {step_count} steps. Success: {terminated}")
		obs, _ = env.reset()
	success_rate = successes / runs
	print(f"Evaluation completed. Success rate: {success_rate:.2f}")
	return success_rate

def save_data(episode_data, images, episode_number, save_location):
	unzipped_data = list(zip(*episode_data))
	observations, actions, rewards, dones = unzipped_data
	np.savez_compressed(os.path.join(save_location, f"demo_{episode_number}.npz"), observations=observations, actions=actions, rewards=rewards, dones=dones, images=images)
	print(f"Episode {episode_number} data saved.")

def get_success_and_fail_examples(env: gym.Env, agent_checkpoint: str, save_path: str, runs: int = 10) -> None:
	os.makedirs(os.path.join(save_path, "success"), exist_ok=True)
	os.makedirs(os.path.join(save_path, "fail"), exist_ok=True)
	# env = ImageCachingWrapper(env)
	agent_model = RLPD_SAC.load(agent_checkpoint, device="cuda:0")
	agent = SB3AgentWrapper(agent_model)
	for i in tqdm(range(runs)):
		obs, _ = env.reset()
		print("Reset environment")
		terminated = False
		truncated = False
		step_count = 0
		demo_data = []
		success = False
		while not terminated and not truncated:
			with torch.no_grad():
				action = agent.act(obs)
			obs, reward, terminated, truncated, info = env.step(action)
			demo_data.append((obs, action, reward, terminated or truncated))
			step_count += 1
			if terminated:
				success = True
		cache_wrapper = unwrap_env_to_specific_wrapper(env, ImageCachingWrapper)
		episode_caches, episode_steps = cache_wrapper.get_episode_caches()
		assert episode_steps[-1] > 0
		episode_images = episode_caches[-1]
		if success:
			save_data(demo_data, episode_images, i, os.path.join(save_path, "success"))
		else:
			save_data(demo_data, episode_images, i, os.path.join(save_path, "fail"))
		cache_wrapper.clear_episode_caches()
		print(f"Episode finished in {step_count} steps. Success: {terminated}")
		obs, _ = env.reset()



def main():
	base_policy_checkpoint = "/home/mverghese/franka_control/fold_cloth_20/dp_model_final.pth"
	# base_policy_checkpoint = "/home/mverghese/franka_control/open_microwave_10/dp_model_epoch_1000.pth"
	# env = make_env(base_policy_checkpoint, reward_mode='sparse', obs_encoder="dinov2_vits14", episode_timeout=200, reset_wait=4)
	env = make_env(base_policy_checkpoint, reward_mode='sparse', obs_encoder="dinov2_vits14", episode_timeout=200, reset_wait=2)
	agent_checkpoint = "/home/mverghese/franka_control/runs/wipe_counter/VIP/wipe counter__reward_mode_VIP_offline_ratio_0.5_seed_2/checkpoints/model_checkpoint_100_episodes_16580_steps.zip"
	print("Environment created successfully.")
	try:
		sr  = eval_policy(env, runs = 20)
		# print(f"Success rate: {sr:.2f}")
		# get_success_and_fail_examples(env, agent_checkpoint, save_path="VIP_Seed_0_Examples/", runs=20)
		# sr = eval_policy(env, agent_checkpoint, runs=20)
		if 'success_rates.json' in os.listdir():
			with open('success_rates.json', 'r') as f:
				success_dict = json.load(f)
		else:
			success_dict = {}
		success_dict[base_policy_checkpoint] = sr
		with open('success_rates.json', 'w') as f:
			json.dump(success_dict, f)
	finally:
		env.close()
	import sys; sys.exit(0)
	env_args = {
		"reward_mode" : "VIP", 
		"base_policy_checkpoint" : "/home/mverghese/franka_control/wipe_counter_10_alt/dp_model_epoch_500.pth",
		"vip_goal_path": "/home/mverghese/franka_control/wipe_counter_10_alt/VIP_Goal_Frame.png",
		"obs_encoder": "dinov2_vits14",
		"episode_timeout": 200,
		"reset_wait": 4,
		# "MPR_Checkpoint_Path": "/home/mverghese/ego_env/Wipe_Counter_Checkpoints_PPT_L/checkpoint_epoch_1000.pth",
		# "obj_language_tag": ["cloth", "towel"],
		"gui_reset": False
		# "replay_env_data_path": "/home/mverghese/franka_control/fold_cloth_data_20/demo_0.npz"

	}
	run_args = {
		"buffer_size": 200*100,
		"alpha": 0.1, 
		"ent_coef": 0.05, 
		"offline_ratio": 0.5, 
		"offline_buffer_path": "/home/mverghese/franka_control/wipe_counter_10_alt/RLPD_VIP", 
		"train_freq":(1, "episode"), 
		"gradient_steps": -4,
		"learning_rate" : 1e-4,
		"pretrain_iterations": 10000,
		"use_dp_encoders": False,
		# "model_checkpoint": "/home/mverghese/franka_control/runs/wipe_counter/MPR/wipe counter__reward_mode_MPR_offline_ratio_0.5/checkpoints/model_checkpoint_80_episodes_13809_steps.zip",
		# "buffer_checkpoint": "/home/mverghese/franka_control/runs/wipe_counter/MPR/wipe counter__reward_mode_MPR_offline_ratio_0.5/checkpoints/model_checkpoint_replay_buffer_80_episodes_13809_steps.pkl"

	}
	run_experiment("wipe counter", 200*100, run_args, env_args, device="cuda:0", log_wandb=True, folder="runs/wipe_counter/VIP")




if __name__ == "__main__":
	main()