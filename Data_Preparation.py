import numpy as np
from matplotlib import pyplot as plt
import os
import Reward_Inference
import cv2
from tqdm import tqdm
import RL_Models
import torch

import sys
sys.path.append("/home/mverghese/ego_env/Franka_Kitchen_Env")
import BehaviorCloning
sys.path.remove("/home/mverghese/ego_env/Franka_Kitchen_Env")

sys.path.append("/home/mverghese/ego_env")
from Inference import MPRInference
from Preprocess_Data import Image_Annotator
sys.path.remove("/home/mverghese/ego_env")

from RL_Models import get_image_encoder
from Franka_Gym_Environment import unscale_action, subtract_actions
from stable_baselines3 import RLPD_SAC
from Unpack_episode_data import unpack_episode_video
from typing import Dict



def add_gripper_observations(data_path: str, obs_key: str = "observations", force = False):
	gripper_latency_steps = 5
	gripper_travel_steps = 14
	data = np.load(data_path, allow_pickle=True)
	observations = data[obs_key]
	pose_stack = np.stack([obs['pose'] for obs in observations])
	print(f"Pose stack shape: {pose_stack.shape}")
	# assert pose_stack.shape[1] == 7, "This routine is for cleaning old data that is missing gripper width information."
	if pose_stack.shape[1] != 7:
		print(f"Data at {data_path} under key {obs_key} already has gripper width information. Skipping.")
		print(f"Data example:\n {pose_stack[0]}")
		if pose_stack.shape[1] == 8 and force:
			print(f"Forcing re-computation of gripper width. Removing existing gripper width column.")
			pose_stack = pose_stack[:, :-1]
		else:
			return

	actions = data['actions']

	# Add an extra column of zeros to the pose stack for gripper width
	gripper_widths = np.zeros((pose_stack.shape[0], 1))
	pose_stack = np.concatenate([pose_stack, gripper_widths], axis=1)

	gripper_width = 0.08
	prev_gripper_action = 1.0
	latency_countdown = 0
	travel_countdown = 0
	for i in range(pose_stack.shape[0]):
		gripper_action = actions[i, 7]
		target_gripper_width = 0.0 if gripper_action < 0 else 0.08
		if gripper_action != prev_gripper_action:
			latency_countdown = gripper_latency_steps
			travel_countdown = gripper_travel_steps
			prev_gripper_action = gripper_action
			
		if latency_countdown > 0:
			latency_countdown -= 1
		elif travel_countdown > 0:
			step_size = 0.08 / gripper_travel_steps * gripper_action
			gripper_width += step_size
			travel_countdown -= 1
		else:
			gripper_width = target_gripper_width
		pose_stack[i, 7] = gripper_width
		prev_gripper_action = gripper_action

	pose_stack[:, 7] = np.clip(pose_stack[:, 7], 0.0000585, 0.0797)

	# plt.plot(pose_stack[:, 7], label="Gripper Width")
	# plt.plot(actions[:, 7]*0.04 + 0.04, label="Gripper Action Scaled")
	# plt.xlabel("Time Step")
	# plt.ylabel("Gripper Width")
	# plt.title("Gripper Width Over Time")
	# plt.legend()
	# plt.show()

	# Update observations with new pose including gripper width
	for i, obs in enumerate(observations):
		obs['pose'] = pose_stack[i]

	# Data is an npz file, create a new dictionary with the updated observations
	data_dict = {key: data[key] for key in data.files}
	data_dict[obs_key] = observations

	# Save the updated data with the data dictionary
	np.savez_compressed(data_path, **data_dict)
	print(f"Updated observations with gripper width in {data_path}")

def prep_data_for_RLPD(data_folder: str, destination_folder: str, dp_checkpoint: str, reward_mode: str, alpha: float = 0.2, vip_goal_path: str = "", MPR_Checkpoint_Path: str = "", obj_language_tag: list = [], obs_preprocess_encoder: str = ""):

	base_policy = BehaviorCloning.DP_Inference(
		model_path=dp_checkpoint,
		obs_dim=8,
		action_dim=8,
		device="cuda:0",
		num_diffusion_iters=50,
		is_transformer=False,
		vision_model = True
	)

	if reward_mode == "VIP":
		inference = Reward_Inference.VIP_Inference(
			device="cuda:0",
		)
		goal_image = cv2.imread(vip_goal_path)
		inference.set_goal_image(goal_image)
	elif reward_mode == "MPR":
		inference = MPRInference(MPR_Checkpoint_Path, obj_language_tag, device="cuda:0")

	if obs_preprocess_encoder:
		encoder_device = "cuda:1"
		image_encoder, preprocess, encode_dim = get_image_encoder(obs_preprocess_encoder)
		image_encoder.to(encoder_device)


	data_files = [f for f in os.listdir(data_folder) if f.endswith('.npz')]
	data_files.sort()

	data = np.load(os.path.join(data_folder, data_files[0]), allow_pickle=True)
	obs_keys = data['observations'][0].keys()
	obs_keys = list(obs_keys)
	obs_keys.append("base_policy_actions")
	obs_keys = sorted(obs_keys)
	print(f"Obs keys: {obs_keys}")

	if not obs_preprocess_encoder:
		obs = {key: [] for key in obs_keys}
		next_obs = {key: [] for key in obs_keys}
	else:
		obs_dim = 0
		for key in obs_keys:
			if "image" in key:
				obs_dim += encode_dim
			elif key == "base_policy_actions":
				obs_dim += 8  # Assuming base policy actions are 8-dimensional
			else:
				print(f"Key: {key}, Shape: {data['observations'][0][key].shape}")
				obs_dim += data['observations'][0][key].shape[0]
		print(f"Combined obs shape: {obs_dim}")
		obs = np.zeros((0, obs_dim))
		next_obs = np.zeros((0, obs_dim))
	acts = []
	rews = []
	dns = []

	for data_file in tqdm(data_files):
		data_path = os.path.join(data_folder, data_file)
		data = np.load(data_path, allow_pickle=True)
		observations = data['observations']
		actions = data['actions']
		next_observations = data['next_observations']
		rewards = data['rewards']
		dones = data['dones']
		world_images = np.array([obs["world_image"] for obs in observations])
		if reward_mode == "MPR":
			final_image = next_observations[-1]["world_image"]
			world_images = np.concatenate((world_images, final_image[None]), axis=0)

		rewards += inference.video_inference(world_images)
		# Copy the init obs to prevent modification by reference
		init_obs = observations[0].copy()
		base_policy.reset(init_obs)
		residual_actions = []
		base_actions = []
		for i in range(len(actions)):
			base_action = base_policy.act(observations[i].copy())
			base_actions.append(base_action)
			res_action = unscale_action(subtract_actions(actions[i], base_action), alpha, rotation_factor=.5)
			# print(f"Base action: {base_action}, Recorded action: {actions[i]}, Action diff: {actions[i] - base_action}, Residual action: {res_action}")
			residual_actions.append(res_action)
		acts.extend(residual_actions)

		# add the episode data to the buffers
		if not obs_preprocess_encoder:
			for key in obs_keys:
				if key == "base_policy_actions":
					obs[key].extend(base_actions)
					next_obs[key].extend(base_actions)
				else:
					obs[key].extend([o[key] for o in observations])
					next_obs[key].extend([o[key] for o in next_observations])
		else:
			obs_stack = []
			next_obs_stack = []
			for key in obs_keys:
				if "image" in key:
					obs_image_stack = np.array([o[key] for o in observations])
					next_obs_image_stack = np.array([o[key] for o in next_observations])
					with torch.no_grad():
						obs_preprocess = preprocess(torch.from_numpy(obs_image_stack).float().permute(0,3,1,2)).to(encoder_device)
						obs_encoded = image_encoder(obs_preprocess)
						next_obs_preprocess = preprocess(torch.from_numpy(next_obs_image_stack).float().permute(0,3,1,2)).to(encoder_device)
						next_obs_encoded = image_encoder(next_obs_preprocess)
					obs_stack.append(obs_encoded.cpu().numpy())
					next_obs_stack.append(next_obs_encoded.cpu().numpy())
				elif key == "base_policy_actions":
					obs_stack.append(np.array(base_actions))
					next_obs_stack.append(np.array(base_actions))
				else:
					obs_stack.append(np.array([o[key] for o in observations]))
					next_obs_stack.append(np.array([o[key] for o in next_observations]))
			for o in obs_stack:
				print(f"Obs component shape: {o.shape}")
			# import pdb; pdb.set_trace()
			obs_stack = np.hstack(obs_stack)
			next_obs_stack = np.hstack(next_obs_stack)
			print(f"Obs shape: {obs_stack.shape}")
			obs = np.vstack((obs, obs_stack))
			next_obs = np.vstack((next_obs, next_obs_stack))

		
		

		rews.extend(rewards)
		dns.extend(dones)

	if not obs_preprocess_encoder:
		for key in obs_keys:
			obs[key] = np.array(obs[key])
			next_obs[key] = np.array(next_obs[key])

	actions = np.array(actions)
	rewards = np.array(rewards)
	dones = np.array(dones)

	os.makedirs(destination_folder, exist_ok=True)
	if not obs_preprocess_encoder:
		np.savez_compressed(os.path.join(destination_folder, "obs.npz"), **obs)
		np.savez_compressed(os.path.join(destination_folder, "next_obs.npz"), **next_obs)
	else:
		np.save(os.path.join(destination_folder, "obs.npy"), obs)
		np.save(os.path.join(destination_folder, "next_obs.npy"), next_obs)
	np.save(os.path.join(destination_folder, "actions.npy"), acts)
	np.save(os.path.join(destination_folder, "rewards.npy"), rews)
	np.save(os.path.join(destination_folder, "dones.npy"), dns)

def compute_reward_and_value(data_path: str, agent_checkpoint: str, reward_mode: str, vip_goal_path: str = "", MPR_Checkpoint_Path: str = "", obj_language_tag: list = [], ):

	if reward_mode == "VIP":
		inference = Reward_Inference.VIP_Inference(
			device="cuda:0",
		)
		goal_image = cv2.imread(vip_goal_path)
		inference.set_goal_image(goal_image)
	elif reward_mode == "MPR":
		inference = MPRInference(MPR_Checkpoint_Path, obj_language_tag, device="cuda:0")


	data = np.load(data_path, allow_pickle=True)

	observations = data['observations']
	actions = data['actions']
	rewards = data['rewards']
	images = data['images']


	rewards = inference.video_inference(images)


	agent_device = "cuda:0"
	agent = RLPD_SAC.load(agent_checkpoint, device=agent_device)
	with torch.no_grad():
		obs = torch.from_numpy(observations).float().to(agent_device)
		actions = torch.from_numpy(actions).float().to(agent_device)
		values, _ = torch.min(torch.cat(agent.critic(obs, actions), dim=1), dim=1, keepdim=True)
		values = values.squeeze(-1).cpu().numpy()

	return rewards, values, images

def white_mask_images(image_folder, save_folder, start_frame=0):
	image_files = [f for f in os.listdir(image_folder) if f.endswith('.png') or f.endswith('.jpg')]
	image_files.sort()
	image_stack = [cv2.imread(os.path.join(image_folder, f))[:,:,::-1].copy() for f in image_files]
	image_stack = np.array(image_stack)
	os.makedirs(save_folder, exist_ok=True)

	annotator = Image_Annotator()
	masks = annotator.select_and_track(image_stack, initial_frame=start_frame)
	# Set all pixels in the mask to white for each image in the stack
	for i, mask in enumerate(masks):
		image_stack[i][mask == 1] = 255
		cv2.imwrite(os.path.join(save_folder, f"masked_{i:04d}.png"), cv2.cvtColor(image_stack[i], cv2.COLOR_RGB2BGR))

	# plt.imshow(masked_image)
	# plt.show()

if __name__ == "__main__":
	# data_folder = '/home/mverghese/franka_control/open_microwave_data_20/'
	# npz_files = [f for f in os.listdir(data_folder) if f.endswith('.npz')]
	# for npz_file in npz_files:
	# 	data_path = os.path.join(data_folder, npz_file)
	# 	add_gripper_observations(data_path, obs_key="observations", force=True)
	# 	add_gripper_observations(data_path, obs_key="next_observations", force=True)


	prep_data_for_RLPD(
		data_folder='/home/mverghese/franka_control/wipe_counter_10_alt',
		destination_folder='/home/mverghese/franka_control/wipe_counter_10_alt/RLPD_VIP/',
		dp_checkpoint='/home/mverghese/franka_control/wipe_counter_10_alt/dp_model_epoch_500.pth',
		reward_mode='VIP',
		alpha=0.1,
		vip_goal_path='/home/mverghese/franka_control/wipe_counter_10_alt/VIP_Goal_Frame.png',
		MPR_Checkpoint_Path="/home/mverghese/ego_env/Wipe_Counter_Checkpoints_PPT_L/checkpoint_epoch_1000.pth",
		obj_language_tag=['cloth'],
		obs_preprocess_encoder='dinov2_vits14'
	)

	prep_data_for_RLPD(
		data_folder='/home/mverghese/franka_control/wipe_counter_10_alt',
		destination_folder='/home/mverghese/franka_control/wipe_counter_10_alt/RLPD_MPR/',
		dp_checkpoint='/home/mverghese/franka_control/wipe_counter_10_alt/dp_model_epoch_500.pth',
		reward_mode='MPR',
		alpha=0.1,
		vip_goal_path='/home/mverghese/franka_control/wipe_counter_10_alt/VIP_Goal_Frame.png',
		MPR_Checkpoint_Path="/home/mverghese/ego_env/Wipe_Counter_Checkpoints_PPT_L/checkpoint_epoch_1000.pth",
		obj_language_tag=['cloth'],
		obs_preprocess_encoder='dinov2_vits14'
	)
	# vip_agent_checkpoint = '/home/mverghese/franka_control/runs/open_microwave/VIP/open microwave__reward_mode_VIP_offline_ratio_0.5_seed_0/checkpoints/model_checkpoint_100_episodes_16379_steps.zip'
	# mpr_agent_checkpoint = '/home/mverghese/franka_control/runs/open_microwave/MPR/open microwave__reward_mode_MPR_offline_ratio_0.5_seed_0/checkpoints/model_checkpoint_100_episodes_13395_steps.zip'
	# reward_mode = 'VIP'
	# data_path = '/home/mverghese/franka_control/VIP_Seed_0_Examples/fail/demo_2.npz'

	# rewards, values, images = compute_reward_and_value(
	# 	data_path=data_path,
	# 	agent_checkpoint=vip_agent_checkpoint if reward_mode == 'VIP' else mpr_agent_checkpoint,
	# 	reward_mode=reward_mode,
	# 	vip_goal_path='/home/mverghese/franka_control/open_microwave_data_10/VIP_goal.png',
	# 	MPR_Checkpoint_Path="/home/mverghese/ego_env/checkpoints/Open_Microwave_PPT_L_Checkpoint.pth",
	# 	obj_language_tag=['microwave'],
	# )
	# print(f"Rewards: {rewards}")
	# print(f"Values: {values}")

	# np.savez_compressed(data_path[:-4]+f"_{reward_mode}_rewards_values.npz", rewards=rewards, values=values)

	# rewards, values, images = compute_reward_and_value(
	# 	data_path='/home/mverghese/franka_control/VIP_Seed_0_Examples/fail/demo_1.npz',
	# 	agent_checkpoint='/home/mverghese/franka_control/runs/open_microwave/MPR/open microwave__reward_mode_MPR_offline_ratio_0.5_seed_0/checkpoints/model_checkpoint_100_episodes_13395_steps.zip',
	# 	reward_mode='MPR',
	# 	vip_goal_path='/home/mverghese/franka_control/open_microwave_data_10/VIP_goal.png',
	# 	MPR_Checkpoint_Path="/home/mverghese/ego_env/checkpoints/Open_Microwave_PPT_L_Checkpoint.pth",
	# 	obj_language_tag=['microwave'],
	# )
	# print(f"Rewards: {rewards}")
	# print(f"Values: {values}")

	# np.savez_compressed('/home/mverghese/franka_control/VIP_Seed_0_Examples/fail/demo_1_MPR_rewards_values.npz', rewards=rewards, values=values)

	# os.makedirs(data_path[:-4] + f"_frames", exist_ok=True)
	# for i, image in enumerate(images):
	# 	# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	# 	cv2.imwrite(data_path[:-4] + f"_frames/frame_{i:04d}.png", image)

	# white_mask_images('/home/mverghese/franka_control/VIP_Seed_0_Examples/fail/demo_2_frames_masked', '/home/mverghese/franka_control/VIP_Seed_0_Examples/fail/demo_2_frames_masked', 46)
