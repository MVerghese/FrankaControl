import Franka_Gym_Environment
import Dualsense
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import time
import os
import threading

class InputDeviceParser:
	def __init__(self, deadzone = 0.1, position_scale=.5, rotation_scale=.1):
		self.device = Dualsense.DualSense(reverse_xy=True)
		self.device.start_control()
		self.position_scale = position_scale
		self.rotation_scale = rotation_scale
		self.deadzone = deadzone

	def get_control(self):
		control, gripper = self.device.control, self.device.control_gripper
		control = np.array(control)
		control[np.abs(control) < self.deadzone] = 0  # Apply deadzone
		nonzero_input = np.any(control != 0) or np.any(gripper != 0)
		gripper = np.array(gripper)
		pos, rot = control[:3], control[3:]
		print(f"Raw control input: pos {pos}, rot {rot}, gripper {gripper}")
		pos[:2] = -pos[:2]  # Invert x and y axes
		rot[0], rot[1], rot[2] = rot[1]*-1, rot[0], rot[2]*-1  # Rearrange rotation axes

		pos = pos * self.position_scale  # Scale position
		rot = rot * self.rotation_scale  # Scale rotation
		rot = R.from_euler('xyz', rot, degrees=False).as_quat()  # Convert to quaternion
		gripper = gripper*-2 + 1
		# print(f"Position delta: {pos}, Rotation delta (euler): {control[3:]}, Rotation delta (quat): {rot}, Gripper: {gripper}")
		full_control = np.concatenate([pos, rot, np.array([gripper])])
		return full_control, nonzero_input

def run_data_collection(env,device_parser):
	obs, info = env.reset()
	done = False
	demo_data = []
	step_count = 0
	while not done:
		action, nonzero_input = device_parser.get_control()
		if not nonzero_input:
			time.sleep(0.1)
			continue  # Skip if no input
		next_obs, reward, terminated, truncated, _ = env.step(action)
		done = terminated or truncated
		demo_data.append((obs, action, reward, next_obs, done))
		obs = next_obs
		step_count += 1
		print(f"Step {step_count}: Action taken: {action}, Reward: {reward}, Done: {done}, Pose: {obs['pose']}")
	print("Episode complete.")
	return demo_data

def save_data(episode_data, episode_number, save_location):
	unzipped_data = list(zip(*episode_data))
	observations, actions, rewards, next_observations, dones = unzipped_data
	np.savez_compressed(os.path.join(save_location, f"demo_{episode_number}.npz"), observations=observations, actions=actions, rewards=rewards, next_observations=next_observations, dones=dones)
	print(f"Episode {episode_number} data saved.")

def save_data_threaded(episode_data, episode_number, save_location):
	thread = threading.Thread(target=save_data, args=(episode_data, episode_number, save_location))
	thread.start()
	return thread

def collect_demo_dataset(save_location, num_episodes=5, overwrite = False, episode_timeout=200, reset_wait=0, reset_gui = False):
	os.makedirs(os.path.dirname(save_location), exist_ok=True)

	start_episode = 0
	if not overwrite:
		existing_files = [f for f in os.listdir(save_location) if f.endswith(".npz") and f.startswith("demo_")]
		print(f"Found {len(existing_files)} existing demo files in {save_location}")
		if len(existing_files) > 0:
			existing_episodes = [int(f.split('_')[-1].split('.')[0]) for f in existing_files]
			start_episode = max(existing_episodes) + 1
			print(f"Resuming from episode {start_episode}")
	env = Franka_Gym_Environment.FrankaGymEnvironment(load_vision_node=True, use_gui=True, episode_timeout=episode_timeout, reset_wait=reset_wait, wait_for_gui_reset=reset_gui)
	device_parser = InputDeviceParser()
	try:
		saving = False
		for episode in range(start_episode, num_episodes):
			print(f"Starting episode {episode+1}/{num_episodes}")
			episode_data = run_data_collection(env, device_parser)
			# Save episode data
			unzipped_data = list(zip(*episode_data))
			observations, actions, rewards, next_observations, dones = unzipped_data
			if not reset_gui:
				env.reset()
			if saving:
				save_thread.join()
				saving = False
			save_thread = save_data_threaded(episode_data, episode, save_location)
			saving = True
			# np.savez_compressed(os.path.join(save_location, f"demo_{episode}.npz"), observations=observations, actions=actions, rewards=rewards, next_observations=next_observations, dones=dones)
		if saving:
			save_thread.join()
	except KeyboardInterrupt:
		print("Data collection interrupted by user.")
	finally:
		env.close()

def main():
	collect_demo_dataset(save_location="./Test/", num_episodes=21, overwrite=False, episode_timeout=200, reset_wait=2, reset_gui=True)

if __name__ == "__main__":
	main()