from openpi_client import image_tools
from openpi_client import websocket_client_policy
import numpy as np

class PiModelInference:
	def __init__(self, host="localhost", port=8000):
		self.client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
		self.task_instruction = ""
		self.execution_horizon = 8 # The number of steps to execute the action sequence for
		self.action_buffer = []
	
	def set_task_instruction(self, task_instruction):
		self.task_instruction = task_instruction

	def predict_action_sequence(self, observation):
		# Observation is a dictionary consisting of a 480x640 RGB world image (world_image), a 480x640 RGB wrist image (wrist_image), and a 15D pose (pose) consisting of joint positions, joint velocities, and gripper width.

		# Map the observation to the OpenPI input format
		openpi_observation = {
			"observation/exterior_image_1_left": image_tools.convert_to_uint8(
				image_tools.resize_with_pad(observation["world_image"], 224, 224)
			),
			"observation/wrist_image_left": image_tools.convert_to_uint8(
				image_tools.resize_with_pad(observation["wrist_image"], 224, 224)
			),
			"observation/joint_position": observation["pose"][:7],
			"observation/gripper_position": observation["pose"][14],
			"prompt": self.task_instruction,
		}

		# Call the policy server with the current observation.
		# This returns an action chunk of shape (action_horizon, action_dim).
		# Note that you typically only need to call the policy every N steps and execute steps
		# from the predicted action chunk open-loop in the remaining steps.
		action_chunk = self.client.infer(openpi_observation)["actions"][:self.execution_horizon, :].copy()
		# The last dim of action is the gripper width, if its greater than 0.5, set it to 1.0, otherwise set it to -1.0
		action_chunk[:, -1] = np.where(action_chunk[:, -1] > 0.5, 1.0, -1.0)
		return action_chunk
	
	def act(self, observation):
		# If the action buffer is empty, predict a new action sequence
		if len(self.action_buffer) == 0:
			self.action_buffer = self.predict_action_sequence(observation)
		# Return the first action in the buffer and remove it from the buffer
		action = self.action_buffer[0]
		# Remove the first action from the buffer
		self.action_buffer = self.action_buffer[1:,:]
		return action
	
	def reset(self):
		self.action_buffer = []

def main():
	pi_model_inference = PiModelInference(host="localhost", port=8000)
	# Outside of episode loop, initialize the policy client.
	# Point to the host and port of the policy server (localhost and 8000 are the defaults).
	for step in range(20):
		# observation = {
		#     "observation/exterior_image_1_left": image_tools.convert_to_uint8(
		#         image_tools.resize_with_pad(img, 224, 224)
		#     ),
		#     "observation/wrist_image_left": image_tools.convert_to_uint8(
		#         image_tools.resize_with_pad(wrist_img, 224, 224)
		#     ),
		#     "observation/joint_position": state,
		#     "observation/gripper_position": gripper_position,
		#     "prompt": task_instruction,
		# }
		observation = {
			"world_image": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
			"wrist_image": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
			"pose": np.random.rand(15),
		}
		# Call the policy server with the current observation.
		# This returns an action chunk of shape (action_horizon, action_dim).
		# Note that you typically only need to call the policy every N steps and execute steps
		# from the predicted action chunk open-loop in the remaining steps.
		action = pi_model_inference.act(observation)
		print(action)

if __name__ == "__main__":
	main()