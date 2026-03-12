from OpenPi_Interface import PiModelInference
from Franka_Gym_Environment import FrankaGymEnvironment
import time

def run_episode(inference, environment, task_instruction):
    inference.set_task_instruction(task_instruction)
    observation , _ = environment.reset()
    done = False
    prev_time = time.time()
    while not done:
        print(f"Time taken: {time.time() - prev_time}")
        prev_time = time.time()
        action = inference.act(observation)
        print(f"Action: {action}")
        observation, reward, terminated, truncated, info = environment.step(action)
        done = terminated or truncated

if __name__ == "__main__":
    pi_model_inference = PiModelInference(host="localhost", port=8000)
    franka_gym_environment = FrankaGymEnvironment(profile="droid", load_vision_node=False, use_gui=True)
    run_episode(pi_model_inference, franka_gym_environment, "Pick up the apple.")