import numpy as np
import cv2
import os
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt


def unpack_episode_video(episode_file):
    data = np.load(episode_file, allow_pickle=True)
    world_images = [obs["world_image"] for obs in data['observations']]
    return world_images


def save_video(frames, output_file, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width = frames[0].shape[:2]
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()

def get_angle_magnitudes(episode_folder):
    episode_files = [f for f in os.listdir(episode_folder) if f.endswith('.npz')]
    angle_magnitudes = []
    for episode_file in episode_files:
        data = np.load(os.path.join(episode_folder, episode_file), allow_pickle=True)
        angles = [action[3:7] for action in data['actions']]
        rot_vecs = [R.from_quat(angle).as_rotvec() for angle in angles]
        angle_magnitudes.extend([np.linalg.norm(rot) for rot in rot_vecs])
    return angle_magnitudes

def sample_random_angles(num_samples=1000):
    sampled_quats = np.random.uniform(low=-1.0, high=1.0, size=(num_samples, 4))
    normalized_quats = sampled_quats / np.linalg.norm(sampled_quats, axis=1)[:, np.newaxis]
    # print(f"Normalized quats: {normalized_quats}")
    rot_vecs = [R.from_quat(quat).as_rotvec() for quat in normalized_quats]
    angle_magnitudes = [np.linalg.norm(rot) for rot in rot_vecs]
    return angle_magnitudes

def get_position_magnitudes(episode_folder):
    episode_files = [f for f in os.listdir(episode_folder) if f.endswith('.npz')]
    position_magnitudes = []
    for episode_file in episode_files:
        data = np.load(os.path.join(episode_folder, episode_file), allow_pickle=True)
        positions = [action[:3] for action in data['actions']]
        position_magnitudes.extend([np.linalg.norm(pos) for pos in positions])
    return position_magnitudes



def main():
    episode_file = "/home/mverghese/franka_control/wipe_counter_10_alt/demo_0.npz"
    # output_video_file = "/home/mverghese/franka_control/Wipe_Counter/demo_1_video.mp4"

    frames = unpack_episode_video(episode_file)
    cv2.imwrite("/home/mverghese/franka_control/wipe_counter_10_alt//VIP_Goal_Frame.png", frames[-1])
    # save_video(frames, output_video_file, fps=30)
    # print(f"Video saved to {output_video_file}")
    # angle_magnitudes = sample_random_angles()
    # print(f"max magnitude: {np.max(angle_magnitudes)}, min magnitude: {np.min(angle_magnitudes)}")
    # plt.hist(angle_magnitudes, bins=30)
    # plt.xlabel("Rotation Angle Magnitude (radians)")
    # plt.show()


    # angle_magnitudes = get_position_magnitudes("/home/mverghese/franka_control/open_microwave_data_20")
    # print(f"max magnitude: {np.max(angle_magnitudes)}, min magnitude: {np.min(angle_magnitudes)}")
    # plt.hist(angle_magnitudes, bins=30)
    # plt.xlabel("Position Magnitude (meters)")
    # plt.show()

if __name__ == "__main__":
    main()