import numpy as np
import cv2
import os

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

def main():
    episode_file = "./open_microwave_data/demo_0.npz"
    output_video_file = "./open_microwave_data/demo_0_video.mp4"
    
    frames = unpack_episode_video(episode_file)
    save_video(frames, output_video_file, fps=30)
    print(f"Video saved to {output_video_file}")

if __name__ == "__main__":
    main()