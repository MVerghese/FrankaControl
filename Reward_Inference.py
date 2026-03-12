import torch
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
import cv2

from vip import load_vip

def load_video_frames(video_path):
	cap = cv2.VideoCapture(video_path)
	frames = []
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
		frames.append(frame)
	cap.release()
	return frames  

def get_VIP_Preprocess(): 
	## DEFINE PREPROCESSING
	preprocess_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Resize(256),
		transforms.CenterCrop(224),
		]) # ToTensor() divides by 255

	return preprocess_transform

class VIP_Inference:
	def __init__(self, device: str = "cuda:0"):
		self.device = device
		self.vip = load_vip().to(self.device)
		self.vip.eval()
		self.preprocess = get_VIP_Preprocess()
		print(f"VIP model loaded on {self.device}")

	def set_goal_image(self, goal_image: np.ndarray):
		self.goal_image = goal_image
		# print(f"Goal image shape: {self.goal_image.shape}")

		with torch.no_grad():
			self.goal_features = self.vip(self.preprocess(self.goal_image).to(self.device).unsqueeze(0)*255.0).squeeze(0)
		self.goal_features = self.goal_features.cpu().numpy()

	def video_inference(self, video_frames: np.ndarray):
		video_frames = [self.preprocess(frame).to(self.device) for frame in video_frames]
		video_frames = torch.stack(video_frames)
		with torch.no_grad():
			video_features = self.vip(video_frames * 255.0).squeeze(0)
		video_features = video_features.cpu().numpy()
		# print(f"Video features shape: {video_features.shape}, Goal features shape: {self.goal_features.shape}")
		distance = np.linalg.norm(video_features - self.goal_features, axis=1)
		return distance * -1

def main():
	video_path = "/home/mverghese/franka_control/open_microwave_data_10/demo_0_video.mp4"

	video_frames = load_video_frames(video_path)
	video_frames = np.array(video_frames)

	goal_image = video_frames[-1]
	cv2.imwrite("/home/mverghese/franka_control/open_microwave_data_10/VIP_goal.png", cv2.cvtColor(goal_image, cv2.COLOR_RGB2BGR))

	vip_inference = VIP_Inference()
	vip_inference.set_goal_image(goal_image)

	distance = vip_inference.video_inference(video_frames)
	print(f"Distance for video: {distance}")
	plt.plot(distance)
	plt.xlabel("Frame")
	plt.ylabel("Distance")
	plt.title("Distance between Video Frames and Goal Image")
	plt.show()

if __name__ == "__main__":
	main()

