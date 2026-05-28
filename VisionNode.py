import time
import cv2
import pyrealsense2 as rs
import numpy as np
from skimage.measure import label, regionprops
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import maximum_filter
import os
import threading
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from matplotlib.path import Path
import sys
from utils import *



np.set_printoptions(suppress=True)





cov_1 = np.eye(3)*np.array([1,1,5])
cov_2 = np.eye(3)*np.array([1,1,5])
cov_3 = np.eye(3)*np.array([1,1,5])


def get_coords(intr,pixel_coords,depth_image,mask=np.array([]),mode = "closest"):
	if mask.any():
		depths = depth_image[mask[:,0].flatten(),mask[:,1].flatten()]
		if(depths[np.nonzero(depths)].shape[0] > 0):
			if mode == "closest":
				depth = np.min(depths[np.nonzero(depths)])
			elif mode == "average":
				depth = np.mean(depths[np.nonzero(depths)])
			elif mode == "farthest":
				depth = np.max(depths[np.nonzero(depths)])
			else:
				print("invalid depth mode")
				depth = 0

		else:
			print("invalid depth")
			depth = 0

	else:
		depth = depth_image[pixel_coords]

	x = (pixel_coords[1] - intr[0])/intr[2] *depth
	y = (pixel_coords[0] - intr[1])/intr[3] *depth
	return(np.array([x,y,depth]))

def batch_coords(intr,pixel_coords,depth_image):
	out_coords =np.zeros((pixel_coords.shape[0],3))
	out_coords[:,2] = depth_image[pixel_coords[:,0].flatten(),pixel_coords[:,1].flatten()]
	out_coords[:,0] = (pixel_coords[:,1] - intr[0])/intr[2] *out_coords[:,2]
	out_coords[:,1] = (pixel_coords[:,0] - intr[1])/intr[3] *out_coords[:,2]
	return(out_coords)

def expand_mask(mask, kernel_size=3):
	kernel = np.ones((kernel_size,kernel_size))
	expanded_mask = cv2.dilate(mask,kernel,iterations=1)
	return(expanded_mask)

def decode_and_save_recording(recording_buffer,save_path, fps = 30):
	assert len(recording_buffer) > 0
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	print("Saving recording to {}".format(save_path))
	first_image = cv2.imdecode(recording_buffer[0], cv2.IMREAD_UNCHANGED)
	height, width = first_image.shape[:2]
	video_name = os.path.join(save_path,"recording.mp4")
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	video_writer = cv2.VideoWriter(video_name, fourcc, fps, (width, height))
	for i in range(len(recording_buffer)):
		image = cv2.imdecode(recording_buffer[i], cv2.IMREAD_UNCHANGED)
		video_writer.write(image)
	video_writer.release()


class VisionNode:
	def __init__(self,camera_list,extrinsics=None,intrinsics=None,display_camera = True, save_path = "", update_frequency = 24, load_OWL = False):
		self.camera_list = camera_list
		self.num_cams = len(camera_list)
		self.pipelines = []
		self.cam_configs = []
		self.aligns = []
		if extrinsics is None:
			extrinsics = [np.eye(4)]*self.num_cams
		
		self.extrinsics = extrinsics
		
		if intrinsics is None:
			intrinsics = [np.zeros(4)]*self.num_cams

		self.intrinsics = intrinsics
		self.intrinsics_mat = []
		for intr in intrinsics:
			mat = build_intr_mat(intr)
			self.intrinsics_mat.append(mat)
		self.color_images = [None]*self.num_cams
		self.depth_images = [None]*self.num_cams
		self.depth_colormaps = [None]*self.num_cams
		self.camera_rotations = [None]*self.num_cams
		self.stopped = False

		self.display_camera = display_camera

		
		
		self.save_path = save_path
		self.save_counter = 0
		if self.save_path != "" and not os.path.exists(self.save_path):
			os.makedirs(self.save_path)
		
		self.update_frequency = update_frequency

		if load_OWL:
			from OWL import OWLDetector
			self.OWL = OWLDetector(sam_checkpoint="/home/mverghese/sam2/checkpoints/sam2.1_hiera_large.pt", device="cuda:0")
		else:
			self.OWL = None

		aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
		parameters = cv2.aruco.DetectorParameters()
		self.aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

		self.recording_buffer = [[]] * self.num_cams
		self.recording = [False] * self.num_cams
		self.last_call_time = 0
		self.average_fps = 0
	

		self._vision_thread = threading.Thread(target=self._vision_thread)
		self._mutex = threading.Lock()

		print("reset start")
		ctx = rs.context()
		devices = ctx.query_devices()
		for dev in devices:
			print("Resetting device: {}".format(dev.get_info(rs.camera_info.name)))
			dev.hardware_reset()
		time.sleep(3)
		print("reset done")

	def _vision_thread(self):
		while not self.stopped:
			
			self.update_frames(render_depth_colormap = True)
			if self.display_camera:
				self.render_camera_views()
			# self._mutex.acquire()
			# if self.save_path != "":
			# 	self.save_frames()
			# self._mutex.release()
			time.sleep(1./self.update_frequency)
		cv2.destroyAllWindows()
	
	def save_frames(self):
		for i in range(len(self.color_images)):
			bgr = cv2.cvtColor(self.color_images[i], cv2.COLOR_RGB2BGR)
			cv2.imwrite(os.path.join(self.save_path,"color_image_"+str(i).zfill(2)+"_"+str(self.save_counter).zfill(4)+".png"),bgr)
		self.save_counter += 1

	def start_recording(self,cameras):
		self._mutex.acquire()
		if isinstance(cameras,int):
			cameras = [cameras]
		for cam in cameras:
			self.recording_buffer[cam] = []
			self.recording[cam] = True
			print("Starting recording for camera {}".format(cam))
		self._mutex.release()

	def stop_recording(self,cameras):
		self._mutex.acquire()
		if isinstance(cameras,int):
			cameras = [cameras]
		ret_buffer = []
		for cam in cameras:
			self.recording[cam] = False
			print("Stopping recording for camera {}".format(cam))
			if len(self.recording_buffer[cam]) > 0:
				ret_buffer.append(self.recording_buffer[cam][:])
				self.recording_buffer[cam] = []
			else:
				print("No frames recorded for camera {}".format(cam))
				ret_buffer.append([])
		self._mutex.release()
		if len(ret_buffer) == 1:
			return(ret_buffer[0], self.average_fps)
		else:
			return(ret_buffer, self.average_fps)


	def start_cameras(self,camera_ids=None,exposure=0,resolution=(640,480)):
		if camera_ids is None:
			camera_ids = self.camera_list
		for cam, cid in enumerate(camera_ids):
			print("Starting camera {}: {}".format(cam,cid))
			cam_pipeline = rs.pipeline()
			cam_config = rs.config()
			cam_config.enable_device(cid)
			cam_config.enable_stream(rs.stream.depth, resolution[0], resolution[1], rs.format.z16, 30)
			cam_config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.bgr8, 30)
			align = rs.align(rs.stream.color)
			cfg = cam_pipeline.start(cam_config)
			if exposure > 0:
				color_sensor = cfg.get_device().query_sensors()[1]
				color_sensor.set_option(rs.option.exposure, exposure)

			self.pipelines.append(cam_pipeline)
			self.cam_configs.append(cam_config)
			self.aligns.append(align)
		
		self.check_intrinsics()

		self._vision_thread.start()
	
	def check_intrinsics(self):
		for cam in range(self.num_cams):
			if np.all(self.intrinsics[cam]==0):
				intr = self.query_intrinsics(cam)
				self.intrinsics[cam] = [intr.ppx, intr.ppy, intr.fx, intr.fy]
				self.intrinsics_mat[cam] = build_intr_mat(self.intrinsics[cam])
				print("Setting camera {} intrinsics: {}".format(cam,self.intrinsics[cam]))


	def stop_cameras(self):
		self.stopped = True
		if self._vision_thread.is_alive():
			self._vision_thread.join()
		print("Vision thread stopped")
		for pipeline in self.pipelines:
			pipeline.stop()
	
	def query_intrinsics(self, cam):
		intr = self.pipelines[cam].get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
		return(intr)
	
	def query_all_intrinsics(self, cam):
		for cam in range(len(self.pipelines)):
			intr = self.query_intrinsics(cam)
			print("Camera {} intrinsics: {}".format(cam,intr))

	def update_frames(self,render_depth_colormap = False):
		self._mutex.acquire()
		for i in range(len(self.pipelines)):
			while True:
				try:
					start_time = time.time()
					frames = self.pipelines[i].wait_for_frames()
					depth_frame = frames.get_depth_frame()
					color_frame = frames.get_color_frame()
					frame_time = time.time() - start_time
					if frame_time > 0.5:
						print("Warning: Camera {} frame wait took {:.2f} seconds".format(i,frame_time))
				except RuntimeError:
					print("Camera {} failed to send a frame".format(i))
					depth_frame = self.depth_images[i]
					color_frame = self.color_images[i]
				if depth_frame and color_frame:
					break
			color_image = np.asanyarray(color_frame.get_data())
			frameset = self.aligns[i].process(frames)
			aligned_depth_frame = frameset.get_depth_frame()
			aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())
			color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
			self.color_images[i] = color_image
			self.depth_images[i] = aligned_depth_image
			if self.camera_rotations[i] != None:
				self.color_images[i] = cv2.rotate(self.color_images[i],self.camera_rotations[i])
				self.depth_images[i] = cv2.rotate(self.depth_images[i],self.camera_rotations[i])

			if render_depth_colormap:
				depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_images[i], alpha=0.5), cv2.COLORMAP_JET)
				self.depth_colormaps[i] = depth_colormap
			
			if self.recording[i]:
				bgr = cv2.cvtColor(self.color_images[i], cv2.COLOR_RGB2BGR)
				encoded_im = cv2.imencode('.png', bgr)[1]
				self.recording_buffer[i].append(encoded_im)
		self._mutex.release()
		if np.any(self.recording):
			if self.last_call_time == 0:
				self.last_call_time = time.time()
			else:
				delta_time = time.time() - self.last_call_time
				self.last_call_time = time.time()
				if self.average_fps == 0:
					self.average_fps = 1/delta_time
				else:
					self.average_fps = self.average_fps*0.9 + (1/delta_time)*0.1
				

	def render_camera_views(self):
		cols = []
		for cam in range(self.num_cams):
			bgr = cv2.cvtColor(self.color_images[cam], cv2.COLOR_RGB2BGR)
			cols.append(np.vstack((bgr, self.depth_colormaps[cam])))
		images = np.hstack(cols)
		# print(images.shape,images.dtype)
		cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
		cv2.imshow('RealSense', images)
		cv2.waitKey(1)

	def get_camera_images(self):
		while not np.any(self.color_images) or not np.any(self.depth_images):
			time.sleep(.001)
		return(self.color_images.copy(),self.depth_images.copy())

	def render_camera_views_threaded(self,fps = 30.):
		while not self.stopped:
			self.render_camera_views()
			time.sleep(1./fps)
	
	def get_rgb_image(self,cam):
		image = self.color_images[cam]
		# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		return(image)
		


	def world_frame_point_cloud(self,mask,cam,robot_pose):
		points = batch_coords(self.intrinsics[cam],mask,self.depth_images[cam])
		print(points.shape)
		homogenous_points = np.vstack((points.T,np.ones((1,points.shape[0]))))
		cam_pose = np.dot(robot_pose, self.extrinsics[self.robot_camera])
		world_points = np.dot(cam_pose,homogenous_points).T
		return(world_points[:,:-1])


	def get_point_cloud(self,cam,depth_max = 1000, depth_min = 200):
		depth_image = self.depth_images[cam]

		color_image = self.color_images[cam]
		u = np.arange(depth_image.shape[0])
		v = np.arange(depth_image.shape[1])
		uu,vv = np.meshgrid(u,v)
		coords = np.hstack((uu.reshape((-1,1)),vv.reshape((-1,1))))
		coords = [coord for coord in coords if depth_image[int(coord[0]),int(coord[1])] < depth_max and depth_image[int(coord[0]),int(coord[1])] > depth_min]
		coords = np.array(coords)
		points = batch_coords(self.intrinsics[cam],coords,depth_image)

		colors = color_image[coords[:,0].astype(int),coords[:,1].astype(int)]
		return(points, colors)

	def detect_aruco(self,cam):
		timeout_counter = 0
		while self.color_images[cam] is None:
			time.sleep(.1)
			timeout_counter += 1
			if timeout_counter > 50:
				print("Detect Aruco timed out while waiting for an image")
				return(None, None)
		gray = cv2.cvtColor(self.color_images[cam],cv2.COLOR_RGB2GRAY)
		corners, ids, _ = self.aruco_detector.detectMarkers(gray)
		return(corners,ids)

	def compute_marker_pose(self,cam,marker_id, marker_size = 100):
		corners, ids = self.detect_aruco(cam)
		if marker_id not in ids:
			return(None, None, None)
		idx = np.where(ids == marker_id)[0][0]
		corners = corners[idx]
		obj_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
		ret, rvecs, tvecs = cv2.solvePnP(obj_points, corners, self.intrinsics_mat[cam], np.zeros(5))
		rot = cv2.Rodrigues(rvecs)[0]
		pose = np.eye(4)
		pose[:3,:3] = rot
		pose[:3,3] = tvecs.flatten()
		return(pose, rvecs, tvecs)
	
	def update_camera_extrinsics_from_marker(self,cam,marker_id,marker_size = 100):
		pose, _, _ = self.compute_marker_pose(cam,marker_id,marker_size)
		if pose is not None:
			self.extrinsics[cam] = invertSE3(pose)
			print("Updated extrinsics for camera {}: \n{}".format(cam,self.extrinsics[cam]))
			return(self.extrinsics[cam])
		else:
			print("Marker not found")
			return(None)
	
	def compute_robot_base_from_marker(self,cam,robot_pose,ee_to_camera,marker_id,marker_size = 100):
		pose, _, _ = self.compute_marker_pose(cam,marker_id,marker_size)
		if pose is not None:
			world_to_camera = invertSE3(pose)
			world_to_robot = world_to_camera @ invertSE3(ee_to_camera) @ invertSE3(robot_pose)
			return(world_to_robot)
		else:
			print("Marker not found")
			return(None)
	
	def get_obj_mask(self, cam, obj_label):
		image = self.get_rgb_image(cam)
		mask = self.OWL.get_obj_mask(image, obj_label)
		return(mask)
	
	def get_obj_in_obj(self,cam, obj_label, parent_obj_label):
		image = self.get_rgb_image(cam)
		bbox = self.OWL.get_obj_box(image, parent_obj_label)
		cropped_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
		cropped_mask = self.OWL.get_obj_mask(cropped_image, obj_label)
		full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
		full_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = cropped_mask
		return(full_mask)
	
	def get_mask_pointcloud(self,mask,cam, depth_max = 1000, depth_min = 200):
		pixels = np.argwhere(mask)
		depth_image = self.depth_images[cam]
		pixels = [pixel for pixel in pixels if depth_image[pixel[0],pixel[1]] < depth_max and depth_image[pixel[0],pixel[1]] > depth_min]
		pixels = np.array(pixels)
		points = batch_coords(self.intrinsics[cam],pixels,self.depth_images[cam])
		colors = self.color_images[cam][pixels[:,0],pixels[:,1]]
		return(points, colors)
	
	def get_obj_pointcloud(self,cam,obj_label, pad_object = False, pad_size = 5, parent_obj_label = None):
		if parent_obj_label is not None:
			mask = self.get_obj_in_obj(cam,obj_label,parent_obj_label)
		else:
			mask = self.get_obj_mask(cam,obj_label)
		if pad_object:
			mask = expand_mask(mask, kernel_size=pad_size)
		if np.all(mask == 0):
			return(np.zeros((0,3)), np.zeros((0,3)))
		points, colors = self.get_mask_pointcloud(mask,cam)
		return(points, colors)

def create_vision_node(display_camera = False,load_owl = False):
	camera_list = ['152122075524','827312070621']
	# camera_list = ['152122075524']
	# camera_list = ['152122075524','242322071433']

	# camera_list = ['746612070227','827312072396']
	# camera_list = ['827312070621','827312072396']
	# camera_list = ['152122075524','827312070621','827312072396']

	vis_node = VisionNode(camera_list,
						  display_camera=display_camera,
						  save_path = "",
						  load_OWL=load_owl) # place the name of the objective here

	print("Starting cameras")
	vis_node.start_cameras(camera_list,exposure=200)
	return(vis_node)

def capture_pc():

	vis_node = create_vision_node()
	_ = input("Press enter to capture")
	pc, colors = vis_node.get_point_cloud(1,depth_max = 800)
	pc /= 1000
	np.save("point_cloud.npy",pc)
	np.save("colors.npy",colors)
	vis_node.stop_cameras()
	sys.exit(0)

def main():
	# L_robot_pose = np.array([321.1,-69.9,346.8,164.6,-15.6,46.5])
	# L_robot_pose[3:] = np.deg2rad(L_robot_pose[3:])
	# L_robot_pose = vec2SE3(L_robot_pose, intrinsic = False)
	# R_robot_pose = np.array([206.3,237.7,392.4,-156.4,-35.4,7.5])
	# R_robot_pose[3:] = np.deg2rad(R_robot_pose[3:])
	# R_robot_pose = vec2SE3(R_robot_pose, intrinsic = False)

	# # L_robot_pose = np.eye(4)
	# # R_robot_pose = np.eye(4)

	# # fp = FramePlot()
	# # fp.plotFrame(np.eye(4), label = "L Robot base")
	# # fp.plotFrame(L_robot_pose, label = "L robot ee")
	# # fp.showPlot()

	# # fp = FramePlot()
	# # fp.plotFrame(np.eye(4), label = "R Robot base")
	# # fp.plotFrame(R_robot_pose, label = "R robot ee")
	# # fp.showPlot()

	# ee_to_camera = np.array([69.54,-38.72,87.53,0.,0.,np.pi/2])
	# ee_to_camera = vec2SE3(ee_to_camera)

	vis_node = create_vision_node(display_camera=True, load_owl=True)
	# # vis_node.query_intrinsics()
	
	time.sleep(3)
	cam = 0
	obj_label = "drawer"
	# # # # fixed_cam = 0
	# # # # corners, ids = vis_node.detect_aruco(fixed_cam)
	# # # # pose = vis_node.update_camera_extrinsics_from_marker(fixed_cam,ids[0])
	# # # # print(vis_node.OWL.inference(vis_node.color_images[cam], [obj_label]))
	# try:
	# 	fig = plt.figure(figsize=(10,10))
	# 	while True:
	_ = input("Press enter to get object mask for {}\n".format(obj_label))
	# mask  = vis_node.get_obj_mask(cam,obj_label)
	# mask = vis_node.get_obj_in_obj(cam,obj_label,"white cutting board")

	images, _ = vis_node.get_camera_images()
	cv2.imwrite("Workspace_Drawer.png",cv2.cvtColor(images[0],cv2.COLOR_BGR2RGB))
	# plt.imshow(images[0])
	# plt.show()
	# mask_im = np.zeros(image.shape,dtype=np.uint8)
	# mask_im[:,:,2] = 255
	# mask_im = cv2.bitwise_and(mask_im,mask_im,mask=mask.astype(np.uint8))
	# alpha = .7
	# disp_im = cv2.addWeighted(image, alpha , mask_im, 1-alpha, 0)
	# plt.imshow(disp_im)
	# plt.show()
	# 		plt.clf()
	# 		plt.imshow(disp_im)
	# 		plt.draw()
	# 		plt.pause(0.01)
	# except KeyboardInterrupt:
	# 	print("Exiting object mask loop")
	# finally:
	# 	vis_node.stop_cameras()
	# 	sys.exit(0)

	# # vis_node.start_recording(0)
	# # time.sleep(3)
	# # buffer, fps = vis_node.stop_recording(0)
	# # print("Average FPS: ",fps)
	# # decode_and_save_recording(buffer,"recordings/", fps=fps)

	# # obj_points = vis_node.get_obj_pointcloud(cam,obj_label)
	# # obj_centroid = np.mean(obj_points,axis = 0)
	# # print("Object centroid: ",obj_centroid)
	
	# # points = vis_node.get_obj_pointcloud(0,obj_label)
	# # print(points.shape)


	# # cams = [0,1,2]
	# # disp_images = []
	# # fp = FramePlot()
	
	# # fp.plotFrame(np.eye(4), label = "World frame")
	# # for cam in cams:
	# # 	corners, ids = vis_node.detect_aruco(cam)
	# # 	print("ids: ",ids)
	# # 	image = vis_node.color_images[cam].copy()
	# # 	cv2.aruco.drawDetectedMarkers(image, corners, ids)
	# # 	pose, rvecs, tvecs = vis_node.compute_marker_pose(cam,ids[0])
	# # 	pose = invertSE3(pose)
	# # 	print("Camera {} pose:\n {}".format(cam,pose))
	# # 	fp.plotFrame(pose, label = "Camera {} pose".format(cam))
	# # 	image = cv2.drawFrameAxes(image, vis_node.intrinsics_mat[cam], np.zeros(5), rvecs, tvecs, 100)
	# # 	disp_images.append(image)
	# # image = np.hstack(disp_images)
	# # # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
	# # # cv2.imshow("Marker poses",image)
	# # # cv2.waitKey(1)


	# # world_to_L_robot = vis_node.compute_robot_base_from_marker(2,L_robot_pose,ee_to_camera,ids[0])
	# # world_to_R_robot = vis_node.compute_robot_base_from_marker(1,R_robot_pose,ee_to_camera,ids[0])
	# # print("World to L robot:\n{}".format(world_to_L_robot))
	# # print("L robot Z vector: {}".format(world_to_L_robot[:3,:3] @ np.array([0,0,1])))
	# # print("World to R robot:\n{}".format(world_to_R_robot))
	# # print("R robot Z vector: {}".format(world_to_R_robot[:3,:3] @ np.array([0,0,1])))
	# # fp.plotFrame(world_to_L_robot, label = "L robot base")
	# # fp.plotFrame(world_to_R_robot, label = "R robot base")

	# # fp.showPlot()
	
	

	_ = input("Press enter to stop\n")
	vis_node.stop_cameras()
	sys.exit(0)



if __name__ == '__main__':
	# import cProfile
	# cProfile.run('main()')
	main()