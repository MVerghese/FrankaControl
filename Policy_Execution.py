import numpy as np
import torch
import time
import collections
import cv2
from scipy.spatial.transform import Rotation as R, Slerp

import sys
sys.path.append("/home/mverghese/ego_env/Franka_Kitchen_Env")
import BehaviorCloning
sys.path.remove("/home/mverghese/ego_env/Franka_Kitchen_Env")

import Franka_Gym_Environment
from utils import invertSE3, FramePlot

LOOK_AT_MARKER_POSE = np.array([ 0.52237135, -0.1262271,   0.38110489,  0.82422328, -0.52577913,  0.19288467, -0.08371254,  0.0796622 ])


def _interpolated_move(env, target_pos, target_quat, num_steps=20,
					   step_delay=0.05, gripper_width=0.08):
	"""Move to a target pose via interpolated waypoints to avoid jerky motion.

	Reads the current EE state, linearly interpolates position and uses
	SLERP for orientation, then sends each intermediate pose as an absolute
	command with a short delay between steps.

	Args:
		env: FrankaGymEnvironment with an active impedance controller.
		target_pos: Desired 3D position (meters).
		target_quat: Desired orientation quaternion (xyzw).
		num_steps: Number of interpolation waypoints.
		step_delay: Seconds to wait between waypoints.
		gripper_width: Gripper width to hold throughout the motion.
	"""
	env.client_controller.null_command()
	cur_pos, cur_quat, _, _, _ = env.client_controller.get_state()
	cur_pos = np.array(cur_pos)
	cur_quat = np.array(cur_quat)

	key_rots = R.from_quat(np.stack([cur_quat, target_quat]))
	slerp = Slerp([0.0, 1.0], key_rots)
	target_pos = np.asarray(target_pos, dtype=np.float64)

	for step in range(1, num_steps + 1):
		alpha = step / num_steps
		interp_pos = cur_pos + alpha * (target_pos - cur_pos)
		interp_quat = slerp(alpha).as_quat()

		command = torch.tensor(
			list(interp_pos) + list(interp_quat) + [gripper_width],
			dtype=torch.float64,
		)
		env.client_controller.move_to_absolute_pose(command)
		env.client_controller.get_state()
		time.sleep(step_delay)


def _move_to_absolute_and_settle(env, pos, quat, settle_time, gripper_width=0.08):
	"""Move to a target pose with interpolated waypoints, then wait for settling."""
	_interpolated_move(env, pos, quat, gripper_width=gripper_width)

	time.sleep(settle_time)

	env.client_controller.null_command()
	settled_pos, settled_quat, _, _, _ = env.client_controller.get_state()
	return np.array(settled_pos), np.array(settled_quat)


def _generate_calibration_offsets():
	"""Generate diverse EE offsets (translation in meters, rotation in radians)
	for hand-eye calibration. Returns list of (pos_offset, euler_offset) tuples."""
	offsets = []
	d = 0.04
	a = np.radians(15)

	for axis in range(3):
		for sign in [1, -1]:
			trans = np.zeros(3)
			trans[axis] = sign * d
			offsets.append((trans, np.zeros(3)))

	for axis in range(3):
		for sign in [1, -1]:
			rot = np.zeros(3)
			rot[axis] = sign * a
			offsets.append((np.zeros(3), rot))

	offsets.append((np.array([d/2, d/2, 0]), np.array([0, 0, a/2])))
	offsets.append((np.array([-d/2, 0, d/2]), np.array([a/2, 0, 0])))
	offsets.append((np.array([0, -d/2, -d/2]), np.array([0, a/2, -a/2])))

	return offsets


def identify_robot_base_and_ee_to_wrist_transform(
	look_at_marker_pose,
	marker_id=0,
	marker_size=100,
	wrist_camera_idx=1,
	settle_time=1.5,
	position_unit_scale=1000.0,
):
	"""Calibrate the EE-to-wrist-camera transform and robot base pose using
	an ArUco marker fixed on the table.

	Moves the robot to a known pose where the wrist camera can see the marker,
	then moves through diverse offsets around that pose, detecting the marker
	at each one, and solves the AX=XB hand-eye calibration.

	Args:
		look_at_marker_pose: An 8D array [x, y, z, qx, qy, qz, qw, gripper_width]
			giving a known EE pose from which the wrist camera can see the
			ArUco marker. Obtain this by manually teleoperating the robot.
		marker_id: ArUco marker ID to detect.
		marker_size: Physical marker side length. Units here determine the
			units of the output transforms (default 100 = mm).
		wrist_camera_idx: Camera index for the wrist camera in the VisionNode.
		settle_time: Seconds to wait after each movement for the robot to settle.
		position_unit_scale: Factor to convert robot EE position units to
			marker_size units (default 1000.0 converts meters to mm).

	Returns:
		ee_T_cam: 4x4 SE3 matrix transforming points from camera frame to EE frame.
		world_T_base: 4x4 SE3 matrix giving the robot base pose in the
			marker/world frame.
	"""
	look_at_marker_pose = np.asarray(look_at_marker_pose, dtype=np.float64)
	viewing_pos = look_at_marker_pose[:3]
	viewing_quat = look_at_marker_pose[3:7]
	gripper_width = look_at_marker_pose[7]
	viewing_rot = R.from_quat(viewing_quat)

	env = Franka_Gym_Environment.FrankaGymEnvironment(
		load_vision_node=True,
		use_gui=False,
		episode_timeout=10000,
		reset_wait=0,
		open_gripper_on_reset=True,
	)

	calibration_offsets = _generate_calibration_offsets()

	R_gripper2base = []
	t_gripper2base = []
	R_target2cam = []
	t_target2cam = []

	try:
		obs, _ = env.reset()

		print(f"Moving to viewing pose: pos={viewing_pos}")
		_move_to_absolute_and_settle(
			env, viewing_pos, viewing_quat, settle_time,
			gripper_width=gripper_width,
		)

		for i, (pos_offset, euler_offset) in enumerate(calibration_offsets):
			target_pos = viewing_pos + pos_offset
			offset_rot = R.from_euler('xyz', euler_offset)
			target_quat = (viewing_rot * offset_rot).as_quat()

			ee_pos, ee_quat = _move_to_absolute_and_settle(
				env, target_pos, target_quat, settle_time,
				gripper_width=gripper_width,
			)

			try:
				marker_pose, _, _ = env.vision_node.compute_marker_pose(
					wrist_camera_idx, marker_id, marker_size
				)
			except TypeError:
				marker_pose = None

			if marker_pose is None:
				print(f"Pose {i+1}/{len(calibration_offsets)}: marker not detected, skipping")
				_move_to_absolute_and_settle(
					env, viewing_pos, viewing_quat, settle_time,
					gripper_width=gripper_width,
				)
				continue

			print(f"Pose {i+1}/{len(calibration_offsets)}: marker detected, "
				  f"EE pos={ee_pos}")

			R_ee = R.from_quat(ee_quat).as_matrix()
			t_ee = (ee_pos * position_unit_scale).reshape(3, 1)

			R_gripper2base.append(R_ee)
			t_gripper2base.append(t_ee)
			R_target2cam.append(marker_pose[:3, :3].copy())
			t_target2cam.append(marker_pose[:3, 3].reshape(3, 1).copy())

			_move_to_absolute_and_settle(
				env, viewing_pos, viewing_quat, settle_time,
				gripper_width=gripper_width,
			)

	finally:
		env.close()

	n_valid = len(R_gripper2base)
	if n_valid < 3:
		raise RuntimeError(
			f"Only {n_valid} valid poses collected, need at least 3 for calibration"
		)

	print(f"Solving hand-eye calibration with {n_valid} pose pairs")

	R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
		R_gripper2base, t_gripper2base,
		R_target2cam, t_target2cam,
		method=cv2.CALIB_HAND_EYE_TSAI,
	)

	ee_T_cam = np.eye(4)
	ee_T_cam[:3, :3] = R_cam2gripper
	ee_T_cam[:3, 3] = t_cam2gripper.flatten()

	world_T_base_estimates = []
	for i in range(n_valid):
		base_T_ee_i = np.eye(4)
		base_T_ee_i[:3, :3] = R_gripper2base[i]
		base_T_ee_i[:3, 3] = t_gripper2base[i].flatten()

		cam_T_marker_i = np.eye(4)
		cam_T_marker_i[:3, :3] = R_target2cam[i]
		cam_T_marker_i[:3, 3] = t_target2cam[i].flatten()

		world_T_base_i = (
			invertSE3(cam_T_marker_i) @ invertSE3(ee_T_cam) @ invertSE3(base_T_ee_i)
		)
		world_T_base_estimates.append(world_T_base_i)

	avg_t = np.mean([T[:3, 3] for T in world_T_base_estimates], axis=0)
	avg_R = np.mean([T[:3, :3] for T in world_T_base_estimates], axis=0)
	U, _, Vt = np.linalg.svd(avg_R)
	det = np.linalg.det(U @ Vt)
	avg_R = U @ np.diag([1, 1, det]) @ Vt

	world_T_base = np.eye(4)
	world_T_base[:3, :3] = avg_R
	world_T_base[:3, 3] = avg_t

	print(f"\nEE-to-wrist-camera transform (ee_T_cam):\n{ee_T_cam}")
	print(f"\nRobot base in world/marker frame (world_T_base):\n{world_T_base}")

	return ee_T_cam, world_T_base


def visualize_calibration(ee_T_cam, world_T_base, position_unit_scale=1000):
	"""Plot the marker, robot base, EE, and wrist camera frames to verify calibration.

	All frames are plotted in the world/marker coordinate system (mm).

	Args:
		ee_T_cam: 4x4 SE3 transform from EE to wrist camera (mm).
		world_T_base: 4x4 SE3 transform from robot base to world/marker frame (mm).
		position_unit_scale: Multiplier to convert robot EE position (meters)
			to the calibration unit system (default 1000 for mm).
	"""
	env = Franka_Gym_Environment.FrankaGymEnvironment(
		load_vision_node=False, use_gui=False, episode_timeout=200,
		reset_wait=0,
	)
	try:
		env.reset()
		env.client_controller.null_command()
		ee_pos, ee_quat, _, _, _ = env.client_controller.get_state()

		base_T_ee = np.eye(4)
		base_T_ee[:3, :3] = R.from_quat(ee_quat).as_matrix()
		base_T_ee[:3, 3] = np.array(ee_pos) * position_unit_scale

		world_T_ee = world_T_base @ base_T_ee
		world_T_cam = world_T_ee @ ee_T_cam

		fp = FramePlot()
		fp.plotFrame(np.eye(4), label="Marker (origin)")
		fp.plotFrame(world_T_base, label="Robot base")
		fp.plotFrame(world_T_ee, label="EE")
		fp.plotFrame(world_T_cam, label="Wrist camera")
		fp.showPlot()
	finally:
		env.close()


class ObjectGraspPolicy:
	def __init__(self, ee_T_cam, world_T_base, vision_node,):
		self.ee_T_cam = ee_T_cam
		self.world_T_base = world_T_base
		self.vision_node = vision_node

	def update_world_camera_extrinsics(self, marker_id=0, marker_size=100):
		self.vision_node.update_camera_extrinsics_from_marker(0, marker_id, marker_size)

	def get_object_centroid_in_base_frame(self, object_label, world_camera_idx=0):
		"""Detect an object with the world camera and return its centroid
		in the robot base frame (meters).

		Args:
			object_label: Text label for the OWL detector.
			world_camera_idx: Camera index for the world camera.

		Returns:
			centroid_base_m: 3D centroid position in robot base frame (meters).
		"""
		points, _ = self.vision_node.get_obj_pointcloud(world_camera_idx, object_label)
		if points.shape[0] == 0:
			raise RuntimeError(f"No points detected for object '{object_label}'")

		centroid_cam = np.mean(points, axis=0)

		centroid_cam_h = np.append(centroid_cam, 1.0)
		world_T_cam = self.vision_node.extrinsics[world_camera_idx]
		centroid_world = (world_T_cam @ centroid_cam_h)[:3]

		base_T_world = invertSE3(self.world_T_base)
		centroid_base = (base_T_world @ np.append(centroid_world, 1.0))[:3]

		centroid_base_m = centroid_base / 1000.0
		return centroid_base_m

	def pregrasp_position(self, object_label, height=0.35, world_camera_idx=0,
						  ee_quat=None):
		"""Compute an EE position that centers the wrist camera above the object.

		Updates the world camera extrinsics from the ArUco marker, detects the
		object, computes its centroid in the robot base frame, and offsets the
		EE position so the wrist camera (not the EE origin) is centered above
		the object.

		Args:
			object_label: Text label for the OWL detector.
			height: Height above the table to position the EE (meters).
			world_camera_idx: Camera index for the world camera.
			ee_quat: Expected EE orientation quaternion (xyzw) at the pregrasp
				pose. Used to compute the camera offset in the base frame.
				If None, the camera offset is not applied.

		Returns:
			pregrasp_pos: 3D EE position [x, y, z] in robot base frame (meters).
		"""
		self.update_world_camera_extrinsics()

		centroid = self.get_object_centroid_in_base_frame(object_label, world_camera_idx)
		pregrasp_pos = np.array([centroid[0], centroid[1], height])

		if ee_quat is not None:
			R_base_ee = R.from_quat(ee_quat).as_matrix()
			cam_offset_base = R_base_ee @ self.ee_T_cam[:3, 3] / 1000.0
			pregrasp_pos[0] -= cam_offset_base[0]
			pregrasp_pos[1] -= cam_offset_base[1]

		print(f"Object '{object_label}' centroid in base frame: {centroid}")
		print(f"Pregrasp position (camera-centered): {pregrasp_pos}")
		return pregrasp_pos

	def visual_servo(self, object_label, env, wrist_camera_idx=1,
					 p_gain=0.3, tolerance_px=30.0, relaxed_tolerance_px=40.0,
					 max_iterations=20):
		"""Center the object under the wrist camera using proportional control.

		Uses get_obj_mask to locate the object in the image, computes the pixel
		offset from image center, converts to a 3D displacement via depth and
		intrinsics, and iteratively corrects the EE position.

		Args:
			object_label: Text label for the OWL detector.
			env: FrankaGymEnvironment instance.
			wrist_camera_idx: Camera index for the wrist camera.
			p_gain: Proportional gain for the controller.
			tolerance_px: Pixel tolerance for convergence.
			relaxed_tolerance_px: If the error is within this tolerance after
				max_iterations, proceed anyway with a warning.
			max_iterations: Maximum servo iterations.

		Returns:
			True if converged or within relaxed tolerance, False otherwise.
		"""
		env.client_controller.null_command()
		pos, quat, _, _, _ = env.client_controller.get_state()
		current_pos = np.array(pos)
		current_quat = np.array(quat)
		for iteration in range(max_iterations):
			mask = self.vision_node.get_obj_mask(wrist_camera_idx, object_label)
			if np.all(mask == 0):
				print("Object not detected in wrist camera")
				return False

			ys, xs = np.where(mask > 0)
			centroid_px = np.array([np.mean(xs), np.mean(ys)])

			h, w = mask.shape[:2]
			center_px = np.array([w / 2.0, h / 2.0])
			offset_px = centroid_px - center_px

			offset_mag = np.linalg.norm(offset_px)
			print(f"Servo iteration {iteration+1}: pixel offset = {offset_px}, "
				  f"magnitude = {offset_mag:.1f}")

			if offset_mag < tolerance_px:
				print("Object centered in wrist camera")
				return True

			depth_image = self.vision_node.depth_images[wrist_camera_idx]
			cy, cx = int(centroid_px[1]), int(centroid_px[0])
			cy = np.clip(cy, 0, h - 1)
			cx = np.clip(cx, 0, w - 1)
			r = 10
			region = depth_image[max(0, cy-r):min(h, cy+r),
								 max(0, cx-r):min(w, cx+r)]
			nonzero = region[region > 0]
			if len(nonzero) == 0:
				print("No valid depth at object centroid")
				return False
			depth = float(np.mean(nonzero))

			intr = self.vision_node.intrinsics[wrist_camera_idx]
			ppx, ppy, fx, fy = intr[0], intr[1], intr[2], intr[3]
			dx_cam = offset_px[0] * depth / fx
			dy_cam = offset_px[1] * depth / fy
			delta_cam = np.array([dx_cam, dy_cam, 0.0])

			env.client_controller.null_command()
			pos, quat, _, _, _ = env.client_controller.get_state()
			current_pos = np.array(pos)
			

			R_base_ee = R.from_quat(current_quat).as_matrix()
			R_base_cam = R_base_ee @ self.ee_T_cam[:3, :3]
			delta_base_m = R_base_cam @ delta_cam / 1000.0 * p_gain

			new_pos = current_pos + delta_base_m
			command = torch.tensor(
				list(new_pos) + list(current_quat) + [0.08],
				dtype=torch.float64,
			)
			env.client_controller.move_to_absolute_pose(command)
			env.client_controller.get_state()
			time.sleep(0.1)

		if offset_mag < relaxed_tolerance_px:
			print(f"Visual servo did not fully converge after {max_iterations} "
				  f"iterations (error {offset_mag:.1f}px), but within relaxed "
				  f"tolerance ({relaxed_tolerance_px}px) — proceeding")
			return True
		print(f"Visual servo did not converge after {max_iterations} iterations "
			  f"(error {offset_mag:.1f}px > relaxed tolerance {relaxed_tolerance_px}px)")
		return False

	def align_gripper(self, object_label, env, wrist_camera_idx=1):
		"""Align the gripper center and orientation to the object for grasping.

		Detects the object with get_obj_mask, runs PCA on the mask to find the
		minor axis, then:
		  1. Moves the gripper center (EE origin) directly above the object
		     centroid, accounting for the ee_T_cam offset.
		  2. Rotates the gripper around its z-axis so the fingers align with
		     the object's minor axis (thinnest dimension).

		Args:
			object_label: Text label for the OWL detector.
			env: FrankaGymEnvironment instance.
			wrist_camera_idx: Camera index for the wrist camera.

		Returns:
			True if alignment succeeded, False if the object was not detected.
		"""
		mask = self.vision_node.get_obj_mask(wrist_camera_idx, object_label)
		if np.all(mask == 0):
			print("Object not detected in wrist camera")
			return False

		ys, xs = np.where(mask > 0)
		points_2d = np.column_stack([xs, ys]).astype(np.float64)
		mean_px = np.mean(points_2d, axis=0)
		centered = points_2d - mean_px
		cov = np.cov(centered.T)
		eigenvalues, eigenvectors = np.linalg.eigh(cov)
		sort_indices = np.argsort(eigenvalues)[::-1]
		sorted_eigenvectors = eigenvectors[:, sort_indices]
		minor_axis_px = sorted_eigenvectors[:, 1]

		minor_axis_cam = np.array([minor_axis_px[0], minor_axis_px[1], 0.0])
		minor_axis_ee = self.ee_T_cam[:3, :3] @ minor_axis_cam
		minor_axis_ee_xy = minor_axis_ee[:2]
		minor_axis_ee_xy = minor_axis_ee_xy / np.linalg.norm(minor_axis_ee_xy)

		yaw_angle = np.arctan2(-minor_axis_ee_xy[0], minor_axis_ee_xy[1])
		if yaw_angle > np.pi / 2:
			yaw_angle -= np.pi
		elif yaw_angle < -np.pi / 2:
			yaw_angle += np.pi

		yaw_angle += np.pi/4

		h, w = mask.shape[:2]
		depth_image = self.vision_node.depth_images[wrist_camera_idx]
		cy, cx = int(mean_px[1]), int(mean_px[0])
		cy = np.clip(cy, 0, h - 1)
		cx = np.clip(cx, 0, w - 1)
		r = 10
		region = depth_image[max(0, cy-r):min(h, cy+r),
							 max(0, cx-r):min(w, cx+r)]
		nonzero = region[region > 0]
		if len(nonzero) == 0:
			print("No valid depth for gripper alignment")
			return False
		depth = float(np.mean(nonzero))

		intr = self.vision_node.intrinsics[wrist_camera_idx]
		ppx, ppy, fx, fy = intr[0], intr[1], intr[2], intr[3]
		obj_x_cam = (mean_px[0] - ppx) * depth / fx
		obj_y_cam = (mean_px[1] - ppy) * depth / fy
		centroid_cam_h = np.array([obj_x_cam, obj_y_cam, depth, 1.0])

		centroid_ee = (self.ee_T_cam @ centroid_cam_h)[:3]
		delta_ee = np.array([centroid_ee[0], centroid_ee[1], 0.0])

		env.client_controller.null_command()
		pos, quat, _, _, _ = env.client_controller.get_state()
		current_pos = np.array(pos)
		current_rot = R.from_quat(np.array(quat))


		delta_base_m = current_rot.as_matrix() @ delta_ee / 1000.0
		new_pos = current_pos + delta_base_m

		yaw_rot = R.from_euler('z', yaw_angle)
		new_rot = current_rot * yaw_rot
		new_quat = new_rot.as_quat()

		print(f"Align gripper: yaw={np.degrees(yaw_angle):.1f}deg, "
			  f"xy_offset_ee=[{centroid_ee[0]:.1f}, {centroid_ee[1]:.1f}]mm")

		_move_to_absolute_and_settle(env, new_pos, new_quat, settle_time=1.0)
		return True

	def grasp_object(self, object_label, env, pregrasp_height=0.35,
					 grasp_offset=0.005,
					 wrist_camera_idx=1, settle_time=1.0):
		"""Full pick pipeline: pregrasp, visual servo, align, lower, grasp, lift.

		Args:
			object_label: Text label for the OWL detector.
			env: FrankaGymEnvironment instance.
			pregrasp_height: Height above table for the pregrasp pose (meters).
			grasp_offset: Small clearance above the object surface when
				closing the gripper (meters).
			wrist_camera_idx: Camera index for the wrist camera.
			settle_time: Seconds to wait after movements.

		Returns:
			True if the grasp sequence completed, False if detection failed
			or an error occurred.
		"""
		try:
			env.reset()

			env.client_controller.open_gripper()
			env.client_controller.get_state()

			env.client_controller.null_command()
			init_pos, init_quat, _, _, _ = env.client_controller.get_state()
			down_quat = np.array(init_quat)

			pregrasp_pos = self.pregrasp_position(
				object_label, height=pregrasp_height, ee_quat=down_quat,
			)

			print("Moving to pregrasp position")
			_move_to_absolute_and_settle(
				env, pregrasp_pos, down_quat, settle_time,
			)

			print("Running visual servo")
			servo_ok = self.visual_servo(
				object_label, env, wrist_camera_idx=wrist_camera_idx,
			)
			if not servo_ok:
				print("Visual servo failed, aborting grasp")
				return False

			print("Aligning gripper")
			align_ok = self.align_gripper(
				object_label, env, wrist_camera_idx=wrist_camera_idx,
			)
			if not align_ok:
				print("Gripper alignment failed, aborting grasp")
				return False

			mask = self.vision_node.get_obj_mask(wrist_camera_idx, object_label)
			if np.all(mask == 0):
				print("Lost sight of object after alignment")
				return False

			ys, xs = np.where(mask > 0)
			centroid_px = np.array([np.mean(xs), np.mean(ys)])
			h, w = mask.shape[:2]
			depth_image = self.vision_node.depth_images[wrist_camera_idx]
			cy, cx = int(centroid_px[1]), int(centroid_px[0])
			cy = np.clip(cy, 0, h - 1)
			cx = np.clip(cx, 0, w - 1)
			r = 10
			region = depth_image[max(0, cy-r):min(h, cy+r),
								 max(0, cx-r):min(w, cx+r)]
			nonzero = region[region > 0]

			TABLE_HEIGHT = 0.19
			env.client_controller.null_command()
			pos, quat, _, _, _ = env.client_controller.get_state()
			grasp_pos = np.array(pos)
			grasp_quat = np.array(quat)

			if len(nonzero) == 0:
				print("No valid depth for grasp height, using table height instead")
				grasp_pos[2] = TABLE_HEIGHT
			else:
				obj_depth_mm = float(np.mean(nonzero))
				intr = self.vision_node.intrinsics[wrist_camera_idx]
				ppx, ppy, fx, fy = intr[0], intr[1], intr[2], intr[3]
				obj_x_cam = (centroid_px[0] - ppx) * obj_depth_mm / fx
				obj_y_cam = (centroid_px[1] - ppy) * obj_depth_mm / fy
				obj_cam_h = np.array([obj_x_cam, obj_y_cam, obj_depth_mm, 1.0])
				obj_ee = (self.ee_T_cam @ obj_cam_h)[:3]
				obj_z_below_ee_m = obj_ee[2] / 1000.0
				grasp_pos[2] += -obj_z_below_ee_m + grasp_offset
				grasp_pos[2] = max(grasp_pos[2], TABLE_HEIGHT)

			print(f"Lowering to grasp: z={grasp_pos[2]:.4f}m")
			_move_to_absolute_and_settle(
				env, grasp_pos, grasp_quat, settle_time,
			)

			print("Closing gripper")
			env.client_controller.close_gripper()
			env.client_controller.get_state()
			time.sleep(0.5)

			prev_width = None
			stable_count = 0
			while stable_count < 5:
				time.sleep(0.1)
				env.client_controller.null_command()
				_, _, _, _, gripper_width = env.client_controller.get_state()
				if prev_width is not None and abs(gripper_width - prev_width) < 1e-4:
					stable_count += 1
				else:
					stable_count = 0
				prev_width = gripper_width
			print(f"Gripper settled at width {prev_width:.4f}")

			lift_pos = grasp_pos.copy()
			lift_pos[2] = pregrasp_height
			print(f"Lifting to z={lift_pos[2]:.4f}m")
			_move_to_absolute_and_settle(
				env, lift_pos, grasp_quat, settle_time, gripper_width=0.00,
			)

			


			print(f"Grasp of '{object_label}' complete")
			return True

		except Exception as e:
			import traceback
			print(f"Error during grasp_object: {e}")
			traceback.print_exc()
			return False




class PolicyExecution:
	VRAM_USAGE_THRESHOLD = 0.85

	def __init__(self, action_sources: dict, device="cuda:0",
				 obs_dim=8, action_dim=8, num_diffusion_iters=50,
				 episode_timeout=200, reset_wait=0, use_gui=True,
				 wait_for_gui_reset=False, open_gripper_on_reset=True,
				 load_owl=True, ee_T_cam=None, world_T_base=None):
		"""
		Args:
			action_sources: Dict mapping natural language tags to file paths.
				.pth files are treated as diffusion policy checkpoints.
				.npz files are treated as recorded trajectories to replay.
				Tags starting with "pickup" trigger the ObjectGraspPolicy
				and do not need an entry in action_sources.
				e.g. {"Open Drawer": "/path/to/dp_model_final.pth",
				       "Place Object": "/path/to/demo_0.npz"}
			device: CUDA device string.
			obs_dim: Observation dimensionality for the diffusion policy (pose only).
			action_dim: Action dimensionality.
			num_diffusion_iters: Number of DDPM denoising steps at inference time.
			episode_timeout: Max steps per episode for the environment.
			reset_wait: Seconds to wait after environment reset.
			use_gui: Whether to launch the tkinter GUI for the environment.
			wait_for_gui_reset: Whether to wait for a GUI button press on reset.
			open_gripper_on_reset: Whether to open gripper during environment reset.
			load_owl: Whether to load the OWL detector in the vision node.
			ee_T_cam: 4x4 SE3 EE-to-wrist-camera transform (mm). Required
				for pickup commands.
			world_T_base: 4x4 SE3 world/marker-to-robot-base transform (mm).
				Required for pickup commands.
		"""
		self.action_sources = action_sources
		self.device = device
		self.obs_dim = obs_dim
		self.action_dim = action_dim
		self.num_diffusion_iters = num_diffusion_iters

		self.env = Franka_Gym_Environment.FrankaGymEnvironment(
			load_vision_node=True,
			use_gui=use_gui,
			episode_timeout=episode_timeout,
			reset_wait=reset_wait,
			wait_for_gui_reset=wait_for_gui_reset,
			open_gripper_on_reset=open_gripper_on_reset,
			load_owl=load_owl,
		)

		self._loaded_policies = {}
		self._access_order = []

		self._grasp_policy = None
		self._ee_T_cam = ee_T_cam
		self._world_T_base = world_T_base
		self._holding_object = False

		self.env.reset()
		self.env.client_controller.null_command()
		self.init_pos, self.init_quat, _, _, _ = self.env.client_controller.get_state()

	def _manual_reset(self):
		self.env.client_controller.null_command()
		_, _, _, _, gripper_width = self.env.client_controller.get_state()
		_move_to_absolute_and_settle(
				self.env, self.init_pos, self.init_quat, 1.0, gripper_width=gripper_width,
			)
		self.env.client_controller.null_command()
		obs = self.env.unwrapped._get_obs()
		return obs

	def _get_vram_usage_fraction(self) -> float:
		if not torch.cuda.is_available():
			return 0.0
		device_idx = torch.device(self.device).index or 0
		allocated = torch.cuda.memory_allocated(device_idx)
		total = torch.cuda.get_device_properties(device_idx).total_mem
		return allocated / total

	def _evict_least_recent(self):
		"""Remove the least recently used policy from GPU to free VRAM."""
		while (self._access_order
			   and self._get_vram_usage_fraction() > self.VRAM_USAGE_THRESHOLD):
			tag_to_evict = self._access_order.pop(0)
			if tag_to_evict in self._loaded_policies:
				print(f"Evicting policy '{tag_to_evict}' from GPU to free VRAM")
				del self._loaded_policies[tag_to_evict]
				torch.cuda.empty_cache()

	def _mark_accessed(self, tag: str):
		if tag in self._access_order:
			self._access_order.remove(tag)
		self._access_order.append(tag)

	def _load_policy(self, tag: str) -> BehaviorCloning.DP_Inference:
		if tag in self._loaded_policies:
			self._mark_accessed(tag)
			return self._loaded_policies[tag]

		self._evict_least_recent()

		checkpoint_path = self.action_sources[tag]
		print(f"Loading policy '{tag}' from {checkpoint_path}")
		inference = BehaviorCloning.DP_Inference(
			model_path=checkpoint_path,
			obs_dim=self.obs_dim,
			action_dim=self.action_dim,
			device=self.device,
			num_diffusion_iters=self.num_diffusion_iters,
			is_transformer=False,
			vision_model=True,
		)
		self._loaded_policies[tag] = inference
		self._mark_accessed(tag)
		return inference

	def _load_trajectory(self, tag: str) -> np.ndarray:
		path = self.action_sources[tag]
		print(f"Loading trajectory '{tag}' from {path}")
		data = np.load(path, allow_pickle=True)
		actions = data["actions"]
		return actions

	def _run_policy(self, tag: str, num_steps: int, reset_env: bool = True):
		policy = self._load_policy(tag)

		if reset_env:
			if self._holding_object:
				obs = self._manual_reset()
			else:
				obs, _ = self.env.reset()
		else:
			self.env.client_controller.null_command()
			obs = self.env.unwrapped._get_obs()

		policy.reset(obs)

		last_obs = obs
		for step in range(num_steps):
			action = policy.act(last_obs)
			last_obs, reward, terminated, truncated, info = self.env.step(action)
			# print(f"[{tag}] Step {step + 1}/{num_steps}  "
			# 	  f"reward={reward:.3f}  terminated={terminated}  truncated={truncated}")
			if terminated or truncated:
				print(f"[{tag}] Episode ended early at step {step + 1}")
				break

		return True

	def _run_trajectory(self, tag: str, reset_env: bool = True):
		actions = self._load_trajectory(tag)
		num_steps = len(actions)

		if reset_env:
			print(f"[{tag}] Resetting environment")
			print(f"[{tag}] Holding object: {self._holding_object}")
			if self._holding_object:
				obs = self._manual_reset()
			else:
				obs, _ = self.env.reset()
		else:
			self.env.client_controller.null_command()
			obs = self.env.unwrapped._get_obs()

		last_obs = obs
		for step, action in enumerate(actions):
			last_obs, reward, terminated, truncated, info = self.env.step(action)
			# print(f"[{tag}] Step {step + 1}/{num_steps}  "
			# 	  f"reward={reward:.3f}  terminated={terminated}  truncated={truncated}")
			if terminated or truncated:
				print(f"[{tag}] Trajectory replay ended early at step {step + 1}")
				break

		return True

	def _find_putobject_source(self, receptacle: str) -> str:
		"""Find an action source key matching 'PutObject [object] <receptacle>'.

		Matches case-insensitively on the receptacle (last word) of any
		action source whose key starts with 'PutObject'.
		"""
		receptacle_lower = receptacle.lower()
		for key in self.action_sources:
			if key.lower().startswith("putobject"):
				key_parts = key.split()
				if len(key_parts) >= 3 and key_parts[-1].lower() == receptacle_lower:
					return key
		return None

	def _get_grasp_policy(self):
		if self._grasp_policy is None:
			if self._ee_T_cam is None or self._world_T_base is None:
				raise RuntimeError(
					"ee_T_cam and world_T_base must be provided to "
					"PolicyExecution for pickup commands"
				)
			self._grasp_policy = ObjectGraspPolicy(
				self._ee_T_cam, self._world_T_base, self.env.vision_node,
			)
		return self._grasp_policy

	def run(self, tag: str, num_steps: int = 200, reset_env: bool = True):
		"""
		Run the named action source on the robot.

		For tags starting with "pickup", uses ObjectGraspPolicy to grasp the
		object named by the rest of the tag (e.g. "pickup purple grapes").
		For .pth sources, runs the diffusion policy for num_steps.
		For .npz sources, replays the full recorded trajectory (num_steps is ignored).

		Args:
			tag: The natural language tag identifying which source to run.
			num_steps: Number of environment steps (only used for policy checkpoints).
			reset_env: Whether to reset the environment before running.

		Returns:
			The last observation from the environment, or True/False for pickup.
		"""
		if tag.lower().startswith("pickupobject"):
			object_label = tag[len("pickupobject"):].strip()
			if not object_label:
				raise ValueError(
					f"Pickup tag '{tag}' must include an object label, "
					f"e.g. 'PickupObject purple grapes'"
				)
			grasp_policy = self._get_grasp_policy()
			print(f"[{tag}] Grasping object '{object_label}'")
			success = grasp_policy.grasp_object(object_label, self.env)
			if success:
				print(f"[{tag}] Grasping object '{object_label}' successful")
				self._holding_object = True
			self.env.client_controller.null_command()
			obs = self.env.unwrapped._get_obs()
			return obs, success

		is_put = False
		if tag.lower().startswith("putobject"):
			print(f"[{tag}] Putting object into receptacle")
			is_put = True
			parts = tag.split()
			if len(parts) < 3:
				raise ValueError(
					f"PutObject tag '{tag}' must have the format "
					f"'PutObject <object> <receptacle>', "
					f"e.g. 'PutObject Banana Drawer'"
				)
			receptacle = parts[-1]
			source_key = self._find_putobject_source(receptacle)
			if source_key is None:
				raise ValueError(
					f"No action source found for receptacle '{receptacle}'. "
					f"Available PutObject sources: "
					f"{[k for k in self.action_sources if k.lower().startswith('putobject')]}"
				)
			tag = source_key

		if tag not in self.action_sources:
			raise ValueError(f"Unknown tag '{tag}'. "
							 f"Available: {list(self.action_sources.keys())}")

		path = self.action_sources[tag]
		if path.endswith(".npz"):
			print(f"[{tag}] Running trajectory")
			result = self._run_trajectory(tag, reset_env=reset_env)
		else:
			result = self._run_policy(tag, num_steps, reset_env=reset_env)
		
		if is_put:
			self._holding_object = False
		self.env.client_controller.null_command()
		obs = self.env.unwrapped._get_obs()
		return obs, result

	def close(self):
		self._loaded_policies.clear()
		torch.cuda.empty_cache()
		self.env.close()

def get_policy_execution(task_name: str, device: str = "cuda:0"):
	if task_name == "Put Away Fruit in the Drawer":
		action_sources = {
			"OpenObject Drawer": "/home/mverghese/franka_control/Demos/Open_Drawer/demo_1.npz",
			"CloseObject Drawer": "/home/mverghese/franka_control/Demos/Close_Drawer/demo_0.npz",
			"PutObject [object] Drawer": "/home/mverghese/franka_control/Demos/Put_Banana_Drawer/demo_4.npz"
		}
		open_gripper_on_reset = False
	else:
		raise ValueError(f"Unknown task name: {task_name}")
	ee_T_cam, world_T_base = np.load("/home/mverghese/franka_control/robot_base_and_ee_to_wrist_transform.npz")["ee_T_cam"], np.load("/home/mverghese/franka_control/robot_base_and_ee_to_wrist_transform.npz")["world_T_base"]
	executor = PolicyExecution(action_sources, device=device, open_gripper_on_reset=open_gripper_on_reset, ee_T_cam=ee_T_cam, world_T_base=world_T_base)
	return executor

if __name__ == "__main__":
	action_sources = {
		"OpenObject Drawer": "/home/mverghese/franka_control/Demos/Open_Drawer/demo_1.npz",
		"CloseObject Drawer": "/home/mverghese/franka_control/Demos/Close_Drawer/demo_0.npz",
		"PutObject [object] Drawer": "/home/mverghese/franka_control/Demos/Put_Banana_Drawer/demo_4.npz"
	}
	ee_T_cam, world_T_base = np.load("robot_base_and_ee_to_wrist_transform.npz")["ee_T_cam"], np.load("robot_base_and_ee_to_wrist_transform.npz")["world_T_base"]
	executor = PolicyExecution(action_sources, device="cuda:0", open_gripper_on_reset=False, ee_T_cam=ee_T_cam, world_T_base=world_T_base)
	try:
		last_obs, success = executor.run("OpenObject Drawer", num_steps=200)
		last_obs, success = executor.run("PickupObject lime", num_steps=200)
		last_obs, success = executor.run("PutObject lime Drawer", num_steps=200)
		last_obs, success = executor.run("CloseObject Drawer", num_steps=200)
		print(f"Final pose: {last_obs['pose']}")
	finally:
		executor.close()

	# ee_T_cam, world_T_base = identify_robot_base_and_ee_to_wrist_transform(look_at_marker_pose=LOOK_AT_MARKER_POSE)
	# print(f"EE-to-wrist-camera transform (ee_T_cam):\n{ee_T_cam}")
	# print(f"Robot base in world/marker frame (world_T_base):\n{world_T_base}")
	# np.savez_compressed("robot_base_and_ee_to_wrist_transform.npz", ee_T_cam=ee_T_cam, world_T_base=world_T_base)
	# ee_T_cam, world_T_base = np.load("robot_base_and_ee_to_wrist_transform.npz")["ee_T_cam"], np.load("robot_base_and_ee_to_wrist_transform.npz")["world_T_base"]
	# # visualize_calibration(ee_T_cam, world_T_base)
	# env = Franka_Gym_Environment.FrankaGymEnvironment(load_vision_node=True, use_gui=False, episode_timeout=10000, reset_wait=0, open_gripper_on_reset=True, load_owl=True)
	# env.reset()
	# grasp_policy = ObjectGraspPolicy(ee_T_cam, world_T_base, env.vision_node)
	# grasp_policy.grasp_object("purple grapes", env)