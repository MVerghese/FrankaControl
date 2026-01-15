import numpy as np
from scipy.spatial.transform import Rotation as Rotation
from scipy.spatial.transform import Slerp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.path import Path
# import open3d as o3d


def rodriguesrot(x,omega,theta):
	xrot = x*np.cos(theta) + np.cross(omega,x)*np.sin(theta)+omega*np.dot(omega,x)*(1-np.cos(theta))
	return(xrot)

def rodriguesVec(vec, n0, n1):
	# Get vector of rotation k and angle theta
	n0 = n0/np.linalg.norm(n0)
	n1 = n1/np.linalg.norm(n1)
	k = np.cross(n0,n1)
	k = k/np.linalg.norm(k)
	theta = np.arccos(np.dot(n0,n1))
	return(rodriguesrot(vec,k,theta))

def fitPlane(pointcloud, max_points = 20):
	if pointcloud.shape[0] > max_points:
		U,E,V = np.linalg.svd(pointcloud[np.linspace(0,pointcloud.shape[0]-1,max_points).astype(int),:])
	else:
		U,E,V = np.linalg.svd(pointcloud)
	normal_vec = V.T[:,0]
	center = np.mean(pointcloud,axis=0)
	return(normal_vec,center)

def coordTransform(coord, extrinsics):
	homogenous = np.append(coord,1.).reshape((4,1))
	return(np.dot(extrinsics,homogenous)[:3])

def Euler2SO3(angles):
	x, y, z = angles[0], angles[1], angles[2]
	Rx = np.array([[1,0,0],
				   [0,np.cos(x),-np.sin(x)],
				   [0,np.sin(x),np.cos(x)]])
	Ry = np.array([[np.cos(y),0,np.sin(y)],
				   [0,1,0],
				   [-np.sin(y),0,np.cos(y)]])
	Rz = np.array([[np.cos(z),-np.sin(z),0],
				   [np.sin(z),np.cos(z),0],
				   [0,0,1]])
	return(np.dot(Rx,Ry).dot(Rz))

def vec2SE3(vec,intrinsic=True):
	T = np.eye(4)
	if intrinsic:
		T[:3,:3] = Euler2SO3(vec[3:])
	else:
		r = Rotation.from_euler('xyz',vec[3:])
		T[:3,:3] = r.as_matrix()
	T[:3,3] = vec[:3]
	return(T)

def tvecrvec2SE3(tvec,rvec):
	T = np.eye(4)
	T[:3,:3] = Euler2SO3(rvec)
	T[:3,3] = rvec
	return(T)

def invertSE3(T):
	R = T[:3,:3]
	p = T[:3,3]
	outT = np.eye(4)
	outT[:3,:3] = R.T
	outT[:3,3] = np.dot(R.T,p)*-1
	return(outT)

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
	Rt = np.transpose(R)
	shouldBeIdentity = np.dot(Rt, R)
	I = np.identity(3, dtype = R.dtype)
	n = np.linalg.norm(I - shouldBeIdentity)
	invert_check =  n < 1e-6
	magnitudes = np.linalg.norm(R,axis=0)
	magnitude_check = np.all(np.abs(magnitudes - 1) < 1e-6)
	dots = np.array([np.dot(R[:,0],R[:,1]),np.dot(R[:,0],R[:,2]),np.dot(R[:,1],R[:,2])])
	orthogonality_check = np.all(np.abs(dots) < 1e-6)
	if invert_check and magnitude_check and orthogonality_check:
		return True
	else:
		print("invert check",invert_check, n)
		print("magnitude check",magnitude_check, magnitudes)
		print("orthogonality check",orthogonality_check, dots)
		return False

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def SO32Euler(R) :

	assert(isRotationMatrix(R))

	sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

	singular = sy < 1e-6

	if  not singular :
		x = np.arctan2(R[2,1] , R[2,2])
		y = np.arctan2(-R[2,0], sy)
		z = np.arctan2(R[1,0], R[0,0])
	else :
		print("singular")
		x = np.arctan2(-R[1,2], R[1,1])
		y = np.arctan2(-R[2,0], sy)
		z = 0

	return np.array([x, y, z])

def SE32vec(T):
	vec = np.zeros(6)
	vec[:3] = T[:3,3]
	vec[3:] = SO32Euler(T[:3,:3])
	return(vec)

def leftHanded2RightHanded(R):
	pass


def SO32so3(R):
	# theta = np.arccos(np.clip((np.trace(R)-1)/2, -1, 1))
	# omega = 1/(2*np.sin(theta))*np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])
	# return(omega*theta)
	# if np.linalg.det(R) < 0:
	return(Rotation.from_matrix(R).as_rotvec())
	

def SE32AA(T):
	ret = np.zeros(6)
	ret[:3] = T[:3,3]
	ret[3:] = SO32so3(T[:3,:3])
	return(ret)

def AA2SE3(T):
	ret = np.eye(4)
	ret[:3,3] = T[:3]
	ret[:3,:3] = so32SO3(T[3:])
	return(ret)

def vec2SE2(vec):
	T = np.eye(3)
	T[:2,:2] = Euler2SO2(vec[2])
	T[:2,2] = vec[:2]
	return(T)

def Euler2SO2(theta):
	return(np.array([[np.cos(theta),-np.sin(theta)],
					 [np.sin(theta),np.cos(theta)]]))

def invertSE2(T):
	R = T[:2,:2]
	p = T[:2,2]
	outT = np.eye(3)
	outT[:2,:2] = R.T
	outT[:2,2] = np.dot(R,p)*-1
	return(outT)

def interpolateTransforms(T1,T2,u):
	matstack = np.vstack((T1[:3,:3].reshape(1,3,3),T2[:3,:3].reshape(1,3,3)))
	rots = Rotation.from_matrix(matstack)
	slerp = Slerp([0,1],rots)
	Tret = np.eye(4)
	Tret[:3,:3] = slerp([u])[0].as_matrix()
	Tret[:3,3] = (T2[:3,3] - T1[:3,3])*u + T1[:3,3]
	return(Tret)

def hatmap(vec):
	return(np.array([[0,-vec[2],vec[1]],
					 [vec[2],0,-vec[0]],
					 [-vec[1],vec[0],0]]))

def so32SO3(omega):
	# theta = np.linalg.norm(omega)
	# omega /= theta
	# K = hatmap(omega)
	# return(np.eye(3)+np.sin(theta)*K+(1-np.cos(theta))*np.dot(K,K))
	return(Rotation.from_rotvec(omega).as_matrix())

def computeVectorRot(v1,v2):
	omega = np.cross(v1,v2)
	omega /= np.linalg.norm(omega)
	theta = np.arccos(np.dot(v1,v2)/np.linalg.norm(v1)/np.linalg.norm(v2))
	return(omega*theta)

def fix_rotation(ang):
	return((ang + np.pi)%(np.pi*2)-np.pi)

def build_intr_mat(intr):
	mat = np.eye(3)
	mat[0,0] = intr[2]
	mat[1,1] = intr[3]
	mat[0,2] = intr[0]
	mat[1,2] = intr[1]
	return(mat)

def vec_pose_distance(p1,p2,linear_weight = 1,rotational_weight=1):
	linear_dist = np.linalg.norm(p1[:3]-p2[:3])
	rot1 = p1[3:6]
	rot2 = p2[3:6]
	rotational_dist = np.linalg.norm(np.concatenate((np.sin(rot1),np.cos(rot1))) - np.concatenate((np.sin(rot2),np.cos(rot2))))
	return(linear_dist*linear_weight + rotational_dist*rotational_weight)

def pose_distance(p1,p2,linear_weight = 1,rotational_weight=1):
	linear_dist = np.linalg.norm(p1[:3,3]-p2[:3,3])
	R = np.dot(p1[:3,:3],p2[:3,:3].T)

	rotational_distance = np.arccos(np.clip((np.trace(R)-1)/2,-1,1))
	# print("linear dist",linear_dist, "rot dist",rotational_distance, "rot 1 ", textR(p1[:3,:3]), "rot 2 ", textR(p2[:3,:3]), "rot diff", SO32so3(R), "R", textR(R))

	return(linear_dist*linear_weight + rotational_distance*rotational_weight)

def rotation_distance(r1,r2):
	R = np.dot(r1,r2.T)
	rotational_distance = np.arccos(np.clip((np.trace(R)-1)/2,-1,1))
	return(rotational_distance)

def textR(R):
	return "["+", ".join(str(val) for val in R.flatten())+"]"

class FramePlot:
	def __init__(self):
		self.xlim = [-300,1000]
		self.ylim = [-1000,300]
		self.zlim = [-300,1000]

		self.fig= plt.figure()
		self.ax = self.fig.add_subplot(111,projection='3d')
		self.ax.set_xlim(self.xlim)
		self.ax.set_ylim(self.ylim)
		self.ax.set_zlim(self.zlim)
		self.ax.set_xlabel("X")
		self.ax.set_ylabel("Y")
		self.ax.set_zlabel("Z")

	def clearAxis(self):
		plt.cla()
		self.ax.set_xlim(self.xlim)
		self.ax.set_ylim(self.ylim)
		self.ax.set_zlim(self.zlim)
		self.ax.set_xlabel("X")
		self.ax.set_ylabel("Y")
		self.ax.set_zlabel("Z")

	def plotFrame(self,T,vec_length = 100, label = ""):
		pos = T[:3,3]
		R = T[:3,:3]
		xvec = np.dot(R,np.array([[vec_length,0,0]]).T).flatten()
		yvec = np.dot(R,np.array([[0,vec_length,0]]).T).flatten()
		zvec = np.dot(R,np.array([[0,0,vec_length]]).T).flatten()
		self.ax.quiver(pos[0],pos[1],pos[2],xvec[0],xvec[1],xvec[2],color='r')
		self.ax.quiver(pos[0],pos[1],pos[2],yvec[0],yvec[1],yvec[2],color='g')
		self.ax.quiver(pos[0],pos[1],pos[2],zvec[0],zvec[1],zvec[2],color='b')
		self.ax.text(pos[0],pos[1],pos[2],label,'y')

	def drawPlot(self,t =.05):
		plt.draw()
		plt.show(block=False)
		plt.pause(t)
		
	def showPlot(self):
		plt.show()

def rotation_sanity_check(num_samples):
	for i in range(num_samples):
		angles = np.random.rand(3)*np.pi*2
		R = Euler2SO3(angles)
		print(isRotationMatrix(R))
		print("Magniudes: ",np.linalg.norm(R[:,0]), np.linalg.norm(R[:,1]), np.linalg.norm(R[:,2]))
		print("Dot Products", np.dot(R[:,0],R[:,1]), np.dot(R[:,0],R[:,2]), np.dot(R[:,1],R[:,2]))

# def visualize_pc(pc, colors):
# 	pcd = o3d.geometry.PointCloud()
# 	pcd.points = o3d.utility.Vector3dVector(pc)
# 	pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255)

# 	vis = o3d.visualization.Visualizer()
# 	vis.create_window()
# 	vis.add_geometry(pcd)
# 	vis.run()
# 	vis.destroy_window()


# def fuse_point_clouds(pcs, pc_colors, ref_index, icp = True, visualize = False):
# 	if visualize:
# 		all_pcs = np.concatenate(pcs, axis=0)
# 		all_colors = np.concatenate(pc_colors, axis=0)
# 		visualize_pc(all_pcs, all_colors)
	
# 	target_pcd = o3d.geometry.PointCloud()
# 	target_pcd.points = o3d.utility.Vector3dVector(pcs[ref_index])
# 	target_pcd.colors = o3d.utility.Vector3dVector(pc_colors[ref_index].astype(np.float64) / 255)
# 	updated_pcs = []
# 	for i in range(len(pcs)):
# 		if i == ref_index:
# 			updated_pcs.append(target_pcd)
# 			continue
# 		source_pcd = o3d.geometry.PointCloud()
# 		source_pcd.points = o3d.utility.Vector3dVector(pcs[i])
# 		source_pcd.colors = o3d.utility.Vector3dVector(pc_colors[i].astype(np.float64) / 255)
		
# 		# Apply ICP registration
# 		if icp:
# 			registration_result = o3d.pipelines.registration.registration_icp(
# 				source_pcd, 
# 				target_pcd, 
# 				max_correspondence_distance=0.02, 
# 				init = np.eye(4),
# 				estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())
			
# 			print("ICP converged:", registration_result.fitness)
# 			print("Transformation matrix:\n", registration_result.transformation)
			
# 			source_pcd.transform(registration_result.transformation)
# 		dists = source_pcd.compute_point_cloud_distance(target_pcd)
# 		ind = np.where(np.array(dists) > 0.002)[0]
# 		source_pcd = source_pcd.select_by_index(ind)
# 		updated_pcs.append(source_pcd)
	
# 	# Combine all point clouds
# 	combined_pcd = o3d.geometry.PointCloud()
# 	for pc in updated_pcs:
# 		combined_pcd += pc
# 	pc_numpy = np.asarray(combined_pcd.points)
# 	pc_colors_numpy = np.asarray(combined_pcd.colors)
# 	pc_colors_numpy = (pc_colors_numpy * 255).astype(np.uint8)
# 	if visualize:
# 		visualize_pc(pc_numpy, pc_colors_numpy)

# 	return pc_numpy, pc_colors_numpy

def benchmark_rotations(num_samples=1000):
	np.random.seed(42)  # For reproducibility
	angles = np.random.uniform(-np.pi/1.01,np.pi/1.01, size=(num_samples, 3))
	print(angles.shape)
	# SO3s = Rotation.from_euler('xyz', angles).as_matrix()
	# print(SO3s.shape)
	# ret_angles = Rotation.from_matrix(SO3s).as_euler('xyz')
	# print(ret_angles.shape)
	
	for i in range(num_samples):
		# SO3 = Rotation.from_euler('xyz', angles[i]).as_matrix()
		# ret_angles = Rotation.from_matrix(SO3).as_euler('xyz')
		ret_angles = Rotation.from_euler('XYZ', angles[i]).as_euler('XYZ')

		assert np.allclose(angles[i], ret_angles), f"Rotation mismatch at sample {i}, original: {angles[i]}, recovered: {ret_angles}"

	print(f"All {num_samples} rotations passed the sanity check.")




if __name__ == "__main__":
	np.set_printoptions(suppress=True)
	benchmark_rotations(num_samples=1000)

	# T = np.array([[ -0.86293704,   0.07971779,  -0.49898384, 491.04569061],
	# 			  [ -0.06713754,  -0.99681054,  -0.04314379, 272.42484282],
	# 			  [  0.50083169,   0.00372986,  -0.86553669, 222.09874391],
	# 			  [  0.        ,   0.        ,   0.        ,   1.        ]])
	# R = T[:3,:3]
	# print(np.linalg.det(R))
	# print(isRotationMatrix(R))
	# print("R_init\n",R)
	# print("R_post\n", so32SO3(SO32so3(R)))
	# print(Rotation.from_matrix(R).as_matrix())
	# rotation_sanity_check(10)
	# R = np.array([-0.54582428, -0.78986669,  0.27961808, -0.70921588,  0.61322006,  0.3478131,   0.44619346,  0.00846476,  0.89489643]).reshape((3,3))
	# print(isRotationMatrix(R))
	# aa_rot = SO32so3(R)
	# print("Rotation magnitude", np.linalg.norm(aa_rot))
	# print("Rotation distance", np.arccos(np.clip((np.trace(R)-1)/2,-1,1)))
	# R1 = np.array([0.999881204996914, -0.014908269184710889, 0.003914001000903549, -0.01510755016745925, -0.9982536960787457, 0.05710796961073078, 0.003055784982058816, -0.05716031643578129, -0.9983603359524577]).reshape((3,3))
	# R2 = np.array([-0.8644461930396976, -0.12119631185277299, -0.48789789122148447, 0.1061816578619548, -0.9926277220551234, 0.05844357314776713, 0.4913841316984235, 0.0012844908657327178, -0.8709420856334122]).reshape((3,3))
	# T1 = np.eye(4)
	# T1[:3,:3] = R1
	# T1[:3,3] = np.array([300,0,0])
	# T2 = np.eye(4)
	# T2[:3,:3] = R2
	# fp = FramePlot()
	# fp.plotFrame(T1, label = "R1")
	# fp.plotFrame(T2, label = "R2")
	# fp.showPlot()

	# rot_diff = np.array([-0.00423385, 0.03966656, -0.02277856])
	# R = np.array([ 1.36404415,  0.43932866, -0.18027906,  0.43600796,  0.38730787,  0.24031812, -0.31073854,  0.21397328,  0.9241595 ]).reshape((3,3))
	# R_prime = so32SO3(rot_diff)
	# print("Original R:\n",R)
	# print("R_prime:\n",R_prime)
	# rotational_distance = np.arccos(np.clip((np.trace(R)-1)/2,-1,1))
	# print("Original R rotational_distance", rotational_distance)
	# rotational_distance = np.arccos(np.clip((np.trace(R_prime)-1)/2,-1,1))
	# print("R_prime rotational_distance", rotational_distance)