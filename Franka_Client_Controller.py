from multiprocessing.managers import BaseManager
from queue import Queue
import numpy as np
import torch
 
def check_quaternion_normalized(quat):
    norm = torch.norm(quat)
    return torch.isclose(norm, torch.tensor(1.0), atol=1e-3)

def normalize_quaternion(quat):
    norm = torch.norm(quat)
    if norm < 1e-6:
        raise ValueError("Cannot normalize a zero-length quaternion.")
    return quat / norm

class QueueManager(BaseManager):
    pass

class FrankaClientController:
    def __init__(self, address=('localhost', 50000), authkey=b'abc'):
        self.address = address
        self.authkey = authkey
        self.send_queue = None
        self.recv_queue = None
        QueueManager.register('get_send_queue')
        QueueManager.register('get_recv_queue')
        self.manager = QueueManager(address=self.address, authkey=self.authkey)
        self.manager.connect()
        self.send_queue = self.manager.get_send_queue()
        self.recv_queue = self.manager.get_recv_queue()

    def reconnect(self):
        QueueManager.register('get_send_queue')
        QueueManager.register('get_recv_queue')
        self.manager = QueueManager(address=self.address, authkey=self.authkey)
        self.manager.connect()
        self.send_queue = self.manager.get_send_queue()
        self.recv_queue = self.manager.get_recv_queue()

    def send_command(self, command):
        if self.recv_queue.full():
            raise RuntimeError("Recv queue is full. The server did not process the previous command yet.")
        # print(f"Client Sending command: {command}")
        self.recv_queue.put(command)
    
    def move_to_absolute_pose(self, pose):
        '''pose is an 8-vector consisting of [x,y,z,qx,qy,qz,qw,gripper_width]'''
        if isinstance(pose, np.ndarray):
            pose = pose.tolist()
        if isinstance(pose, torch.Tensor):
            pose = pose.tolist()
        command = {
            'function': 'move_to_absolute_pose',
            'val': pose
        }
        self.send_command(command)

    def move_to_relative_pose(self, pose):
        '''pose is an 8-vector consisting of [dx,dy,dz,dqx,dqy,dqz,dqw,gripper_width].
        Note that the gripper width is still an absolute value even in the relative pose command.'''
        if isinstance(pose, np.ndarray):
            pose = pose.tolist()
        if isinstance(pose, torch.Tensor):
            pose = pose.tolist()
        command = {
            'function': 'move_to_relative_pose',
            'val': pose
        }
        self.send_command(command)
    
    def move_to_absolute_joint_pose(self, pose):
        '''pose is an 7-vector consisting of [j1,j2,j3,j4,j5,j6,j7,gripper_width]'''
        if isinstance(pose, np.ndarray):
            pose = pose.tolist()
        if isinstance(pose, torch.Tensor):
            pose = pose.tolist()
        command = {
            'function': 'move_to_absolute_joint_pose',
            'val': pose
        }
        self.send_command(command)
    
    def move_to_relative_joint_pose(self, pose):
        '''pose is an 7-vector consisting of [dj1,dj2,dj3,dj4,dj5,dj6,dj7,gripper_width]'''
        if isinstance(pose, np.ndarray):
            pose = pose.tolist()
        if isinstance(pose, torch.Tensor):

            pose = pose.tolist()
        command = {
            'function': 'move_to_relative_joint_pose',
            'val': pose
        }
        self.send_command(command)
    
    def close_gripper(self, speed=0.05, force=0.05):
        command = {
            'function': 'close_gripper',
            'val': [speed, force]
        }
        self.send_command(command)
    
    def open_gripper(self, speed=0.05, force=0.05):
        command = {
            'function': 'open_gripper',
            'val': [speed, force]
        }
        self.send_command(command)

    def go_home(self, droid_home = False, open_gripper = True):
        command = {
            'function': 'go_home',
            'val': [droid_home, open_gripper]
        }
        self.send_command(command)

    def null_command(self):
        command = {
            'function': 'null_command'
        }
        self.send_command(command)

    def shutdown(self):
        command = {
            'function': 'shutdown'
        }
        self.send_command(command)

    def get_state(self):
        try:
            ret = self.send_queue.get()
        except EOFError:
            print("EOFError encountered while trying to get state from server. Attempting to reconnect...")
            _ = input("Please ensure the server is running and then press Enter to continue...")
            self.reconnect()
            self.null_command()
            ret = self.send_queue.get()
        # print(f"Client Received state: ee_pos: {ret['ee_pos']}, ee_quat: {ret['ee_quat']}")
        return ret["ee_pos"], ret["ee_quat"], ret["joint_positions"], ret["joint_velocities"], ret["gripper_width"]
