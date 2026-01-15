from polymetis import RobotInterface, GripperInterface
import torch
import numpy as np
from multiprocessing.managers import BaseManager
from queue import Queue
from torchcontrol.transform import Rotation as R
from torchcontrol.transform import Transformation as T
import time
class QueueManager(BaseManager):
    pass

def check_quaternion_normalized(quat):
    norm = torch.norm(quat)
    return torch.isclose(norm, torch.tensor(1.0), atol=1e-3)


class FrankaServerController:
    def __init__(self, ip_address="172.16.0.1", address=('localhost', 50000), authkey=b'abc', reply_delay = 1/50):
        self.robot = RobotInterface(
            ip_address=ip_address,
        )
        self.gripper = GripperInterface(
            ip_address=ip_address,
        )
        self.address = address
        self.authkey = authkey
        send_queue = Queue(maxsize=1)
        recv_queue = Queue(maxsize=1)
        QueueManager.register('get_send_queue', callable=lambda: send_queue)
        QueueManager.register('get_recv_queue', callable=lambda: recv_queue)
        self.manager = QueueManager(address=self.address, authkey=self.authkey)
        self.manager.start()
        self.send_queue = self.manager.get_send_queue()
        self.recv_queue = self.manager.get_recv_queue()
        self.reply_delay = reply_delay

    def initialize_robot(self):
        self.open_gripper(None)
        self.robot.go_home()
        self.robot.start_cartesian_impedance()
        self.ee_pos, self.ee_quat = self.robot.get_ee_pose()
    
    def move_to_absolute_pose(self, pose):
        # if not self.robot.is_running_policy():
        #     self.robot.start_cartesian_impedance()
        pos, quat, gripper_width = pose[:3], pose[3:7], pose[7]
        if not check_quaternion_normalized(quat):
            raise ValueError(f"Quaternion {quat} is not normalized.")
        self.robot.update_desired_ee_pose(position=pose[:3], orientation=pose[3:])
        self.gripper.goto(width=gripper_width, speed=0.05, force=0.1)

    def move_to_relative_pose(self, pose):
        '''Note: the gripper width is still an absolute value even in the relative pose command'''
        # if not self.robot.is_running_policy():
        #     self.robot.start_cartesian_impedance()
        pos_delta = pose[:3]
        quat_delta = pose[3:7]
        gripper_width = pose[7]
        if not check_quaternion_normalized(quat_delta):
            raise ValueError(f"Quaternion {quat_delta} is not normalized.")
        new_pos = self.ee_pos + pos_delta
        new_quat = (R.from_quat(quat_delta) * R.from_quat(self.ee_quat)).as_quat()
        
        # print(f"Moving to new absolute pose: pos {new_pos}, quat {new_quat}")
        self.robot.update_desired_ee_pose(position=new_pos, orientation=new_quat)
        self.gripper.goto(width=gripper_width, speed=0.05, force=0.1)

    def go_home(self):
        self.robot.terminate_current_policy()
        self.open_gripper(None)
        self.robot.go_home()
        self.robot.start_cartesian_impedance()

    def close_gripper(self, vals):
        speed = 0.05
        force = 0.1
        if vals is not None:
            speed = vals[0]
            force = vals[1]
        self.gripper.grasp(speed=speed, force=force)
    
    def open_gripper(self, vals):
        speed = 0.05
        force = 0.1
        if vals is not None:
            speed = vals[0]
            force = vals[1]
        self.gripper.goto(width=0.08, speed=speed, force=force)

    def run(self):
        self.initialize_robot()
        try:
            while True:

                # Get the command from the receive queue
                command = self.recv_queue.get()
                # print(f"Server Received command: {command}")
                function = command.get('function')
                val = command.get('val', None)
                val = torch.tensor(val) if val is not None else None
                if function == 'move_to_absolute_pose':
                    self.move_to_absolute_pose(val)
                elif function == 'move_to_relative_pose':
                    self.move_to_relative_pose(val)
                elif function == 'close_gripper':
                    self.close_gripper(val)
                elif function == 'open_gripper':
                    self.open_gripper(val)
                elif function == "go_home":
                    self.go_home()
                elif function == 'shutdown':
                    self.shutdown()
                    break

                # Wait for a short duration to allow the robot to move
                time.sleep(self.reply_delay)

                self.ee_pos, self.ee_quat = self.robot.get_ee_pose()
                self.gripper_width = self.gripper.get_state().width
                # convert the tensors to lists for sending through the queue
                ee_pos = self.ee_pos.tolist()
                ee_quat = self.ee_quat.tolist()
                # Send the current end-effector pose back through the send queue
                if self.send_queue.full():
                    # _ = self.send_queue.get()
                    # print("Warning: send_queue was full, overwriting old data. This is likely undesired.")
                    raise RuntimeError("Send queue is full. The client controller did not read the last observation")
                self.send_queue.put_nowait({'ee_pos': ee_pos, 'ee_quat': ee_quat, 'gripper_width': self.gripper_width})
                # print(f"Server Sent state: ee_pos: {ee_pos}, ee_quat: {ee_quat}")
        except KeyboardInterrupt:
            print("Shutting down Franka server controller.")
        finally:
            self.shutdown()

    def shutdown(self):
        self.robot.terminate_current_policy()
        self.manager.shutdown()


if __name__ == "__main__":
    server_controller = FrankaServerController(reply_delay=1/10)
    server_controller.run()

