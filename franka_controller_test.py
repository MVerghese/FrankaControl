from polymetis import RobotInterface, GripperInterface
import torch
import numpy as np
import time

def main():
    robot = RobotInterface(
        ip_address="172.16.0.1",
    )
    gripper = GripperInterface(
        ip_address="172.16.0.1",
    )

    robot.go_home()

    # Get ee pose
    ee_pos, ee_quat = robot.get_ee_pose()
    print(f"Current ee position: {ee_pos}")
    print(f"Current ee orientation: {ee_quat}  (xyzw)")

    # # Command robot to ee xyz position
    # ee_pos_desired = torch.Tensor([0.5, 0.0, 0.4])
    # print(f"\nMoving ee pos to: {ee_pos_desired} ...\n")
    # state_log = robot.move_to_ee_pose(
    #     position=ee_pos_desired, orientation=None, time_to_go=2.0
    # )

    # # Get updated ee pose
    # ee_pos, ee_quat = robot.get_ee_pose()
    # print(f"New ee position: {ee_pos}")
    # print(f"New ee orientation: {ee_quat}  (xyzw)")

    # Cartesian impedance control
    # print("Performing Cartesian impedance control...")
    ee_pos, ee_quat = robot.get_ee_pose()

    # example usages
    gripper_state = gripper.get_state()
    print(f"Gripper state: {gripper_state.width}")
    # for i in range(10):
    gripper.goto(width=0.09, speed=0.05, force=0.1)
        # time.sleep(1)
    # gripper.grasp(speed=0.05, force=0.1)

    # robot.start_cartesian_impedance()

    # for i in range(40):
    #     ee_pos += torch.Tensor([-0.0025, 0.0, 0.0])
    #     robot.update_desired_ee_pose(position=ee_pos)
    #     time.sleep(0.1)

    # robot.terminate_current_policy()


    robot.go_home()

if __name__ == "__main__":
    main()