from matplotlib import pyplot as plt
import json
import numpy as np
import matplotlib.font_manager as font_manager
from typing import Dict
import os


method_map = {"dense": "Dense Reward", "sparse": "Sparse Reward", "VIP": "VIP", "MPR": "MPR (Ours)",}
color_map = {"MPR (Ours)": "#0072B2", "VIP": "#D55E00", "Dense Reward": "#009E73", "Sparse Reward": "#CC79A7", "Base Policy": "#009E73"}

def generate_light_and_dark_colors(color_hex, shift_ratio = 1.7):
    import matplotlib.colors as mcolors
    # Convert hex to RGB
    rgb = mcolors.hex2color(color_hex)
    # Convert RGB to HSV
    hsv = mcolors.rgb_to_hsv(rgb)
    # Generate lighter color by increasing value
    light_hsv = (hsv[0], hsv[1], min(hsv[2] * shift_ratio, 1.0))
    # Generate darker color by decreasing value
    dark_hsv = (hsv[0], min(hsv[1]*1.5, 1.0), hsv[2] / shift_ratio)
    # Convert back to RGB
    light_rgb = mcolors.hsv_to_rgb(light_hsv)
    dark_rgb = mcolors.hsv_to_rgb(dark_hsv)
    # Convert back to hex
    light_hex = mcolors.rgb2hex(light_rgb)
    dark_hex = mcolors.rgb2hex(dark_rgb)
    return light_hex, dark_hex
    

def plot_successes(success_path, base_policy_success = 0.5):
    with open(success_path, 'r') as f:
        success_data = json.load(f)
    successes = success_data["successes"]
    ep_timesteps = success_data["timesteps"]
    # Compute a sliding window of average successes across the last 20 episodes
    window_size = 20
    avg_successes = [np.mean(successes[max(0, i - window_size + 1):i + 1]) for i in range(len(successes))]


    episodes = list(range(1, len(successes) + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, avg_successes)
    plt.xlabel('Episode')
    plt.ylabel('Average Success (Last 20 Episodes)')
    plt.title('Episode Successes Over Time')
    plt.ylim(-0.1, 1.1)
    plt.grid(True)
    plt.show()

def plot_reward_and_value(mpr_path, vip_path, plot_name: str, smooth_values = False, save_path: str = ""):
    mpr_data = np.load(mpr_path, allow_pickle=True)
    mpr_value = mpr_data["values"]
    mpr_reward = mpr_data["rewards"] - 1
    vip_data = np.load(vip_path, allow_pickle=True)
    vip_value = vip_data["values"]
    vip_reward = vip_data["rewards"]

    # print(mpr_value)
    # print(vip_value)
    if smooth_values:
        # Pad the edges of each array with the same value
        mpr_value = np.pad(mpr_value, (2, 2), mode='edge')
        vip_value = np.pad(vip_value, (2, 2), mode='edge')
        mpr_value = np.convolve(mpr_value, np.ones(3)/3, mode='valid')
        vip_value = np.convolve(vip_value, np.ones(3)/3, mode='valid')


    # mpr_reward /= np.max(np.abs(mpr_reward))
    # vip_reward /= np.max(np.abs(vip_reward))
    # mpr_value /= np.max(np.abs(mpr_value))
    # vip_value /= np.max(np.abs(vip_value))
    mpr_reward /= np.abs(mpr_reward[0])
    vip_reward /= np.abs(vip_reward[0])
    mpr_value /= np.abs(mpr_value[0])
    vip_value /= np.abs(vip_value[0])
    font = font_manager.FontProperties(weight='bold')

    #Create two plots vertically stacked, the top should show MPR and VIP rewards and the bottom should show MPR and VIP values
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(mpr_reward, label="MPR (Ours) Estimated Reward", color=color_map["MPR (Ours)"])
    axs[0].plot(vip_reward, label="VIP Estimated Reward", color=color_map["VIP"])
    axs[0].set_ylim(-1.05, -0.5)
    axs[0].set_xlabel('Timesteps', fontweight='bold')
    axs[0].set_ylabel('Reward', fontweight='bold')
    axs[0].set_title('Reward Comparison', fontweight='bold')
    axs[0].legend(prop=font)
    axs[0].grid(True)

    axs[1].plot(mpr_value, label="MPR (Ours) Estimated Value", color=color_map["MPR (Ours)"])
    axs[1].plot(vip_value, label="VIP Estimated Value", color=color_map["VIP"])
    axs[1].set_ylim(-1.5, 0.1)
    axs[1].set_xlabel('Timesteps', fontweight='bold')
    axs[1].set_ylabel('Value', fontweight='bold')
    axs[1].set_title('Value Comparison', fontweight='bold')
    axs[1].legend(prop=font)
    axs[1].grid(True)

    fig.suptitle(plot_name, fontweight='bold', fontsize=16)

    plt.tight_layout()
    if save_path == "":
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight', dpi=1000)
        if save_path.endswith(".png"):
            plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches='tight')
    plt.close()

def plot_rl_success_rates(runs_folder: str, task_name: str, save_path: str = "", base_policy_success: float = 0.45, averaging_window: int = 20, num_episodes: int = 100):
    methods = os.listdir(runs_folder)
    method_dict = {}
    for method in methods:
        runs = os.listdir(os.path.join(runs_folder, method))
        runs.sort()
        all_successes = []
        for run in runs:
            success_path = os.path.join(runs_folder, method, run, "successes.json")
            with open(success_path, 'r') as f:
                success_data = json.load(f)
            successes = success_data["successes"][:num_episodes]
            all_successes.append(successes)
        all_successes = np.array(all_successes)
        print(all_successes.shape)
        # Compute a sliding window average
        avg_successes = np.array([np.mean(all_successes[:, max(0, i - averaging_window + 1):i + 1], axis=1) for i in range(num_episodes)]).T
        print(avg_successes.shape)
        method_dict[method] = avg_successes
    font = font_manager.FontProperties(weight='bold')
    plt.figure(figsize=(10, 5))
    methods = list(method_dict.keys())
    methods.sort()
    if "MPR" in methods:
        methods.append(methods.pop(methods.index("MPR")))
    for method in methods:
        print(method)
        avg_successes = method_dict[method][:,averaging_window - 1:]
        print(avg_successes.shape)
        # Add the base policy success as the first point
        # avg_successes = np.hstack((np.ones((avg_successes.shape[0],1)) * base_policy_success, avg_successes))
        # timesteps = np.concatenate(([0], np.arange(averaging_window, num_episodes + 1)))
        timesteps = np.arange(averaging_window, num_episodes + 1)
        print(avg_successes.shape)
        print(timesteps.shape)
        method_label = method_map.get(method, method)
        method_color = color_map[method_label]
        light_color, dark_color = generate_light_and_dark_colors(method_color)
        colors = [light_color, method_color, dark_color]
        for i in range(avg_successes.shape[0]):
            plt.plot(timesteps, avg_successes[i], label=method_label + f" Run {i+1}", color=colors[i])
    plt.plot([0, num_episodes], [base_policy_success, base_policy_success], linestyle='--', label="Base Policy", color=color_map.get("Base Policy", None))
    plt.xlabel('Episodes', fontweight='bold')
    plt.ylabel(f'Average Success Rate Across Last {averaging_window} Episodes', fontweight='bold')
    plt.title(f'Success Rate Comparison on {task_name}', fontweight='bold')
    plt.ylim(-0.1, 1.1)
    plt.grid(True)
    plt.legend(prop=font)
    if save_path == "":
        plt.show()
    else:   
        plt.savefig(save_path, bbox_inches='tight', dpi=1000)
        if save_path.endswith(".png"):
            plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches='tight')
    plt.close()

def main():
    plot_successes("/home/mverghese/franka_control/runs/wipe_counter/VIP/wipe counter__reward_mode_VIP_offline_ratio_0.5_seed_0/successes.json")
    # mpr_path = "/home/mverghese/franka_control/MPR_Seed_1_Examples/fail/demo_3_MPR_rewards_values.npz"
    # vip_path = "/home/mverghese/franka_control/MPR_Seed_1_Examples/fail/demo_3_VIP_rewards_values.npz"
    # plot_reward_and_value(mpr_path, vip_path)


    # mpr_path = "/home/mverghese/franka_control/VIP_Seed_0_Examples/fail/demo_2_MPR_rewards_values.npz"
    # vip_path = "/home/mverghese/franka_control/VIP_Seed_0_Examples/fail/demo_2_VIP_rewards_values.npz"
    # plot_reward_and_value(mpr_path, vip_path, "Failed Episode", smooth_values=True, save_path="demo_2_fail_rewards_values.png")

    # mpr_path = "/home/mverghese/franka_control/MPR_Seed_1_Examples/success/demo_2_MPR_rewards_values.npz"
    # vip_path = "/home/mverghese/franka_control/MPR_Seed_1_Examples/success/demo_2_VIP_rewards_values.npz"
    # plot_reward_and_value(mpr_path, vip_path, "Successful Episode", smooth_values=True, save_path="demo_2_success_rewards_values.png")

    runs_folder = "/home/mverghese/franka_control/runs/wipe_counter"
    plot_rl_success_rates(runs_folder, "Real World Wipe Counter", save_path="wipe_counter_success_rates.png", base_policy_success=0.40, averaging_window=20, num_episodes=100)

if __name__ == "__main__":
    main()