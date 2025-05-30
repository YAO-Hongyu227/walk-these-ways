import isaacgym
import os
assert isaacgym
import torch
import numpy as np

import glob
import pickle as pkl

from go1_gym.envs import *
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv

from tqdm import tqdm
# runs/gait-conditioned-agility/2025-05-20/train/091602.705910/checkpoints/adaptation_module_latest.jit
# runs/gait-conditioned-agility/2025-05-20/train/091602.705910/checkpoints/body_latest.jit
# runs/gait-conditioned-agility/2025-05-20/train/091602.705910/checkpoints/ac_weights_last.pt
def load_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    import os
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')
    
    body_path = logdir + '/checkpoints/body_latest.jit'
    adaptation_module_path = logdir + '/checkpoints/adaptation_module_latest.jit'

    print(f"Body model path: {body_path}")
    print(f"Adaptation module path: {adaptation_module_path}")
    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy


def load_env(label, headless=False):
    dirs = glob.glob(f"../runs/{label}/*")
    dirs = [d for d in dirs if ".%f" not in os.path.basename(d)]
    logdir = sorted(dirs)[-1]

    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)

    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "P"

    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=False, cfg=Cfg)
    env = HistoryWrapper(env)



    # load policy
    from ml_logger import logger
    from go1_gym_learn.ppo_cse.actor_critic import ActorCritic

    policy = load_policy(logdir)

    return env, policy


def play_go1(headless=True):
    from ml_logger import logger

    from pathlib import Path
    from go1_gym import MINI_GYM_ROOT_DIR
    import glob
    import os
    from isaacgym import gymapi


    # label = "gait-conditioned-agility/2025-05-20/train"
    label = "gait-conditioned-agility/2025-05-30/train"

    env, policy = load_env(label, headless=headless)

    num_eval_steps = 500*2
    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5]}

    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 1.5, 0.0, 0.0
    body_height_cmd = 0.0
    step_frequency_cmd = 3.0
    gait = torch.tensor(gaits["trotting"])
    footswing_height_cmd = 0.08*1.5
    pitch_cmd = 0.0
    roll_cmd = 0.0
    stance_width_cmd = 0.25 * 1.5

    measured_x_vels = np.zeros(num_eval_steps)
    target_x_vels = np.ones(num_eval_steps) * x_vel_cmd
    joint_positions = np.zeros((num_eval_steps, 12))

    obs = env.reset()

    saved_obs_list = []

    sim_time_log = []
    sim_time = 0.0  # 初始化时间戳

    saved_actions = []

    # for key, value in obs.items():
    #     value_np = value.detach().cpu().numpy()
    #     shape = value_np.shape
    #     print(f"\nKey: '{key}' | Shape: {shape}")
    #     print("Values:", value_np.flatten()[:5], "...")


    for i in tqdm(range(num_eval_steps)):
        with torch.no_grad():
            actions = policy(obs)
            saved_actions.append(actions[0].detach().cpu().numpy().copy())  # [0] 是因为 actions shape 是 (1, 12)
            # print(actions)
        env.commands[:, 0] = x_vel_cmd
        env.commands[:, 1] = y_vel_cmd
        env.commands[:, 2] = yaw_vel_cmd
        env.commands[:, 3] = body_height_cmd
        env.commands[:, 4] = step_frequency_cmd
        env.commands[:, 5:8] = gait
        env.commands[:, 8] = 0.5
        env.commands[:, 9] = footswing_height_cmd
        env.commands[:, 10] = pitch_cmd
        env.commands[:, 11] = roll_cmd
        env.commands[:, 12] = stance_width_cmd
        obs, rew, done, info = env.step(actions)


        sim_time_log.append(sim_time)
        sim_time += env.dt  # env.dt = dt * decimation，正是控制间隔


        obs_vec = obs['obs'].squeeze().cpu().numpy()
        saved_obs_list.append(obs_vec.copy())

        # print(obs)
        # print(obs['obs'])

        def print_obs_structure(obs_dict, max_values=2100):
            for key, value in obs_dict.items():
                value_np = value.detach().cpu().numpy()
                shape = value_np.shape
                print(f"\nKey: '{key}' | Shape: {shape}")
                print("Values:", value_np.flatten()[:max_values], "...")

        # 示例调用
        # print_obs_structure(obs)
        # print(obs['privileged_obs'])

        # obs_vec = obs['obs'].squeeze().cpu().numpy()
        # print("projected_gravity:", obs_vec[0:3])
        # print("commands:", obs_vec[3:18])
        # print("dof_pos:", obs_vec[18:30])
        # print("dof_vel:", obs_vec[30:42])
        # print("actions:", obs_vec[42:54])
        # print("last_actions:", obs_vec[54:66])
        # print("clock_inputs:", obs_vec[66:70])





        measured_x_vels[i] = env.base_lin_vel[0, 0]
        joint_positions[i] = env.dof_pos[0, :].cpu()

    import matplotlib.pyplot as plt

    # time_diffs = np.diff(sim_time_log)
    # plt.plot(time_diffs)
    # plt.title("Control Interval (s)")
    # plt.xlabel("Control Step")
    # plt.ylabel("Δt")
    # plt.ylim(env.dt - 0.001, env.dt + 0.001)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

####################################################  打印12个关节角度  ##########################################################################
    plot_joint_angles = False
    if plot_joint_angles:
        saved_actions = np.array(saved_actions)  # shape: [num_eval_steps, 12]
        time_axis = np.linspace(0, num_eval_steps * env.dt, num_eval_steps)

        joint_names = [
            "FL_hip", "RL_hip", "FR_hip", "RR_hip",
            "FL_thigh", "RL_thigh", "FR_thigh", "RR_thigh",
            "FL_calf", "RL_calf", "FR_calf", "RR_calf"
        ]

        # 每个组单独画一个 4×1 子图
        joint_groups = {
            "hip": [0, 3, 6, 9],
            "thigh": [1, 4, 7, 10],
            "calf": [2, 5, 8, 11],
        }

        for group_name, indices in joint_groups.items():
            fig, axs = plt.subplots(4, 1, figsize=(16, 10), sharex=True)  # 横向尺寸加倍
            fig.suptitle(f"{group_name.capitalize()} Joint Actions Over Time", fontsize=16)
            
            for i, idx in enumerate(indices):
                axs[i].plot(time_axis, saved_actions[:, idx], label=joint_names[idx], color='black')
                axs[i].set_ylabel("Action")
                axs[i].set_ylim([-4, 4])  # 固定 Y 轴
                axs[i].set_title(joint_names[idx])
                axs[i].grid(True)
            
            axs[-1].set_xlabel("Time (s)")
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.savefig(f"logs/{group_name}_joints_4x1_go1.png")
            plt.show()


##################################################################################################################################

    os.makedirs("logs", exist_ok=True)
    obs_array = np.stack(saved_obs_list, axis=0)
    np.savetxt("logs/obs_go1.txt", obs_array, fmt="%.6f")
    print(f"Saved {obs_array.shape[0]} obs vectors to logs/obs_go1.txt")


    # plot target and measured forward velocity
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(12, 5))
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_x_vels, color='black', linestyle="-", label="Measured")
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), target_x_vels, color='black', linestyle="--", label="Desired")
    axs[0].legend()
    axs[0].set_title("Forward Linear Velocity")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Velocity (m/s)")

    axs[1].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), joint_positions, linestyle="-", label="Measured")
    axs[1].set_title("Joint Positions")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Joint Position (rad)")

    plt.tight_layout()  
    plt.show()

    plot_scaled_actions = True
    if plot_scaled_actions:
        logged_actions = np.array(env.logged_actions).reshape(-1, 12)
        time_axis = np.linspace(0, len(logged_actions) * env.dt, len(logged_actions))

        joint_names = [
            "FL_hip", "RL_hip", "FR_hip", "RR_hip",
            "FL_thigh", "RL_thigh", "FR_thigh", "RR_thigh",
            "FL_calf", "RL_calf", "FR_calf", "RR_calf"
        ]

        # 每个组单独画一个 4×1 子图
        joint_groups = {
            "hip": [0, 3, 6, 9],
            "thigh": [1, 4, 7, 10],
            "calf": [2, 5, 8, 11],
        }

        for group_name, indices in joint_groups.items():
            fig, axs = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
            fig.suptitle(f"{group_name.capitalize()} Joint Actions (Scaled) Over Time", fontsize=16)
            
            for i, idx in enumerate(indices):
                axs[i].plot(time_axis, logged_actions[:, idx], label=joint_names[idx], color='black')
                axs[i].set_ylabel("Action")
                axs[i].set_ylim([-2, 2])
                axs[i].set_title(joint_names[idx])
                axs[i].grid(True)
            
            axs[-1].set_xlabel("Time (s)")
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.savefig(f"logs/{group_name}_logged_actions_4x1.png")
            # plt.show()
        
        # 保存 logged_actions 为 txt 文件
        os.makedirs("logs", exist_ok=True)
        np.savetxt("logs/logged_actions_scaled.txt", logged_actions, fmt="%.6f")
        print(f"Saved {logged_actions.shape[0]}×{logged_actions.shape[1]} actions to logs/logged_actions_scaled.txt")


    from PIL import Image

    # 图像文件路径
    hip_img = Image.open("logs/hip_logged_actions_4x1.png")
    thigh_img = Image.open("logs/thigh_logged_actions_4x1.png")
    calf_img = Image.open("logs/calf_logged_actions_4x1.png")

    # 获取最大高度，总宽度
    total_width = hip_img.width + thigh_img.width + calf_img.width
    max_height = max(hip_img.height, thigh_img.height, calf_img.height)

    # 创建新图像用于拼接
    combined_img = Image.new('RGB', (total_width, max_height), color=(255, 255, 255))

    # 粘贴每张图
    x_offset = 0
    for img in [hip_img, thigh_img, calf_img]:
        combined_img.paste(img, (x_offset, 0))
        x_offset += img.width

    # 保存最终图像
    combined_img.save("logs/combined_joint_actions.png")
    combined_img.show()



if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_go1(headless=False)
