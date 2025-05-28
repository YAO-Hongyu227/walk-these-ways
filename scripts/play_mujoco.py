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
import mujoco
import mujoco.viewer
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

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless = headless, cfg=Cfg)
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

    label = "gait-conditioned-agility/2025-05-20/train"

    env, policy = load_env(label, headless=headless)

#############################################
    xml_path = "/home/tong/Downloads/walk-these-ways/resources/robots/Q20/scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    
    
    default_joint_angles = {
    'FL_hip_joint': -0.25,
    'RL_hip_joint': -0.25,
    'FR_hip_joint': 0.25,
    'RR_hip_joint': 0.25,
    'FL_thigh_joint': -0.45,
    'RL_thigh_joint': -0.9,
    'FR_thigh_joint': -0.45,
    'RR_thigh_joint': -0.9,
    'FL_calf_joint': 1.4,
    'RL_calf_joint': 1.3,
    'FR_calf_joint': 1.4,
    'RR_calf_joint': 1.3,
    }

    # default_joint_angles = {
    # 'FL_hip_joint': -0.0,
    # 'RL_hip_joint': -0.0,
    # 'FR_hip_joint': 0.0,
    # 'RR_hip_joint': 0.,
    # 'FL_thigh_joint': -0.,
    # 'RL_thigh_joint': -0.,
    # 'FR_thigh_joint': -0.,
    # 'RR_thigh_joint': -0.,
    # 'FL_calf_joint': 0.,
    # 'RL_calf_joint': 0.,
    # 'FR_calf_joint': 0.,
    # 'RR_calf_joint': 0.,
    # }

    joint_names = [
    'FL_hip_joint',
    'RL_hip_joint',
    'FR_hip_joint',
    'RR_hip_joint',
    'FL_thigh_joint',
    'RL_thigh_joint',
    'FR_thigh_joint',
    'RR_thigh_joint',
    'FL_calf_joint',
    'RL_calf_joint',
    'FR_calf_joint',
    'RR_calf_joint',
    ]

    default_dof_pos = np.array([default_joint_angles[name] for name in joint_names])

    dof_targets = np.zeros(default_dof_pos.shape, dtype=np.float32)
    dof_stiffness = 150.
    dof_damping = 3.
####################################################

    num_eval_steps = 1
    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5]}

    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
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
    viewer = mujoco.viewer.launch_passive(model, data)

    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        # print(f"Actuator {i}: {name}")


    for i in tqdm(range(num_eval_steps)):

        dof_pos = data.qpos.astype(np.float32)[7:]
        dof_vel = data.qvel.astype(np.float32)[6:]

        with torch.no_grad():
            actions = policy(obs)
            # 策略输出的是关节位置的偏移量，这些偏移量被加到默认关节角度上形成目标关节位置，
            # 然后通过PD控制器转换为实际的力矩命令发送给机器人执行器
            actions_np = actions.cpu().numpy().squeeze()  # shape: (12,)
            for i, name in enumerate(joint_names):
                print(f"{name}: {actions_np[i]:.4f}")
            action_dict = dict(zip(joint_names, actions_np))

        # 正确排序后传给 data.ctrl
        mujoco_joint_order = [
            'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
            'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
            'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
            'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'
        ]

        # 构造 default_dof_pos 和 dof_targets 的正确顺序
        default_dof_pos = np.array([default_joint_angles[j] for j in mujoco_joint_order])
        # print(default_dof_pos)   [-0.25 -0.45  1.4   0.25 -0.45  1.4  -0.25 -0.9   1.3   0.25 -0.9   1.3 ]
        dof_targets = np.array([default_joint_angles[j] + action_dict[j] for j in mujoco_joint_order])

        # 获取当前状态
        dof_pos = np.array([data.qpos[model.joint(name).qposadr] for name in mujoco_joint_order])
        dof_vel = np.array([data.qvel[model.joint(name).dofadr] for name in mujoco_joint_order])


        # PD 控制
        data.ctrl[:12] = np.clip(
            dof_stiffness * (dof_targets - dof_pos) - dof_damping * dof_vel,
            model.actuator_ctrlrange[:, 0],
            model.actuator_ctrlrange[:, 1]
        )
        print(data.ctrl[:12])
        # data.ctrl[:12] = actions

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

        for _ in range(5):
            mujoco.mj_step(model, data)


        viewer.sync()

if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    # play_go1(headless=True)
    play_go1(headless=True)
