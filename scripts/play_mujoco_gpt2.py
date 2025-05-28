# 整合修复 observation 拼接顺序 + 动作历史处理的 Mujoco 脚本
import mujoco
import mujoco.viewer
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque

def load_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

    def policy(obs, info={}):
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action
    return policy

from collections import deque
import torch
import numpy as np


def quat_rotate_inverse(q, v):
    # q: [x, y, z, w]
    q_w = q[3]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c


def get_observation(data, history_buffer, commands, last_action, prev_action, obs_scales, gait, t, device="cpu"):
    # 1. 模拟 projected_gravity（默认重力向下）
    
    quat = data.qpos[3:7]
    projected_gravity = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))

    # 2. 读取状态
    mujoco_dof_pos = data.qpos[7:]           # shape: (12,)
    mujoco_dof_vel = data.qvel[6:18]         # shape: (12,)

    #这里把mujoco读到的电机角度转成wtw的
    wtw_dof_pos = np.zeros(12)
    wtw_dof_vel = np.zeros(12)
    # mujoco --> wtw
    for i in range(12):
        wtw_dof_pos[i] = mujoco_dof_pos[rearranged_index[i]]
        wtw_dof_vel[i] = mujoco_dof_vel[rearranged_index[i]]
    
    print(wtw_dof_pos)
    print(wtw_default_dof_pos)
    print('')

    # 3. 生成 clock_inputs
    t_FR = t + gait[2] + gait[3]
    t_FL = t + gait[1] + gait[3]
    t_RR = t + gait[1]
    t_RL = t + gait[2]

    clock_inputs = np.array([
        np.sin(2 * np.pi * t_FR),
        np.sin(2 * np.pi * t_FL),
        np.sin(2 * np.pi * t_RR),
        np.sin(2 * np.pi * t_RL)
    ])

    # 4. 拼接 obs (70维)
    obs = np.concatenate([
        projected_gravity * obs_scales["gravity"],                  # 3
        commands * command_scales,                                  # 15
        (wtw_dof_pos - wtw_default_dof_pos) * obs_scales["dof_pos"],        # 12
        wtw_dof_vel * obs_scales["dof_vel"],                            # 12
        last_action,                                                # 12
        prev_action,                                                # 12
        clock_inputs                                                # 4
    ])

    # 5. 维护 obs_history（2100维 = 30×70）
    history_buffer.append(obs)      #history_buffer 是一个 deque，每次保存最近的 30 帧观测（即30个 obs 向量）。
    while len(history_buffer) < 30:
        history_buffer.appendleft(np.zeros_like(obs))       #长度小于30时用0填充
    obs_history = np.concatenate(list(history_buffer), axis=0)

    # 6. 构造 obs dict
    obs_dict = {
        "obs": torch.tensor(obs).unsqueeze(0).float().to(device),
        "obs_history": torch.tensor(obs_history).unsqueeze(0).float().to(device),
        "privileged_obs": torch.tensor([[1.0, -1.0]], dtype=torch.float32).to(device)
    }
    return obs_dict

def print_obs_structure(obs_dict, max_values=5):
    for key, value in obs_dict.items():
        value_np = value.detach().cpu().numpy()
        shape = value_np.shape
        print(f"\nKey: '{key}' | Shape: {shape}")
        print("Values:", value_np.flatten()[:max_values], "...")


command_scales = np.array([
    2.0,    # x vel
    2.0,    # y vel
    0.25,   # yaw vel
    2.0,    # body height cmd
    1.0,    # gait freq cmd
    1.0,    # gait phase FR
    1.0,    # gait phase FL
    1.0,    # gait phase RR
    1.0,    # gait phase RL
    0.15,   # footswing height
    0.3,    # pitch
    0.3,    # roll
    1.0,    # stance width
    1.0,    # stance length
    1.0     # aux reward
    ])

rearranged_index = [0,6,3,9,1,7,4,10,2,8,5,11]

def play_mujoco():
    xml_path = "/home/tong/Downloads/walk-these-ways/resources/robots/Q20/scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    global default_dof_pos
    default_dof_pos = np.array([
    -0.25,  -0.45,  1.4,     # FL: hip, thigh, calf
    0.25,   -0.45,  1.4,     # FR: hip, thigh, calf
    -0.25,  -0.9,   1.3,     # RL: hip, thigh, calf
    0.25,   -0.9,   1.3      # RR: hip, thigh, calf
    ])

    global wtw_default_dof_pos
    wtw_default_dof_pos = np.array([
        -0.25,  # FL_hip_joint
        -0.25,  # RL_hip_joint
        0.25,  # FR_hip_joint
        0.25,  # RR_hip_joint

        -0.45,  # FL_thigh_joint
        -0.9,   # RL_thigh_joint
        -0.45,  # FR_thigh_joint
        -0.9,   # RR_thigh_joint

        1.4,   # FL_calf_joint
        1.3,   # RL_calf_joint
        1.4,   # FR_calf_joint
        1.3    # RR_calf_joint
    ], dtype=np.float32)

    # default_dof_pos = np.array([0.1] * 12)
    mujoco.mj_forward(model, data)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    dof_targets = np.zeros(default_dof_pos.shape, dtype=np.float32)

    logdir = "/home/tong/Downloads/walk-these-ways/runs/gait-conditioned-agility/2025-05-20/train/091602.705910"
    policy = load_policy(logdir)

    num_eval_steps = 2
    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 1.5, 0.0, 0.0
    # x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
    body_height_cmd = 0.0
    step_frequency_cmd = 3.0
    gait = np.array([0.0,0.5, 0.0, 0.0])  # trotting , gait[0]是没用的，用来占位  -- 为了和论文对应
    footswing_height_cmd = 0.08 * 1.5
    pitch_cmd, roll_cmd = 0.0, 0.0
    stance_width_cmd = 0.25 * 1.5
    stance_length_cmd = 0.6585
    aux_reward_cmd = 0.0002138


    obs_scales = {
        "gravity": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 0.05
    }

    model.opt.gravity[:] = 0  # 去掉重力


    measured_x_ = np.zeros(num_eval_steps)
    joint_positionvelss = np.zeros((num_eval_steps, 12))
    prev_action = np.zeros(12)
    last_action = np.zeros(12)
    history_buffer = deque(maxlen=30)

    ctrl_range = model.actuator_ctrlrange[:12]
    low, high = ctrl_range[:, 0], ctrl_range[:, 1]

    viewer = mujoco.viewer.launch_passive(model, data)
    

    # for i in range(model.njnt):
    #     joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    #     qpos_index = model.jnt_qposadr[i]
    #     print(f"{joint_name}: data.qpos[{qpos_index}] = {data.qpos[qpos_index]}")


    for i in tqdm(range(num_eval_steps)):
        commands = np.zeros(15)
        commands[0] = x_vel_cmd
        commands[1] = y_vel_cmd
        commands[2] = yaw_vel_cmd
        commands[3] = body_height_cmd
        commands[4] = step_frequency_cmd
        commands[5:8] = gait[1:]
        commands[8] = 0.5
        commands[9] = footswing_height_cmd
        commands[10] = pitch_cmd
        commands[11] = roll_cmd
        commands[12] = stance_width_cmd
        commands[13] = stance_length_cmd
        commands[14] = aux_reward_cmd

        t = (data.time * step_frequency_cmd) % 1.0
        obs_dict = get_observation(data, history_buffer, commands, last_action, prev_action, obs_scales, gait, t)
        
        # 示例调用
        # print_obs_structure(obs_dict)
        # obs_history = obs_dict["obs_history"]
        # print('obs_history')
        # print(len(obs_history[0]))

        inspect_obs = True
        inspect_obs = False
        if inspect_obs:
            print('************************************************************')
            obs_vec = obs_dict['obs'].squeeze().cpu().numpy()
            print("projected_gravity:", obs_vec[0:3])
            print("commands:", obs_vec[3:18])
            print("dof_pos:", obs_vec[18:30])
            print("dof_vel:", obs_vec[30:42])
            print("actions:", obs_vec[42:54])
            print("last_actions:", obs_vec[54:66])
            print("clock_inputs:", obs_vec[66:70])
            print('')

        with torch.no_grad():
            action = policy(obs_dict).squeeze(0).cpu().numpy()
            action = np.clip(action, low, high)
            #action顺序，也是go1的默认电机顺序
            # FR_hip, FR_thigh, FR_calf, 
            # FL_hip, FL_thigh, FL_calf, 
            # RR_hip, RR_thigh, RR_calf, 
            # RL_hip, RL_thigh, RL_calf

        # action = np.zeros(12)

        dof_pos = data.qpos[7:]
        # print('mujoco dof_joints:')
        # print(dof_pos)

        dof_vel = data.qvel[6:18]
        # print('mujoco dof_vel:')
        # print(dof_vel)

        dof_stiffness  = 150.
        dof_damping = 3.

        #这里因为是wtw的policy输出的action，所以关节顺序和mujoco的xml关节顺序不一样，做一个映射
        # wtw --> mujoco
        rearranged_action = np.zeros(12)
        for i in range(12):
            rearranged_action[rearranged_index[i]] = action[i]
        
        action_scale = 0.25
        scaled_rearranged_action = rearranged_action * action_scale

        target_pos = default_dof_pos + scaled_rearranged_action
        # print('target_pos:')
        # print(target_pos)
        
        data.ctrl[:12] = np.clip(
            dof_stiffness * (target_pos - dof_pos) - dof_damping * dof_vel,
            model.actuator_ctrlrange[:, 0],
            model.actuator_ctrlrange[:, 1]
        )
        # data.ctrl[:12] = action
        prev_action = last_action
        last_action = action

        mujoco.mj_step(model, data)

        viewer.sync()

play_mujoco()
