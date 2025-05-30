# 整合修复 observation 拼接顺序 + 动作历史处理的 Mujoco 脚本
import mujoco
import mujoco.viewer
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque
import os
import time

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


def get_observation(data, history_buffer, commands, last_action, prev_action, obs_scales, gait, t, device='cpu'):
    # 1. 模拟 projected_gravity（默认重力向下）
    
    quat = data.qpos[3:7]
    projected_gravity = quat_rotate_inverse(quat, np.array([0.0, 0.0, 1.0]))    #这里之前是[0.0, 0.0, -1.0]，但是和wtw的输出对不上
    # projected_gravity[1] *= -1

    # 2. 读取状态
    mujoco_dof_pos = data.qpos[7:]           # shape: (12,)
    mujoco_dof_vel = data.qvel[6:18]         # shape: (12,)

    #这里把mujoco读到的电机角度转成wtw的
    wtw_dof_pos = np.zeros(12)
    wtw_dof_vel = np.zeros(12)
    # mujoco --> wtw
    # wtw_dof_pos = mujoco_dof_pos[rearranged_index]
    # wtw_dof_vel = mujoco_dof_vel[rearranged_index]

    wtw_dof_pos = mujoco_dof_pos
    wtw_dof_vel = mujoco_dof_vel
    # 传给observation的joint_angle和joint_velocity的关节顺序为：

    # 0-2: FL_hip_joint, FL_thigh_joint, FL_calf_joint (前左腿)
    # 3-5: FR_hip_joint, FR_thigh_joint, FR_calf_joint (前右腿)
    # 6-8: RL_hip_joint, RL_thigh_joint, RL_calf_joint (后左腿)
    # 9-11: RR_hip_joint, RR_thigh_joint, RR_calf_joint (后右腿)

    # print(wtw_dof_pos)
    # print(wtw_default_dof_pos)
    # print('')

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
        (wtw_dof_pos - default_dof_pos) * obs_scales["dof_pos"],        # 12
        wtw_dof_vel * obs_scales["dof_vel"],                            # 12
        last_action,                                                # 12
        prev_action,                                                # 12
        clock_inputs                                                # 4
    ])

    obs = np.clip(obs,-100.0,100.0)

    # 5. 维护 obs_history（2100维 = 30×70）
    #history_buffer 是一个 deque，每次保存最近的 30 帧观测（即30个 obs 向量）。
    if len(history_buffer) == 0:
        for _ in range(30):
            history_buffer.append(obs.copy())  # 替代np.zeros填充
    else:
        history_buffer.append(obs.copy())
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

def update_camera(viewer, data, offset=np.array([-2.0, 0.0, 1.0])):
    """
    offset: 相机相对机器人坐标的位置（单位：米），比如[-2, 0, 1]表示从后上方看。
    """
    # 获取位置与四元数
    pos = data.qpos[:3]
    quat = data.qpos[3:7]  # [x, y, z, w]
    
    # 转换四元数为旋转矩阵
    mat = np.zeros((3, 3))
    mujoco.mju_quat2Mat(mat.ravel(), quat)

    # 获取相机位置（在机器人坐标系偏移后转换为世界系）
    cam_pos = pos + mat @ offset
    viewer.cam.lookat[:] = pos
    viewer.cam.distance = np.linalg.norm(cam_pos - pos)

    # 让 azimuth 和 elevation 固定值（可选，不建议频繁更新）
    # viewer.cam.azimuth = 0
    # viewer.cam.elevation = -30


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


    #mujoco中的关节顺序
    global default_dof_pos
    waiba = True
    waiba = False
    if waiba:
        default_dof_pos = np.array([
        -0.25,  -0.45,  1.4,     # FL: hip, thigh, calf
        0.25,   -0.45,  1.4,     # FR: hip, thigh, calf
        -0.25,  -0.9,   1.3,     # RL: hip, thigh, calf
        0.25,   -0.9,   1.3      # RR: hip, thigh, calf
        ])
    else:
        default_dof_pos = np.array([
        -0.,  -0.8,  1.3,     # FL: hip, thigh, calf
        0.,   -0.8,  1.3,     # FR: hip, thigh, calf
        -0.,  -1.,   1.3,     # RL: hip, thigh, calf
        0.,   -1.,   1.3      # RR: hip, thigh, calf
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

    for name in ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        model.geom_friction[gid] = [0.9, 0.005, 0.0001]

    dof_targets = np.zeros(default_dof_pos.shape, dtype=np.float32)

    #这一版能走，但是是外八 030701.468701
    if waiba:
        logdir = "/home/tong/Downloads/walk-these-ways/runs/gait-conditioned-agility/2025-05-30/train/030701.468701"
    else:
        logdir = "/home/tong/Downloads/walk-these-ways/runs/gait-conditioned-agility/2025-05-30/train/034438.215278"
    policy = load_policy(logdir)


    num_eval_steps = 2
    num_eval_steps = 20000
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

    clip_actions = 10.

    obs_scales = {
        "gravity": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 0.05
    }

    # model.opt.gravity[:] = 0  # 去掉重力


    prev_action = np.zeros(12)
    last_action = np.zeros(12)
    history_buffer = deque(maxlen=30)

    ctrl_range = model.actuator_ctrlrange[:12]
    low, high = ctrl_range[:, 0], ctrl_range[:, 1]

    viewer = mujoco.viewer.launch_passive(model, data)
    

    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        qpos_index = model.jnt_qposadr[i]
        print(f"{joint_name}: data.qpos[{qpos_index}] = {data.qpos[qpos_index]}")

    save_obs = True
    save_obs = False
    saved_obs_list = []


    #从wtw读取actions
    debug_act = True  # <--- 是否从外部动作文件读取
    debug_act = False
    debug_action_path = "logs/obs_go1.txt"  # <--- 替换为你的 action txt 路径
    if debug_act:
        debug_actions = np.loadtxt(debug_action_path)[:, 42:54]  # 提取每行的 action 部分（第 42-54 维）
        assert debug_actions.shape[1] == 12
    
    #从wtw读取torques
    debug_ctrl = False
    debug_ctrl_path = "logs/logged_actions_scaled.txt"  # 每一行是一个 12 维 data.ctrl[:12]
    if debug_ctrl:
        debug_ctrls = np.loadtxt(debug_ctrl_path).reshape(-1, 12)
        assert debug_ctrls.shape[1] == 12

    action_cnt = 0

    model.opt.timestep = 0.001          #对应wtw中的dt
    frame_skip = 20                      #对应wtw中的decimation
    sim_time_log = []

    saved_actions = []


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
        # time.sleep(0.001)
        if i % frame_skip == 0:
            sim_time_log.append(data.time)
            ######################################################## OBSERVATION #####################################################
            obs_dict = get_observation(data, history_buffer, commands, last_action, prev_action, obs_scales, gait, t)
            
            if save_obs:
                obs_vec = obs_dict['obs'].squeeze().cpu().numpy()
                saved_obs_list.append(obs_vec.copy())

            #这里可以把单个的observation打印出来
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

            ############################################################### ACTION ###########################################################
            # 获得action
            
            if debug_act:
                action = debug_actions[action_cnt]
                action_cnt = action_cnt + 1
            else:
                with torch.no_grad():
                    action = policy(obs_dict).squeeze().cpu().float().numpy()
                    action = np.clip(action, -clip_actions, clip_actions)

            saved_actions.append(action.copy())

            #wtw中action的组成,发现是和mojoco中一样的,不过就是hip都需要加个负号才能正常走
                # [FL_hip, FL_thigh, FL_calf,
                # FR_hip, FR_thigh, FR_calf,
                # RL_hip, RL_thigh, RL_calf,
                # RR_hip, RR_thigh, RR_calf]
                    

        #这里可以检查站立情况
        # action = np.zeros(12)
        ##################################################################################################################################

        dof_pos = data.qpos[7:]
        dof_vel = data.qvel[6:18]

        dof_stiffness  = 150.
        dof_damping = 3.

        #这里因为是wtw的policy输出的action，所以关节顺序和mujoco的xml关节顺序不一样，做一个映射
        # wtw --> mujoco

        
        #这是mujoco中的关节顺序.需要改正负号来让wtw中的action正负匹配
        correct_dof_pos = np.array([
        1.,  1.,  1.,     # FL: hip, thigh, calf
        1.,   1.,  1.,     # FR: hip, thigh, calf
        1.,  1.,   1.,     # RL: hip, thigh, calf
        1.,   1.,   1.      # RR: hip, thigh, calf
        ])
        rearranged_action = action * correct_dof_pos

        #乘一个action_scale
        action_scale = 0.25
        scaled_rearranged_action = rearranged_action * action_scale         

        # saved_actions.append(scaled_rearranged_action.copy())

        #计算target_position
        target_pos = default_dof_pos + scaled_rearranged_action

        #data.ctrl的顺序
        # floating_base_joint: data.qpos[0] = 0.0
        # FL_hip_joint: data.qpos[7] = -0.1
        # FL_thigh_joint: data.qpos[8] = -0.8
        # FL_calf_joint: data.qpos[9] = 1.5

        # FR_hip_joint: data.qpos[10] = 0.1
        # FR_thigh_joint: data.qpos[11] = -0.8
        # FR_calf_joint: data.qpos[12] = 1.5

        # RL_hip_joint: data.qpos[13] = -0.1
        # RL_thigh_joint: data.qpos[14] = -0.8
        # RL_calf_joint: data.qpos[15] = 1.5

        # RR_hip_joint: data.qpos[16] = 0.1
        # RR_thigh_joint: data.qpos[17] = -0.8
        # RR_calf_joint: data.qpos[18] = 1.5
        if debug_ctrl:
            data.ctrl[:12] = debug_ctrls[action_cnt]
            print(data.ctrl[:12])
            saved_actions.append(data.ctrl[:12].copy())
        else:
            data.ctrl[:12] = np.clip(
                dof_stiffness * (target_pos - dof_pos) - dof_damping * dof_vel,
                model.actuator_ctrlrange[:, 0],
                model.actuator_ctrlrange[:, 1]
            )
            saved_actions.append(target_pos.copy())


        prev_action = last_action
        last_action = action


        mujoco.mj_step(model, data)
        # viewer.cam.lookat[:] = data.qpos.astype(np.float32)[0:3]
        update_camera(viewer, data)

        viewer.sync()
    
    if save_obs:
        os.makedirs("logs", exist_ok=True)
        obs_array = np.stack(saved_obs_list, axis=0)  # shape: (steps, 70)
        np.savetxt("logs/obs_sequence.txt", obs_array, fmt="%.6f")
        print(f"Saved {obs_array.shape[0]} obs to logs/obs_sequence.txt")
    
    saved_actions = np.array(saved_actions)  # shape: [num_steps, 12]

    plot_joint = True
    if plot_joint:
        os.makedirs("logs/joint_grouped", exist_ok=True)

        joint_groups = {
            "Hip": [0, 3, 6, 9],     # FL_hip, FR_hip, RL_hip, RR_hip
            "Thigh": [1, 4, 7, 10],  # FL_thigh, FR_thigh, RL_thigh, RR_thigh
            "Calf": [2, 5, 8, 11],   # FL_calf, FR_calf, RL_calf, RR_calf
        }

        joint_labels = {
            "Hip": ["FL_hip", "FR_hip", "RL_hip", "RR_hip"],
            "Thigh": ["FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"],
            "Calf": ["FL_calf", "FR_calf", "RL_calf", "RR_calf"],
        }

        for group_name, indices in joint_groups.items():
            fig, axs = plt.subplots(len(indices), 1, figsize=(10, 8), sharex=True)
            fig.suptitle(f"{group_name} Joints Over Time", fontsize=14)
            
            for i, (idx, label) in enumerate(zip(indices, joint_labels[group_name])):
                axs[i].plot(saved_actions[:, idx])
                axs[i].set_ylabel(label)
                axs[i].grid(True)
                axs[i].set_ylim(-2, 2)  # 固定 y 轴范围

            axs[-1].set_xlabel("Control Step")
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            save_path = f"logs/joint_grouped/{group_name.lower()}_joints.png"
            plt.savefig(save_path)
            # plt.show()
        from PIL import Image

        # 横向拼接所有 joint_grouped 图像
        group_names = ["hip", "thigh", "calf"]
        image_paths = [f"logs/joint_grouped/{name}_joints.png" for name in group_names]
        images = [Image.open(path) for path in image_paths]

        # 计算拼接尺寸
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)

        # 创建拼接图
        combined_image = Image.new("RGB", (total_width, max_height), color=(255, 255, 255))

        # 粘贴各图
        x_offset = 0
        for img in images:
            combined_image.paste(img, (x_offset, 0))
            x_offset += img.width

        # 保存和展示
        combined_path = "logs/joint_grouped/all_joint_groups.png"
        combined_image.save(combined_path)
        combined_image.show()






    # import matplotlib.pyplot as plt
    # time_diffs = np.diff(sim_time_log)
    # plt.plot(time_diffs)
    # plt.title("Control Interval (s)")
    # plt.xlabel("Control Step")
    # plt.ylabel("Δt")
    # plt.ylim(0.0, 0.04)  # 固定 Y 轴范围为 0.019 到 0.021
    # plt.grid(True)
    # plt.show()

    viewer.close()





play_mujoco()