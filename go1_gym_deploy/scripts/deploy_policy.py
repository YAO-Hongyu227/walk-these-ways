import glob
import pickle as pkl
import lcm
import sys

from go1_gym_deploy.utils.deployment_runner import DeploymentRunner
from go1_gym_deploy.envs.lcm_agent import LCMAgent
from go1_gym_deploy.utils.cheetah_state_estimator import StateEstimator
from go1_gym_deploy.utils.command_profile import *

import pathlib

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

def load_and_run_policy(label, experiment_name, max_vel=1.0, max_yaw_vel=1.0):
    # 加载训练日志目录
    dirs = glob.glob(f"../../runs/{label}/*")
    logdir = sorted(dirs)[0]

    # 读取训练时保存的参数配置
    with open(logdir+"/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())

    # 初始化状态估计器，用于订阅IMU、足部接触等传感器信息
    se = StateEstimator(lc)

    # 设置遥控器命令的时间步长和比例因子
    control_dt = 0.02
    command_profile = RCControllerProfile(dt=control_dt, state_estimator=se, x_scale=max_vel, y_scale=0.6, yaw_scale=max_yaw_vel)

    # 创建控制代理，用于从遥控器读取命令，并通过LCM控制机器人
    hardware_agent = LCMAgent(cfg, se, command_profile)
    se.spin()  # 启动状态估计器，开始监听LCM订阅信息

    # 使用历史观测包装agent，增强策略的输入（例如：三帧IMU数据）
    from go1_gym_deploy.envs.history_wrapper import HistoryWrapper
    hardware_agent = HistoryWrapper(hardware_agent)

    # 加载训练好的策略（TorchScript模型）
    policy = load_policy(logdir)

    # 构建日志保存路径
    root = f"{pathlib.Path(__file__).parent.resolve()}/../../logs/"
    pathlib.Path(root).mkdir(parents=True, exist_ok=True)

    # 创建部署运行器，用于管理agent、policy、遥控器、日志等
    deployment_runner = DeploymentRunner(experiment_name=experiment_name, se=None,
                                         log_root=f"{root}/{experiment_name}")

    # 注册agent、策略、遥控器配置
    deployment_runner.add_control_agent(hardware_agent, "hardware_closed_loop")
    deployment_runner.add_policy(policy)
    deployment_runner.add_command_profile(command_profile)

    # 支持从命令行指定运行的最大步数，默认10000000
    if len(sys.argv) >= 2:
        max_steps = int(sys.argv[1])
    else:
        max_steps = 10000000
    print(f'max steps {max_steps}')

    # 开始执行部署策略（进入主循环）
    deployment_runner.run(max_steps=max_steps, logging=True)


def load_policy(logdir):
    # 加载策略网络的两个部分：body 和 adaptation module
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')  # 主策略网络
    import os
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')  # 适应模块

    # 返回一个可调用的策略函数 policy(obs, info)
    def policy(obs, info):
        i = 0  # 没用到，可以删
        # 使用适应模块，根据观测历史 obs_history 生成一个 latent 向量
        # 它是一个 隐藏状态（latent state），由 adaptation_module 网络根据 obs["obs_history"] 推理出来，包含了当前环境的动态特征，例如：
            # 地形（平地、坡地、不平整地）
            # 摩擦系数（滑不滑）
            # 地面弹性/硬度
            # 接触模式的变化
            # 甚至是传感器延迟等

        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))

        # 把观测历史和 latent 向量拼接，送入主策略网络 body 得到动作输出
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))

        # 把 latent 存进 info 中，便于调试或分析
        info['latent'] = latent
        return action  # 返回动作

    return policy  # 返回封装好的策略函数



if __name__ == '__main__':
    label = "gait-conditioned-agility/pretrain-v0/train"

    experiment_name = "example_experiment"

    load_and_run_policy(label, experiment_name=experiment_name, max_vel=3.5, max_yaw_vel=5.0)
