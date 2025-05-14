import copy
import time
import os

import numpy as np
import torch

from go1_gym_deploy.utils.logger import MultiLogger

class DeploymentRunner:
    def __init__(self, experiment_name="unnamed", se=None, log_root="."):
        self.agents = {}  # 存储所有机器人 agent
        self.policy = None  # 控制策略函数
        self.command_profile = None  # 用于获取手柄指令
        self.logger = MultiLogger()  # 日志记录器
        self.se = se  # 状态估计器
        self.vision_server = None  # 视觉服务器（可选）

        self.log_root = log_root  # 日志文件保存路径
        self.init_log_filename()  # 初始化日志路径
        self.control_agent_name = None  # 控制 agent 名
        self.command_agent_name = None  # 命令 agent 名

        self.triggered_commands = {i: None for i in range(4)}  # 控制器上 4 个按钮触发的指令配置
        self.button_states = np.zeros(4)  # 当前按钮状态

        self.is_currently_probing = False  # 是否正在采集数据
        self.is_currently_logging = [False, False, False, False]  # 各按钮是否在记录日志

    def init_log_filename(self):
        # 创建日志目录，文件命名为当前时间戳
        datetime = time.strftime("%Y/%m_%d/%H_%M_%S")
        for i in range(100):
            try:
                os.makedirs(f"{self.log_root}/{datetime}_{i}")
                self.log_filename = f"{self.log_root}/{datetime}_{i}/log.pkl"
                return
            except FileExistsError:
                continue

    def add_open_loop_agent(self, agent, name):
        self.agents[name] = agent
        self.logger.add_robot(name, agent.env.cfg)

    def add_control_agent(self, agent, name):
        self.control_agent_name = name
        self.agents[name] = agent
        self.logger.add_robot(name, agent.env.cfg)

    def add_vision_server(self, vision_server):
        self.vision_server = vision_server

    def set_command_agents(self, name):
        self.command_agent = name

    def add_policy(self, policy):
        self.policy = policy

    def add_command_profile(self, command_profile):
        self.command_profile = command_profile

    def calibrate(self, wait=True, low=False):
        # 将机器人缓慢移动到标准初始姿态
        for agent_name in self.agents.keys():
            if hasattr(self.agents[agent_name], "get_obs"):         #判断当前agent是否有get_obs这个方法，以保证接下来调用agent.get_obs()不会出错
                agent = self.agents[agent_name]
                agent.get_obs()
                joint_pos = agent.dof_pos
                if low:
                    final_goal = np.array([0., 0.3, -0.7] * 4)  # 矮姿态
                else:
                    final_goal = np.zeros(12)  # 标准初始姿态
                nominal_joint_pos = agent.default_dof_pos       #记录标准的关节角度

                print(f"About to calibrate; the robot will stand [Press R2 to calibrate]")
                while wait:
                    self.button_states = self.command_profile.get_buttons()
                    if self.command_profile.state_estimator.right_lower_right_switch_pressed:
                        self.command_profile.state_estimator.right_lower_right_switch_pressed = False
                        break

                # 构造目标关节动作序列，逐步插值逼近目标姿态
                cal_action = np.zeros((agent.num_envs, agent.num_actions))
                target_sequence = []
                target = joint_pos - nominal_joint_pos
                while np.max(np.abs(target - final_goal)) > 0.01:
                    target -= np.clip((target - final_goal), -0.05, 0.05)
                    target_sequence += [copy.deepcopy(target)]

                # 执行动作序列，每次更新一步
                for target in target_sequence:
                    next_target = target

                    # 判断配置是字典还是对象，并提取控制参数：
                    # hip_scale_reduction：髋关节的缩放系数（减少其动作幅度）
                    # action_scale：动作缩放因子（调整动作在控制信号中的幅度）
                    if isinstance(agent.cfg, dict):
                        hip_reduction = agent.cfg["control"]["hip_scale_reduction"]
                        action_scale = agent.cfg["control"]["action_scale"]
                    else:
                        hip_reduction = agent.cfg.control.hip_scale_reduction
                        action_scale = agent.cfg.control.action_scale

                    next_target[[0, 3, 6, 9]] /= hip_reduction          #只对 4 个髋关节（第 0、3、6、9 个 DOF）应用缩放因子。这是为了调整髋关节动作的敏感性或力矩输出。
                    next_target = next_target / action_scale            # 所有关节动作统一除以 action_scale，将物理空间动作映射为 归一化的控制空间动作。
                    cal_action[:, 0:12] = next_target
                    agent.step(torch.from_numpy(cal_action))            # 将动作指令转换为 PyTorch 张量，传递给仿真或真实机器人控制接口执行动作。
                    agent.get_obs()
                    time.sleep(0.05)

                print("Starting pose calibrated [Press R2 to start controller]")
                while True:
                    self.button_states = self.command_profile.get_buttons()
                    if self.command_profile.state_estimator.right_lower_right_switch_pressed:
                        self.command_profile.state_estimator.right_lower_right_switch_pressed = False
                        break

                # 所有 agent 重置
                for agent_name in self.agents.keys():
                    obs = self.agents[agent_name].reset()
                    if agent_name == self.control_agent_name:
                        control_obs = obs

        return control_obs

    def run(self, num_log_steps=1000000000, max_steps=100000000, logging=True):
        # 部署前确保控制 agent、策略、手柄都准备好
        assert self.control_agent_name is not None, "cannot deploy, runner has no control agent!"
        assert self.policy is not None, "cannot deploy, runner has no policy!"
        assert self.command_profile is not None, "cannot deploy, runner has no command profile!"

        # 所有 agent 初始化
        for agent_name in self.agents.keys():
            obs = self.agents[agent_name].reset()
            if agent_name == self.control_agent_name:
                control_obs = obs

        # 初始校准
        control_obs = self.calibrate(wait=True)

        try:
            for i in range(max_steps):
                policy_info = {}
                action = self.policy(control_obs, policy_info)

                # 所有 agent 执行动作并记录信息
                for agent_name in self.agents.keys():
                    obs, ret, done, info = self.agents[agent_name].step(action)
                    info.update(policy_info)
                    info.update({
                        "observation": obs, "reward": ret, "done": done, "timestep": i,
                        "time": i * self.agents[self.control_agent_name].dt,
                        "action": action,
                        "rpy": self.agents[self.control_agent_name].se.get_rpy(),
                        "torques": self.agents[self.control_agent_name].torques
                    })
                    if logging:
                        self.logger.log(agent_name, info)

                    if agent_name == self.control_agent_name:
                        control_obs = obs

                # 若姿态角超过阈值，触发紧急校准
                rpy = self.agents[self.control_agent_name].se.get_rpy()
                if abs(rpy[0]) > 1.6 or abs(rpy[1]) > 1.6:
                    self.calibrate(wait=False, low=True)

                # 检查是否触发日志记录
                prev_button_states = self.button_states[:]
                self.button_states = self.command_profile.get_buttons()

                # 左下按钮控制数据探测和保存
                if self.command_profile.state_estimator.left_lower_left_switch_pressed:
                    if not self.is_currently_probing:
                        print("START LOGGING")
                        self.is_currently_probing = True
                        self.agents[self.control_agent_name].set_probing(True)
                        self.init_log_filename()
                        self.logger.reset()
                    else:
                        print("SAVE LOG")
                        self.is_currently_probing = False
                        self.agents[self.control_agent_name].set_probing(False)
                        control_obs = self.calibrate(wait=False)
                        self.logger.save(self.log_filename)
                        self.init_log_filename()
                        self.logger.reset()
                        time.sleep(1)
                        control_obs = self.agents[self.control_agent_name].reset()
                    self.command_profile.state_estimator.left_lower_left_switch_pressed = False

                # 4 个按钮的日志触发逻辑
                for button in range(4):
                    if self.command_profile.currently_triggered[button]:
                        if not self.is_currently_logging[button]:
                            print("START LOGGING")
                            self.is_currently_logging[button] = True
                            self.init_log_filename()
                            self.logger.reset()
                    else:
                        if self.is_currently_logging[button]:
                            print("SAVE LOG")
                            self.is_currently_logging[button] = False
                            control_obs = self.calibrate(wait=False)
                            self.logger.save(self.log_filename)
                            self.init_log_filename()
                            self.logger.reset()
                            time.sleep(1)
                            control_obs = self.agents[self.control_agent_name].reset()

                # R2 强制重新校准流程
                if self.command_profile.state_estimator.right_lower_right_switch_pressed:
                    control_obs = self.calibrate(wait=False)
                    time.sleep(1)
                    self.command_profile.state_estimator.right_lower_right_switch_pressed = False
                    while not self.command_profile.state_estimator.right_lower_right_switch_pressed:
                        time.sleep(0.01)
                    self.command_profile.state_estimator.right_lower_right_switch_pressed = False

            # 所有运行完成后最后一次保存日志
            control_obs = self.calibrate(wait=False)
            self.logger.save(self.log_filename)

        except KeyboardInterrupt:
            # 中断时保存日志
            self.logger.save(self.log_filename)
