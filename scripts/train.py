
def train_go1(headless=True):

    import isaacgym
    assert isaacgym
    import torch

    from go1_gym.envs.base.legged_robot_config import Cfg
    from go1_gym.envs.go1.go1_config import config_go1
    from go1_gym.envs.go1.q20_config import config_q20                    #revised by hongyu, 05/14/2025
    from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv

    from ml_logger import logger

    from go1_gym_learn.ppo_cse import Runner
    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper
    from go1_gym_learn.ppo_cse.actor_critic import AC_Args
    from go1_gym_learn.ppo_cse.ppo import PPO_Args
    from go1_gym_learn.ppo_cse import RunnerArgs
    from torch.utils.tensorboard import SummaryWriter               #revised by hongyu, 05/15/2025, adding tensorboard


    ####################################### #revised by hongyu, 05/15/2025, adding tensorboard ##################
    #创建TensorBoard Writer
    from pathlib import Path
    from go1_gym import MINI_GYM_ROOT_DIR
    import time

    stem = Path(__file__).stem                    #获取当前脚本文件名称，用于命名日志子目录
    #log_dir 构建了层级结构的日志目录，包含日期、脚本名称和时间戳，确保每次运行的日志文件不会互相覆盖
    log_dir = Path(f"{MINI_GYM_ROOT_DIR}/runs/gait-conditioned-agility/{time.strftime('%Y-%m-%d')}/{stem}/{time.strftime('%H%M%S.%f')}")
    log_dir.mkdir(parents=True, exist_ok=True)    #创建上述路径，parents=True表示如果上级目录不存在则一并创建，exist_ok表示如果目录已经存在则不报错

    #创建一个SummaryWriter的实例
    writer = SummaryWriter(log_dir=str(log_dir))


    ##############################################################################################################

    config_go1(Cfg) 

    # config_q20(Cfg)       #revised by hongyu, 05/14/2025

    Cfg.commands.num_lin_vel_bins = 30                                      #线性速度命令的离散化箱数，用于将连续的线性速度命令空间分成30个离散区间
    Cfg.commands.num_ang_vel_bins = 30                                      #角速度命令的离散化箱数，用于将连续的角速度命令空间分成30个离散区间
    Cfg.curriculum_thresholds.tracking_ang_vel = 0.7                        #角速度跟踪性能的课程学习阈值，当性能达到0.7时，会增加任务难度
    Cfg.curriculum_thresholds.tracking_lin_vel = 0.8                        #线性速度跟踪性能的课程学习阈值，当性能达到0.8时，会增加任务难度
    Cfg.curriculum_thresholds.tracking_contacts_shaped_vel = 0.90           #接触速度跟踪性能的课程学习阈值，接触 tracking（机器人腿与地面的接触匹配度），速度 tracking（如身体线速度匹配目标速度），shaped reward（有形奖励函数，比如不是稀疏奖励，而是连续奖励
    Cfg.curriculum_thresholds.tracking_contacts_shaped_force = 0.90         #接触力跟踪性能的课程学习阈值

    Cfg.commands.distributional_commands = True                             #启用分布式命令（distributional commands）模式，使得每次训练中机器人的目标命令（如速度、转向）不再是固定值，而是从设定好的分布中「随机采样」。

    Cfg.domain_rand.lag_timesteps = 6                                        #模拟控制延迟的时间步数，设置为6个时间步 
    Cfg.domain_rand.randomize_lag_timesteps = True                          #启用控制延迟的随机化，使延迟时间在训练中变化
    # Cfg.control.control_type = "actuator_net"                               #设置控制器类型为"actuator_net"，这是一种基于神经网络的执行器模型
    Cfg.control.control_type = "P"

    Cfg.domain_rand.randomize_rigids_after_start = False                    #禁用在模拟开始后随机化刚体参数
    Cfg.env.priv_observe_motion = False                                     #禁用将运动信息作为特权观察的一部分
    Cfg.env.priv_observe_gravity_transformed_motion = False                 #禁用将重力变换后的运动信息作为特权观察的一部分
    Cfg.domain_rand.randomize_friction_indep = False                        #禁用独立摩擦系数随机化（每个接触点独立随机化）
    Cfg.env.priv_observe_friction_indep = False                             #禁用将独立摩擦系数作为特权观察的一部分
    
    Cfg.domain_rand.randomize_friction = True
    Cfg.env.priv_observe_friction = True
    Cfg.domain_rand.friction_range = [0.1, 3.0]

    Cfg.domain_rand.randomize_restitution = True                            #启用弹性系数随机化，使环境中的弹性系数在训练中变化


    Cfg.env.priv_observe_restitution = True
    Cfg.domain_rand.restitution_range = [0.0, 0.4]
    Cfg.domain_rand.randomize_base_mass = True
    Cfg.env.priv_observe_base_mass = False
    Cfg.domain_rand.added_mass_range = [-1.0, 3.0]
    Cfg.domain_rand.randomize_gravity = True
    Cfg.domain_rand.gravity_range = [-1.0, 1.0]
    Cfg.domain_rand.gravity_rand_interval_s = 8.0
    Cfg.domain_rand.gravity_impulse_duration = 0.99
    Cfg.env.priv_observe_gravity = False
    Cfg.domain_rand.randomize_com_displacement = False
    Cfg.domain_rand.com_displacement_range = [-0.15, 0.15]
    Cfg.env.priv_observe_com_displacement = False
    Cfg.domain_rand.randomize_ground_friction = True
    Cfg.env.priv_observe_ground_friction = False
    Cfg.env.priv_observe_ground_friction_per_foot = False
    Cfg.domain_rand.ground_friction_range = [0.0, 0.0]
    Cfg.domain_rand.randomize_motor_strength = True
    Cfg.domain_rand.motor_strength_range = [0.9, 1.1]
    Cfg.env.priv_observe_motor_strength = False
    Cfg.domain_rand.randomize_motor_offset = True
    Cfg.domain_rand.motor_offset_range = [-0.02, 0.02]
    Cfg.env.priv_observe_motor_offset = False
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.env.priv_observe_Kp_factor = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.env.priv_observe_Kd_factor = False
    Cfg.env.priv_observe_body_velocity = False
    Cfg.env.priv_observe_body_height = False
    Cfg.env.priv_observe_desired_contact_states = False
    Cfg.env.priv_observe_contact_forces = False
    Cfg.env.priv_observe_foot_displacement = False
    Cfg.env.priv_observe_gravity_transformed_foot_displacement = False

    Cfg.env.num_privileged_obs = 2
    Cfg.env.num_observation_history = 30
    Cfg.reward_scales.feet_contact_forces = 0.0

    Cfg.domain_rand.rand_interval_s = 4
    Cfg.commands.num_commands = 15
    Cfg.env.observe_two_prev_actions = True
    Cfg.env.observe_yaw = False
    Cfg.env.num_observations = 70                           #default 70
    Cfg.env.num_scalar_observations = 70                    #default 70
    Cfg.env.observe_gait_commands = True
    Cfg.env.observe_timing_parameter = False
    Cfg.env.observe_clock_inputs = True

    Cfg.domain_rand.tile_height_range = [-0.0, 0.0]
    Cfg.domain_rand.tile_height_curriculum = False
    Cfg.domain_rand.tile_height_update_interval = 1000000
    Cfg.domain_rand.tile_height_curriculum_step = 0.01
    Cfg.terrain.border_size = 0.0
    Cfg.terrain.mesh_type = "trimesh"
    Cfg.terrain.num_cols = 20
    Cfg.terrain.num_rows = 20
    Cfg.terrain.terrain_width = 5.0
    Cfg.terrain.terrain_length = 5.0
    Cfg.terrain.x_init_range = 0.2
    Cfg.terrain.y_init_range = 0.2
    Cfg.terrain.teleport_thresh = 0.3
    Cfg.terrain.teleport_robots = False
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 4
    Cfg.terrain.horizontal_scale = 0.10
    Cfg.rewards.use_terminal_foot_height = False       
    Cfg.rewards.use_terminal_body_height = True
    Cfg.rewards.terminal_body_height = 0.3        #原来是0.05，现在训练trot，低于0.3就结束
    Cfg.rewards.use_terminal_roll_pitch = True
    Cfg.rewards.terminal_body_ori = 1.6

    Cfg.commands.resampling_time = 10

    Cfg.reward_scales.feet_slip = -0.04                 #惩罚脚部打滑，鼓励稳定行走
    Cfg.reward_scales.action_smoothness_1 = -0.1        #惩罚动作不平滑，鼓励动作连续性。
    Cfg.reward_scales.action_smoothness_2 = -0.1
    Cfg.reward_scales.dof_vel = -1e-4                   #惩罚关节速度过快，鼓励平稳运动
    Cfg.reward_scales.dof_pos = -0.0
    Cfg.reward_scales.jump = 10.0
    Cfg.reward_scales.base_height = -30.     #0.0
    Cfg.rewards.base_height_target = 0.48               #狗的目标站立高度
    Cfg.reward_scales.estimation_bonus = 0.0
    Cfg.reward_scales.raibert_heuristic = -10.0         #惩罚与 Raibert 启发式不一致的行为，鼓励特定的步态模式。
    Cfg.reward_scales.feet_impact_vel = -0.0
    Cfg.reward_scales.feet_clearance = -0.0
    Cfg.reward_scales.feet_clearance_cmd = -0.0
    Cfg.reward_scales.feet_clearance_cmd_linear = -30.0
    Cfg.reward_scales.orientation = -30.              
    Cfg.reward_scales.orientation_control = -7.5       #defalut-5  #惩罚机器人pitch/roll偏离0的程度
    Cfg.reward_scales.tracking_stance_width = -0.2
    Cfg.reward_scales.tracking_stance_length = -0.2
    Cfg.reward_scales.lin_vel_z = -0.02
    Cfg.reward_scales.ang_vel_xy = -0.001
    Cfg.reward_scales.feet_air_time = 0.0
    Cfg.reward_scales.hop_symmetry = 0.0
    Cfg.rewards.kappa_gait_probs = 0.07
    Cfg.rewards.gait_force_sigma = 100.
    Cfg.rewards.gait_vel_sigma = 10.
    Cfg.reward_scales.tracking_contacts_shaped_force = 4.0
    Cfg.reward_scales.tracking_contacts_shaped_vel = 4.0
    Cfg.reward_scales.collision = -5.0

    Cfg.rewards.reward_container_name = "CoRLRewards"
    Cfg.rewards.only_positive_rewards = False
    Cfg.rewards.only_positive_rewards_ji22_style = True
    Cfg.rewards.sigma_rew_neg = 0.02


    amplified_scalar = 1.5          #算了一下，g20的大小是go1 的1.5倍
    Cfg.commands.lin_vel_x = [-1.0, 1.0]
    Cfg.commands.lin_vel_y = [-0.6, 0.6]
    Cfg.commands.ang_vel_yaw = [-1.0, 1.0]
    Cfg.commands.body_height_cmd = [-0.25* amplified_scalar, 0.15* amplified_scalar] 

    #原来的多种步态
    Cfg.commands.gait_frequency_cmd_range = [2.0, 4.0]
    Cfg.commands.gait_phase_cmd_range = [0.0, 1.0]
    Cfg.commands.gait_offset_cmd_range = [0.0, 1.0]
    Cfg.commands.gait_bound_cmd_range = [0.0, 1.0]
    Cfg.commands.gait_duration_cmd_range = [0.5, 0.5]

    # # 只训练固定的trot步态
    # Cfg.commands.gait_frequency_cmd_range = [3.0, 3.0+1e-8]  # 固定频率
    # Cfg.commands.gait_phase_cmd_range = [0.5, 0.5+1e-8]      # 对角步态相位
    # Cfg.commands.gait_offset_cmd_range = [0.0, 0.0+1e-8]
    # Cfg.commands.gait_bound_cmd_range = [0.0, 0.0+1e-8]
    # Cfg.commands.gait_duration_cmd_range = [0.5, 0.5+1e-8]

    Cfg.commands.footswing_height_range = [0.03 * amplified_scalar, 0.35 * amplified_scalar] 
    # Cfg.commands.body_pitch_range = [-0.4, 0.4]
    Cfg.commands.body_pitch_range = [-0.0, 0.0]
    Cfg.commands.body_roll_range = [-0.0, 0.0]
    Cfg.commands.stance_width_range = [0.10 * amplified_scalar, 0.45 * amplified_scalar]  
    Cfg.commands.stance_length_range = [0.35 * amplified_scalar, 0.45 * amplified_scalar] 

    Cfg.commands.limit_vel_x = [-5.0, 5.0]
    Cfg.commands.limit_vel_y = [-0.6, 0.6]
    Cfg.commands.limit_vel_yaw = [-5.0, 5.0]
    Cfg.commands.limit_body_height = [-0.25, 0.15]
    Cfg.commands.limit_gait_frequency = [2.0, 4.0]
    Cfg.commands.limit_gait_phase = [0.0, 1.0]
    Cfg.commands.limit_gait_offset = [0.0, 1.0]
    Cfg.commands.limit_gait_bound = [0.0, 1.0]
    Cfg.commands.limit_gait_duration = [0.5, 0.5]
    Cfg.commands.limit_footswing_height = [0.03 * amplified_scalar, 0.35 * amplified_scalar]
    Cfg.commands.limit_body_pitch = [-0.4, 0.4]
    Cfg.commands.limit_body_roll = [-0.0, 0.0]
    # Cfg.commands.limit_stance_width = [0.10 * amplified_scalar, 0.45 * amplified_scalar]
    # Cfg.commands.limit_stance_length = [0.35 * amplified_scalar, 0.45 * amplified_scalar]
    Cfg.commands.limit_stance_width = [0.25 * amplified_scalar, 0.45 * amplified_scalar ]
    Cfg.commands.limit_stance_length = [0.4 * amplified_scalar, 0.45 * amplified_scalar]

    Cfg.commands.num_bins_vel_x = 21
    Cfg.commands.num_bins_vel_y = 1
    Cfg.commands.num_bins_vel_yaw = 21
    Cfg.commands.num_bins_body_height = 1
    Cfg.commands.num_bins_gait_frequency = 1
    Cfg.commands.num_bins_gait_phase = 1
    Cfg.commands.num_bins_gait_offset = 1
    Cfg.commands.num_bins_gait_bound = 1
    Cfg.commands.num_bins_gait_duration = 1
    Cfg.commands.num_bins_footswing_height = 1
    Cfg.commands.num_bins_body_roll = 1
    Cfg.commands.num_bins_body_pitch = 1
    Cfg.commands.num_bins_stance_width = 1

    Cfg.normalization.friction_range = [0, 1]
    Cfg.normalization.ground_friction_range = [0, 1]
    Cfg.terrain.yaw_init_range = 3.14
    Cfg.normalization.clip_actions = 10.0

    Cfg.commands.exclusive_phase_offset = False
    Cfg.commands.pacing_offset = False
    Cfg.commands.binary_phases = True
    Cfg.commands.gaitwise_curricula = True           #True的话就是打开课程学习；False禁用课程学习，只训练一种步态
    # config_q20(Cfg)       #revised by hongyu, 05/14/2025

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg)

    # # log the experiment parameters
    # logger.log_params(AC_Args=vars(AC_Args), PPO_Args=vars(PPO_Args), RunnerArgs=vars(RunnerArgs),
    #                   Cfg=vars(Cfg))
    
    #把实验数据记录到TensorBoard              #revised by hongyu, 05/15/2025, adding tensorboard      
    writer.add_text('Parameters/AC_Args', str(vars(AC_Args)))
    writer.add_text('Parameters/PPO_Args', str(vars(PPO_Args)))
    writer.add_text('Parameters/RunnerArgs', str(vars(RunnerArgs)))
    writer.add_text('Parameters/Cfg', str(vars(Cfg)))

    env = HistoryWrapper(env)
    gpu_id = 0
    runner = Runner(env, device=f"cuda:{gpu_id}",writer = writer,log_dir = log_dir)  # 传递writer给Runner 
    runner.learn(num_learning_iterations=100000, init_at_random_ep_len=True, eval_freq=100)

    #关闭writer
    writer.close()

if __name__ == '__main__':
    from pathlib import Path
    from ml_logger import logger
    from go1_gym import MINI_GYM_ROOT_DIR

    stem = Path(__file__).stem
    logger.configure(logger.utcnow(f'gait-conditioned-agility/%Y-%m-%d/{stem}/%H%M%S.%f'),
                     root=Path(f"{MINI_GYM_ROOT_DIR}/runs").resolve(), )
    logger.log_text("""
                charts: 
                - yKey: train/episode/rew_total/mean
                  xKey: iterations
                - yKey: train/episode/rew_tracking_lin_vel/mean
                  xKey: iterations
                - yKey: train/episode/rew_tracking_contacts_shaped_force/mean
                  xKey: iterations
                - yKey: train/episode/rew_action_smoothness_1/mean
                  xKey: iterations
                - yKey: train/episode/rew_action_smoothness_2/mean
                  xKey: iterations
                - yKey: train/episode/rew_tracking_contacts_shaped_vel/mean
                  xKey: iterations
                - yKey: train/episode/rew_orientation_control/mean
                  xKey: iterations
                - yKey: train/episode/rew_dof_pos/mean
                  xKey: iterations
                - yKey: train/episode/command_area_trot/mean
                  xKey: iterations
                - yKey: train/episode/max_terrain_height/mean
                  xKey: iterations
                - type: video
                  glob: "videos/*.mp4"
                - yKey: adaptation_loss/mean
                  xKey: iterations
                """, filename=".charts.yml", dedent=True)

    # to see the environment rendering, set headless=False
    # train_go1(headless=True)
    train_go1(headless=False)
