from typing import Union

from params_proto import Meta

from go1_gym.envs.base.legged_robot_config import Cfg


def config_q20(Cnfg: Union[Cfg, Meta]):
    _ = Cnfg.init_state

    _.pos = [0.0, 0.0, 0.521]  # x,y,z [m]    0.521          #初始位置
    _.default_joint_angles = {  # = target angles [rad] when action = 0.0
        'FL_hip_joint': 0.0,  # [rad]
        'RL_hip_joint': 0.0,  # [rad]
        'FR_hip_joint': 0.0,  # [rad]
        'RR_hip_joint': 0.0,  # [rad]

        'FL_thigh_joint': -0.9,  # [rad]
        'RL_thigh_joint': -0.9,  # [rad]
        'FR_thigh_joint': -0.9,  # [rad]
        'RR_thigh_joint': -0.9,  # [rad]            #数字越大,大腿与身体的夹角越大

        'FL_calf_joint': 1.3,  # [rad]
        'RL_calf_joint': 1.3,  # [rad]
        'FR_calf_joint': 1.3,  # [rad]
        'RR_calf_joint': 1.3  # [rad]

        # 'FL_calf_joint': 1.5,  # [rad]
        # 'RL_calf_joint': 1.5,  # [rad]
        # 'FR_calf_joint': 1.5,  # [rad]
        # 'RR_calf_joint': 1.5  # [rad]           #数字越小,小腿和大腿的夹角越大
    }

    _ = Cnfg.control
    _.control_type = 'P'
    _.stiffness = {'joint': 150.}  # [N*m/rad]
    _.damping = {'joint': 3.}  # [N*m*s/rad]
    # action scale: target angle = actionScale * action + defaultAngle
    _.action_scale = 0.25
    _.hip_scale_reduction = 1.0
    # decimation: Number of control action updates @ sim DT per policy DT
    _.decimation = 4

    _ = Cnfg.asset
    # _.file = '{MINI_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf'
    _.file = "/home/tong/Downloads/walk-these-ways/resources/robots/Q20/urdf/Q20B_simple.urdf"
    _.foot_name = "foot"
    _.penalize_contacts_on = ["thigh", "calf"]
    _.terminate_after_contacts_on = ["base_link", "thigh"]
    _.self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
    _.flip_visual_attachments = False
    _.fix_base_link = False



############################################################# 这里是训练go1时的参数 ############################################

    _ = Cnfg.rewards
    _.soft_dof_pos_limit = 0.9
    _.base_height_target = 0.48

    _ = Cnfg.reward_scales
    _.torques = -0.0001
    _.action_rate = -0.01
    _.dof_pos_limits = -10.0
    _.orientation = -5.
    _.base_height = -30.

    _ = Cnfg.terrain
    _.mesh_type = 'trimesh'
    _.measure_heights = False
    _.terrain_noise_magnitude = 0.0
    _.teleport_robots = True
    _.border_size = 50

    _.terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]
    _.curriculum = False
    # _.curriculum = True

    _ = Cnfg.env
    _.num_observations = 42
    _.observe_vel = False
    _.num_envs = 4000

    _ = Cnfg.commands
    _.lin_vel_x = [-1.0, 1.0]
    _.lin_vel_y = [-1.0, 1.0]

    _ = Cnfg.commands
    _.heading_command = False
    _.resampling_time = 10.0
    _.command_curriculum = True
    _.num_lin_vel_bins = 30
    _.num_ang_vel_bins = 30
    _.lin_vel_x = [-0.6, 0.6]
    _.lin_vel_y = [-0.6, 0.6]
    _.ang_vel_yaw = [-1, 1]

    _ = Cnfg.domain_rand
    _.randomize_base_mass = True
    _.added_mass_range = [-1, 3]
    _.push_robots = False
    _.max_push_vel_xy = 0.5
    _.randomize_friction = True
    _.friction_range = [0.05, 4.5]
    _.randomize_restitution = True
    _.restitution_range = [0.0, 1.0]
    _.restitution = 0.5  # default terrain restitution
    _.randomize_com_displacement = True
    _.com_displacement_range = [-0.1, 0.1]
    _.randomize_motor_strength = True
    _.motor_strength_range = [0.9, 1.1]
    _.randomize_Kp_factor = False
    _.Kp_factor_range = [0.8, 1.3]
    _.randomize_Kd_factor = False
    _.Kd_factor_range = [0.5, 1.5]
    _.rand_interval_s = 6




############################################################## 下面是Q20的设置 ################################################

    # _ = Cnfg.rewards
    # _.only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
    # _.tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
    # _.soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
    # _.soft_dof_vel_limit = 1.
    # _.soft_torque_limit = 1.
    # _.base_height_target = 0.49 # x20: 0.49   go1: 0.4 
    # _.max_contact_force = 250. # forces above this value are penalized
    # #下面两个原来代码里也没有
    # # _.clearance_height_target = -0.30
    # # _.desired_feet_height = 0.20 #0.1: x20 #0.07: go1


    # _ = Cnfg.reward_scales
    # # normal walking
    # _.termination = -0.0
    # _.tracking_lin_vel = 1.2
    # _.tracking_ang_vel = 0.6

    # _.lin_vel_z = -2.0
    # _.ang_vel_xy = -0.1
    # _.orientation = -0.2
    # _.dof_acc = -2.5e-7
    # _.base_height = -1.0
    # _.collision = -1
    # _.feet_stumble = -1.0
    # _.action_rate = -0.01
    
    # _.dof_pos_limits = -5.0
    
    # _.torques = -1.0e-7
    # _.feet_slip = -0.01
    # _.feet_impact_vel = -0.2

    # ######################################################  以下的东西原来代码里没有  ###################
    # # _.dof_vel_limits = -5.0
    # # _.joint_power = -2e-5
    # # _.default_pos = -1e-1 #-1e-1
    # # _.power_distribution = -1e-7
    # # _.smoothness= -0.001
    # # _.torques_rate = -1.0e-7
    # # _.contacts_shaped_force = 0.01 # origin 0.01
    # # _.contacts_shaped_vel = 0.01 # origin 0.01
    # # # feet_clearance_cmd_linear = -0.0 # origin: -1
    # # _.survive = 1.0 # origin: 1.0 used 2.
    # # _.deepmimic_dof_pos = 0.05 # origin 0.25 used 0.5
    # # _.deepmimic_dof_vel = 0.15 # origin 0.75 used 1.5
    # # _.feet_clearance_cmd_cpg = -1.0
    # #####################################################################################################

    # _ = Cnfg.terrain
    # _.mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
    # _.horizontal_scale = 0.1 # [m]
    # _.vertical_scale = 0.005 # [m]
    # _.border_size = 100 # [m]
    # _.curriculum = True
    # _.static_friction = 1.0
    # _.dynamic_friction = 1.0
    # _.restitution = 0.
    # # rough terrain only:
    # _.measure_heights = True
    # _.measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
    # _.measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
    # _.selected = False # select a unique terrain type and pass all arguments
    # _.terrain_kwargs = None # Dict of arguments for selected terrain
    # _.max_init_terrain_level = 5 # starting curriculum state
    # _.terrain_length = 8.
    # _.terrain_width = 8.
    # _.num_rows= 10 # number of terrain rows (levels)
    # _.num_cols = 20 # number of terrain cols (types)
    # # terrain types: [smooth slope, rough slope, stairs up,
    # # stairs down, discrete, stepping_stones,
    # # gap, roughness flat]
    # _.terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.2]
    # #terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0]
    # # trimesh only:
    # _.slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    # _ = Cnfg.env
    # # _.num_observations = 45 + 8
    # _.num_envs = 4096

    # _.num_observations = 42
    # # _.frequencies = 1.5


    # _ = Cnfg.commands
    
    # _.num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
    # _.resampling_time = 10. # time before command are changed[s]
    # _.heading_command = False # if true: compute ang vel command from heading error # origin: True, but now should be False
    # _.pacing_offset = False
    # _.lin_vel_x = [-1.5, 1.5] # min max [m/s] origin: [-1.0, 1.0]
    # _.lin_vel_y = [-0.6, 0.6]   # min max [m/s] origin: [-0.5, 0.5]
    # _.ang_vel_yaw = [-0.6, 0.6]    # min max [rad/s] origin: [-0.5, 0.5]

    # #下面这三个必须得加，不然报错self.transition.env_bins = infos["env_bins"  KeyError: 'env_bins'
    # _.command_curriculum = True
    # _.num_lin_vel_bins = 30
    # _.num_ang_vel_bins = 30

    # # #下面这三个原来的代码里也没有
    # # _.body_pitch_rangeheading = [-3.14, 3.14]
    # # _.curriculum = True
    # # _.max_curriculum = 1.5

    # _ = Cnfg.domain_rand
    # #friction
    # _.rand_interval_s = 10
    # _.randomize_friction = True
    # _.friction_range = [0.5,1.25]

    # #base_mass
    # _.randomize_base_mass = False
    # _.added_mass_range = [-5.0, 15.0]

    # #push robot
    # _.push_robots = True
    # _.max_push_vel_xy = 1.
    # _.push_interval_s = 15

    # #com displacement
    # _.randomize_com_displacement = False
    # _.com_displacement_range = [-0.1, 0.1]

    # #motor strength
    # _.randomize_motor_strength = False
    # _.motor_strength_range = [0.9, 1.1]

    
    # # Kp
    # _.randomize_Kp_factor = False
    # _.Kp_factor_range = [0.8, 1.2]
    # # Kd
    # _.randomize_Kd_factor = False
    # _.Kd_factor_range = [0.3, 2.0]
    # # gravity
    # _.randomize_gravity = False
    # _.gravity_range = [-1.0, 1.0]
    # _.gravity_rand_interval_s = 8.0
    # _.gravity_impulse_duration = 0.99
    # # restitution
    # _.randomize_restitution = False
    # _.restitution_range = [0, 0.3]
    # # lag timesteps
    # _.randomize_lag_timesteps = True
    # _.lag_timesteps = 1
    # # observation lag buffer
    # # randomize_obs_lag_timesteps = False
    # # obs_lag_timesteps = 2

    # ######################################################  以下的东西原来代码里没有  ###################
    # # # motor offset                                 
    # # _.randomize_motor_offset = False
    # # _.motor_offset_range = [-0.01, 0.01]


    # # _.disturbance = True
    # # _.disturbance_range = [-30.0, 30.0]
    # # _.disturbance_interval = 8

    # # _.randomize_link_mass = True
    # # _.link_mass_range = [0.9, 1.1]

    # # # !!! column friction randominzation!!!
    # # _.randomize_column_factor = True
    # # _.column_factor_range = [0, 5]
    # # # !!! column friction randominzation!!!
    # ###############################################################################################


    # _ = Cnfg.normalization
    # _.contact_force_range = [0.0, 100.0] #385 
    # _.clip_observations = 100.
    # _.clip_actions = 100.

    # _ = Cnfg.obs_scales
    # _.lin_vel = 2.0
    # _.ang_vel = 0.25
    # _.dof_pos = 1.0
    # _.dof_vel = 0.05
    # _.height_measurements = 1.0
    # # _.action = 1

    # _ = Cnfg.noise
    # _.add_noise = True
    # _.noise_level = 1.5 # scales other values

    # _ = Cnfg.noise_scales
    # _.dof_pos = 0.01
    # _.dof_vel = 1.5
    # _.lin_vel = 0.1
    # _.ang_vel = 0.2
    # _.gravity = 0.05
    # _.height_measurements = 0.1
    # # _.action = 0.0



