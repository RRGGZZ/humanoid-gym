# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class adamCfg2(LeggedRobotCfg):
    """
    Configuration class for the AdamL humanoid robot.
    """
    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 5
        c_frame_stack = 3
        num_single_obs = 81
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 117
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 23
        num_envs = 4096
        episode_length_s = 24     # episode length in seconds
        use_ref_actions = False  # speed up training by using reference actions

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/pnd_models-main/pnd_adam_lite/urdf/adam_lite.urdf'
#/home/r/Downloads/IsaacGym_Preview_4_Package/isaacgym/legged_gym-master/legged_gym-master/humanoid-gym-main/resources/robots/pnd_models-main/pnd_adam_lite/urdf/adam_lite.urdf
        name = "adam_lite"
        foot_name = "toe"
        knee_name =  "shin"

        terminate_after_contacts_on = ['torso','pelvis','shoulderPitchLeft','shoulderPitchRight','elbowLeft','elbowRight']
        penalize_contacts_on = ["torso","pelvis","shoulderPitchLeft","shoulderPitchRight","elbowLeft","elbowRight"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        # mesh_type = 'trimesh
        curriculum = False
        # rough terrain only:
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.

    class noise:
        add_noise = True
        noise_level = 0.6    # scales other values

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 0.5
            ang_vel = 0.1
            lin_vel = 0.05
            quat = 0.05
            height_measurements = 0.05

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.96]

        default_joint_angles = {  # = target angles [rad] when action = 0.0
        'hipPitch_Left': -0.32,
        'hipRoll_Left': 0.0,
        'hipYaw_Left': -0.18,
        'kneePitch_Left': 0.66,
        'anklePitch_Left': -0.39,
        'ankleRoll_Left': -0.0,

        'hipPitch_Right': -0.32,
        'hipRoll_Right': -0.0,
        'hipYaw_Right': 0.18,
        'kneePitch_Right': 0.66,
        'anklePitch_Right': -0.39,
        'ankleRoll_Right': 0.0,

        'waistRoll': 0.0,
        'waistPitch': 0.0,
        'waistYaw': 0.0,

        'shoulderPitch_Left':0.0,
        'shoulderRoll_Left':0.1,
        'shoulderYaw_Left':0.0,
        'elbow_Left':-0.3,

        'shoulderPitch_Right':0.0,
        'shoulderRoll_Right':-0.1,
        'shoulderYaw_Right':0.0,
        'elbow_Right':-0.3
        }

    class control(LeggedRobotCfg.control):
    # 刚度参数 [N*m/rad]
        stiffness = {
        # 左腿关节
        'hipPitch_Left': 305., 'hipRoll_Left': 700.0, 'hipYaw_Left': 405.0,
        'kneePitch_Left': 305., 'anklePitch_Left': 20.0, 'ankleRoll_Left': 0.,
        
        # 右腿关节
        'hipPitch_Right': 305., 'hipRoll_Right': 700.0, 'hipYaw_Right': 405.0,
        'kneePitch_Right': 305., 'anklePitch_Right': 20.0, 'ankleRoll_Right': 0.,
        
        # 腰部关节
        'waistRoll': 405.0, 'waistPitch': 405.0, 'waistYaw': 205.0,
        
        # 左臂关节
        'shoulderPitch_Left': 18.0, 'shoulderRoll_Left': 9.0, 'shoulderYaw_Left': 9.0, 'elbow_Left': 9.0,
        
        # 右臂关节
        'shoulderPitch_Right': 18.0, 'shoulderRoll_Right': 9.0, 'shoulderYaw_Right': 9.0, 'elbow_Right': 9.0
        }
    
    # 阻尼参数 [N*m*s/rad]
        damping = {
        # 左腿关节
        'hipPitch_Left': 6.1, 'hipRoll_Left': 30.0, 'hipYaw_Left': 6.1,
        'kneePitch_Left': 6.1, 'anklePitch_Left': 2.5, 'ankleRoll_Left': 0.35,
        
        # 右腿关节
        'hipPitch_Right': 6.1, 'hipRoll_Right': 30.0, 'hipYaw_Right': 6.1,
        'kneePitch_Right': 6.1, 'anklePitch_Right': 2.5, 'ankleRoll_Right': 0.35,
        
        # 腰部关节
        'waistRoll': 6.1, 'waistPitch': 6.1, 'waistYaw': 4.1,
        
        # 左臂关节
        'shoulderPitch_Left': 0.9, 'shoulderRoll_Left': 0.9, 'shoulderYaw_Left': 0.9, 'elbow_Left': 0.9,
        
        # 右臂关节
        'shoulderPitch_Right': 0.9, 'shoulderRoll_Right': 0.9, 'shoulderYaw_Right': 0.9, 'elbow_Right': 0.9
        }
    ## KdKd
        #stiffness = {   'hipPitch': 305., 'hipRoll': 700.0, 'hipYaw': 405.0,'kneePitch': 305., 'anklePitch': 20.0,'ankleRoll': 0.,
        #           'waistRoll': 405.0, 'waistPitch': 405.0, 'waistYaw': 205.0,
        #            'shoulderPitch':18.0,  'shoulderRoll':9.0,  'shoulderYaw':9.0, 'elbow':9.0
        #        }  # [N*m/rad]
        #damping = { 'hipPitch': 6.1, 'hipRoll': 30.0, 'hipYaw': 6.1,'kneePitch': 6.1, 'anklePitch': 2.5,'ankleRoll': 0.35,
        #        'waistRoll': 6.1, 'waistPitch': 6.1, 'waistYaw': 4.1,
        #        'shoulderPitch':0.9,  'shoulderRoll':0.9,  'shoulderYaw':0.9, 'elbow':0.9
        #        }  # [N*m*s/rad]

    # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
    # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 100hz
    
        predictive_RL = True 
        decimation_interpolation = 5
        decimation_drive = 2
        predictive_time = 0.02

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 1000 Hz
        substeps = 1
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 1
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand:
        randomize_friction = True
        friction_range = [0.1, 2.0]
        randomize_base_mass = True
        added_mass_range = [-5., 5.]
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.2
        max_push_ang_vel = 0.4
        # dynamic randomization
        action_delay = 0.5
        action_noise = 0.02

    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-0.5, 0.6]   # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0] # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards:
        base_height_target =0.815
        min_dist = 0.2
        max_dist = 0.7 # 0.7
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.17 #0.17   # rad
        target_feet_height = 0.07 #0.075       # m
        cycle_time = 0.64  #0.64              # sec
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        # tracking reward = exp(error*sigma)
        tracking_sigma = 5
        max_contact_force = 700  # Forces above this value are penalized

        class scales:
            # reference motion tracking
            joint_pos = 2.8
            feet_clearance = 1.0
            feet_contact_number = 2.0
            # gait
            feet_air_time = 1.0
            foot_slip = -0.05
            feet_distance = 0.22
            knee_distance = 0.25
            # contact
            feet_contact_forces = -0.01
            # vel tracking
            tracking_lin_vel = 1.2
            tracking_ang_vel = 1.1
            vel_mismatch_exp = 0.5  # lin_z; ang x,y
            low_speed = 0.2
            track_vel_hard = 0.5
            #stand_still=5 # keep stand still
            # base pos
            default_joint_pos = 1.2
            orientation = 1.
            base_height = 0.1
            base_acc = 0.2
            waist_stability = 1.5
            foot_orientation = 1.0
            
            # energy
            action_smoothness = -0.002
            torques = -1e-5
            dof_vel = -5e-4
            dof_acc = -1e-7
            collision = -1.

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        clip_observations = 18.
        clip_actions = 18.


class adamCfgPPO2(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = 'OnPolicyRunner'   # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]
        

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 60  # per iteration
        max_iterations = 10001  # number of policy updates

        # logging
        save_interval = 100  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'AdamWalk_ppo3'
        run_name = ''
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt