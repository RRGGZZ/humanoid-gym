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


from humanoid.envs.base.legged_robot_config import LeggedRobotCfg

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

import torch
from humanoid.envs import LeggedRobot

from humanoid.utils.terrain import  HumanoidTerrain



class adamFreeEnv2(LeggedRobot):
    '''
    XBotLFreeEnv is a class that represents a custom environment for a legged robot.

    Args:
        cfg (LeggedRobotCfg): Configuration object for the legged robot.
        sim_params: Parameters for the simulation.
        physics_engine: Physics engine used in the simulation.
        sim_device: Device used for the simulation.
        headless: Flag indicating whether the simulation should be run in headless mode.

    Attributes:
        last_feet_z (float): The z-coordinate of the last feet position.
        feet_height (torch.Tensor): Tensor representing the height of the feet.
        sim (gymtorch.GymSim): The simulation object.
        terrain (HumanoidTerrain): The terrain object.
        up_axis_idx (int): The index representing the up axis.
        command_input (torch.Tensor): Tensor representing the command input.
        privileged_obs_buf (torch.Tensor): Tensor representing the privileged observations buffer.
        obs_buf (torch.Tensor): Tensor representing the observations buffer.
        obs_history (collections.deque): Deque containing the history of observations.
        critic_history (collections.deque): Deque containing the history of critic observations.

    Methods:
        _push_robots(): Randomly pushes the robots by setting a randomized base velocity.
        _get_phase(): Calculates the phase of the gait cycle.
        _get_gait_phase(): Calculates the gait phase.
        compute_ref_state(): Computes the reference state.
        create_sim(): Creates the simulation, terrain, and environments.
        _get_noise_scale_vec(cfg): Sets a vector used to scale the noise added to the observations.
        step(actions): Performs a simulation step with the given actions.
        compute_observations(): Computes the observations.
        reset_idx(env_ids): Resets the environment for the specified environment IDs.
    '''
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # 初始化平均偏航速度
        self.avg_yaw_vel = torch.zeros(self.num_envs, device=self.device)
        self.last_feet_z = 0.05
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        # 新增：辅助计算张量初始化（形状为(num_envs, num_actions)）
        self.num_actions = self.actions.shape[-1]  # 获取动作维度
        self.action_d = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.last_action_d = torch.zeros_like(self.action_d)
        self.last_action_dot_d = torch.zeros_like(self.action_d)
        self.action_dot_d = torch.zeros_like(self.action_d)  # 补充定义
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        self.compute_observations()
        
    
    # 获取关节名称列表（假设已知关节顺序）
        self.joint_names = [
        'hipPitch_Left', 'hipRoll_Left', 'hipYaw_Left', 'kneePitch_Left', 'anklePitch_Left', 'ankleRoll_Left',
        'hipPitch_Right', 'hipRoll_Right', 'hipYaw_Right', 'kneePitch_Right', 'anklePitch_Right', 'ankleRoll_Right',
        'waistRoll', 'waistPitch', 'waistYaw',
        'shoulderPitch_Left', 'shoulderRoll_Left', 'shoulderYaw_Left', 'elbow_Left',
        'shoulderPitch_Right', 'shoulderRoll_Right', 'shoulderYaw_Right', 'elbow_Right'
        ]
    
    # 确保动作维度与关节数量一致
        assert len(self.joint_names) == self.actions.shape[-1], "动作维度与关节数量不匹配"



    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_push_angular = self.cfg.domain_rand.max_push_ang_vel
        self.rand_push_force[:, :2] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device)  # lin vel x/y
        self.root_states[:, 7:9] = self.rand_push_force[:, :2]

        self.rand_push_torque = torch_rand_float(
            -max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device)

        self.root_states[:, 10:13] = self.rand_push_torque

        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))



    def  _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase

    def _get_gait_phase(self):
        # return float mask 1 is stance, 0 is swing
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos >= 0
        # right foot stance
        stance_mask[:, 1] = sin_pos < 0
        # Double support phase
        stance_mask[torch.abs(sin_pos) < 0.1] = 1

        return stance_mask
    
    def _compute_torques_pv(self, actions):
        """Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.
        Args:
            actions (torch.Tensor): Actions
        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller with vel feedforward
        actions_scaled = actions * self.cfg.control.action_scale
        torques = (
            self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos)
            - self.d_gains * self.dof_vel
        )
        torques *= self.motor_strength
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def compute_ref_state(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        scale_1 = self.cfg.rewards.target_joint_pos_scale
        scale_2 = 2 * scale_1
        # left foot stance phase set to default joint pos
        sin_pos_l[sin_pos_l > 0] = 0
        self.ref_dof_pos[:, 0] = sin_pos_l*scale_1 + self.default_dof_pos[:, 0]
        self.ref_dof_pos[:, 3] = -sin_pos_l*scale_2 + self.default_dof_pos[:, 3]
        self.ref_dof_pos[:, 4] = sin_pos_l*scale_1 + self.default_dof_pos[:, 4]
        self.ref_dof_pos[:, 15] =  -sin_pos_r*scale_1+self.default_dof_pos[:, 15]
        self.ref_dof_pos[:, 16] = self.default_dof_pos[:, 16]
        self.ref_dof_pos[:, 17] = self.default_dof_pos[:, 17]
        self.ref_dof_pos[:, 18] =  -sin_pos_r*scale_1+self.default_dof_pos[:, 18]
        # right foot stance phase set to default joint pos
        sin_pos_r[sin_pos_r < 0] = 0
        self.ref_dof_pos[:, 6] = -sin_pos_r*scale_1 + self.default_dof_pos[:, 6]
        self.ref_dof_pos[:, 9] = sin_pos_r*scale_2 + self.default_dof_pos[:, 9]
        self.ref_dof_pos[:, 10] = -sin_pos_r*scale_1 + self.default_dof_pos[:, 10]
        self.ref_dof_pos[:, 19] = sin_pos_l*scale_1+self.default_dof_pos[:, 19]
        self.ref_dof_pos[:, 20] = self.default_dof_pos[:, 20]
        self.ref_dof_pos[:, 21] = self.default_dof_pos[:, 21]
        self.ref_dof_pos[:, 22] = sin_pos_l*scale_1+self.default_dof_pos[:, 22]
        # Double support phase
        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0

        self.ref_action = 2 * self.ref_dof_pos
    

    # def compute_ref_state(self):
    #     phase = self._get_phase()
    #     sin_pos = torch.sin(2 * torch.pi * phase)
    #     sin_pos_l = sin_pos.clone()
    #     sin_pos_r = sin_pos.clone()
    #     self.ref_dof_pos = torch.zeros_like(self.dof_pos)
    #     scale_1 = self.cfg.rewards.target_joint_pos_scale
    #     scale_2 = 2 * scale_1
    
    # # 基于命令的 roll 和 yaw 控制
    #     roll_cmd = self.commands[:, 1] * 0.2  # 横向速度映射到 roll
    #     yaw_cmd = self.commands[:, 2] * 0.15  # 偏航角速度映射到 yaw
    
    # # 行走方向和速度（用于调整步态形状）
    #     walk_dir = torch.sign(self.commands[:, 0])  # 前向/后向
    #     walk_speed = torch.abs(self.commands[:, 0])  # 速度大小
    
    # # 定义步态调整参数（根据速度和方向动态调整）
    #     hip_roll_offset = 0.08 * walk_dir * (walk_speed > 0.1)  # 髋关节 roll 偏移
    #     hip_yaw_offset = 0.1 * walk_dir * (walk_speed > 0.1)   # 髋关节 yaw 偏移
    
    # # 左足站立阶段 - 设置为默认关节位置 + 步态调整
    #     sin_pos_l[sin_pos_l > 0] = 0
    #     self.ref_dof_pos[:, 0] = sin_pos_l * scale_1 + self.default_dof_pos[:, 0]  # hipPitch_Left
    #     self.ref_dof_pos[:, 1] = hip_roll_offset * sin_pos_l * scale_1 + self.default_dof_pos[:, 1]      # hipRoll_Left (减少外八)
    #     self.ref_dof_pos[:, 2] = hip_yaw_offset * sin_pos_l * scale_1 + self.default_dof_pos[:, 2]       # hipYaw_Left (减少外八)
    #     self.ref_dof_pos[:, 3] = -sin_pos_l * scale_2 + self.default_dof_pos[:, 3]  # kneePitch_Left
    #     self.ref_dof_pos[:, 4] = sin_pos_l * scale_1 + self.default_dof_pos[:, 4]  # anklePitch_Left
    #     self.ref_dof_pos[:, 5] = self.default_dof_pos[:, 5]  # ankleRoll_Left
    
    # # 右臂摆动（与左足站立阶段配合）
    #     self.ref_dof_pos[:, 15] = -sin_pos_r * scale_1 + self.default_dof_pos[:, 15]  # shoulderPitch_Left
    #     #self.ref_dof_pos[:, 16] = self.default_dof_pos[:, 16] + roll_cmd * 0.5 * scale_1 # shoulderRoll_Left
    #     #self.ref_dof_pos[:, 17] = self.default_dof_pos[:, 17] - yaw__cmd * 0.5 * scale_1 # shoulderRoll_Left
    #     #self.ref_dof_pos[:, 17] = self.default_dof_pos[:, 17] - yaw_cmd * 0.3 * scale_1 # shoulderYaw_Left
    #     self.ref_dof_pos[:, 18] = -sin_pos_r*scale_1 + self.default_dof_pos[:, 18]  # elbow_Left
    
    # # 右足站立阶段 - 设置为默认关节位置 + 步态调整
    #     sin_pos_r[sin_pos_r < 0] = 0
    #     self.ref_dof_pos[:, 6] = -sin_pos_r * scale_1 + self.default_dof_pos[:, 6]  # hipPitch_Right
    #     self.ref_dof_pos[:, 7] = -hip_roll_offset * -sin_pos_r * scale_1 + self.default_dof_pos[:, 7]      # hipRoll_Right (减少外八，符号相反)
    #     self.ref_dof_pos[:, 8] = -hip_yaw_offset * -sin_pos_r * scale_1 + self.default_dof_pos[:, 8]       # hipYaw_Right (减少外八，符号相反)
    #     self.ref_dof_pos[:, 9] = sin_pos_r * scale_2 + self.default_dof_pos[:, 9]  # kneePitch_Right
    #     self.ref_dof_pos[:, 10] = -sin_pos_r * scale_1 + self.default_dof_pos[:, 10]  # anklePitch_Right
    #     self.ref_dof_pos[:, 11] = self.default_dof_pos[:, 11]  # ankleRoll_Right
    
    # # 左臂摆动（与右足站立阶段配合）
    #     self.ref_dof_pos[:, 19] = sin_pos_l * scale_1 + self.default_dof_pos[:, 19]  # shoulderPitch_Right
    #     #self.ref_dof_pos[:, 20] = self.default_dof_pos[:, 20] - roll_cmd * 0.5 * scale_1  # shoulderRoll_Right
    #     #self.ref_dof_pos[:, 21] = self.default_dof_pos[:, 21] + yaw_cmd * 0.3 * scale_1   # shoulderYaw_Right
    #     self.ref_dof_pos[:, 22] = sin_pos_l * scale_1 + self.default_dof_pos[:, 22]  # elbow_Right
    
    # # 腰部关节控制 - 基于命令的 roll 和 yaw
    #    # self.ref_dof_pos[:, 12] = roll_cmd * 0.3 * scale_1  # waistRoll (横向倾斜)
    #    # self.ref_dof_pos[:, 14] = yaw_cmd * 0.8  * scale_1 # waistYaw (身体转向)
    
    # # 双支撑阶段 - 减少动作幅度，提高稳定性
    #     double_support = torch.abs(sin_pos) < 0.1
    #     self.ref_dof_pos[double_support] *= 0.7  # 降低双支撑阶段的动作幅度
    
    # # 将参考位置转换为参考动作（比例缩放）
    #     self.ref_action = 2 * self.ref_dof_pos


    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = HumanoidTerrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(
            self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_vec[0: 6] = 0.  # commands
        noise_vec[6: 29] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[29: 52] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[52: 75] = 0.  # previous actions
        noise_vec[75: 78] = noise_scales.ang_vel * self.obs_scales.ang_vel   # ang vel
        noise_vec[78: 81] = noise_scales.quat * self.obs_scales.quat         # euler x,y
        return noise_vec

##########################################################################################################
        def step(self, actions):

            """Apply actions, simulate, call self.post_physics_step()

            Args:actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)"""

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        

        # step physics and render each frame
        # interplation actions
        self.tra_plan_0 = self.last_action_d
        self.tra_plan_1 = self.last_action_dot_d
        self.tra_plan_2 = (
            3
            * (
                self.actions
                - self.last_action_d
                - self.last_action_dot_d * self.cfg.control.predictive_time
            )
            / self.cfg.control.predictive_time
            / self.cfg.control.predictive_time
            - (-self.last_action_dot_d) / self.cfg.control.predictive_time
        )
        self.tra_plan_3 = (
            -2
            * (
                self.actions
                - self.last_action_d
                - self.last_action_dot_d * self.cfg.control.predictive_time
            )
            / self.cfg.control.predictive_time
            / self.cfg.control.predictive_time
            / self.cfg.control.predictive_time
            + (-self.last_action_dot_d)
            / self.cfg.control.predictive_time
            / self.cfg.control.predictive_time
        )
        # step physics and render each frame
        self.render()
        for i in range(self.cfg.control.decimation_interpolation):
            self.current_time = (i + 1) * self.cfg.control.decimation_drive * self.sim_params.dt

            if self.cfg.control.predictive_RL:
                self.action_d = (
                    self.tra_plan_3
                    * self.current_time
                    * self.current_time
                    * self.current_time
                    + self.tra_plan_2 * self.current_time * self.current_time
                    + self.tra_plan_1 * self.current_time
                    + self.tra_plan_0
                )
                self.action_dot_d = (
                    3 * self.tra_plan_3 * self.current_time * self.current_time
                    + 2 * self.tra_plan_2 * self.current_time
                    + self.tra_plan_1
                )
                self.action_ddot_d = (
                    6 * self.tra_plan_3 * self.current_time + 2 * self.tra_plan_2
                )
            else:
                self.action_d = self.actions * 0.8 + self.last_actions * 0.2

            for j in range(self.cfg.control.decimation_drive):
                self.torques = self._compute_torques_pv(self.action_d).view(self.torques.shape)
                self.gym.set_dof_actuation_force_tensor(
                    self.sim, gymtorch.unwrap_tensor(self.torques)
                )
                self.gym.simulate(self.sim)
                if self.device == "cpu":
                    self.gym.fetch_results(self.sim, True)
                self.gym.refresh_dof_state_tensor(self.sim)

        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras



#########################################################################################################
    def compute_observations(self):
    # 初始化帧堆叠参数
        frame_stack = 5  
        c_frame_stack = 3

        stance_mask = self._get_gait_phase()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.

        ##pndbotics 
        phase = self._get_phase()
        self.compute_ref_state()
        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)
        self.gait_phase = torch.cat([sin_pos, cos_pos], dim=1)
        self.avg_yaw_vel = (
        self.dt / self.cfg.rewards.cycle_time * self.base_ang_vel[:, 2]
        + (1 - self.dt / self.cfg.rewards.cycle_time) * self.avg_yaw_vel
        )

        self.command_input = torch.cat(
            (sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)
    
        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel
    
        diff = self.dof_pos - self.ref_dof_pos



        self.privileged_obs_buf = torch.cat((
            self.command_input,  # 2 + 3
            (self.dof_pos - self.default_joint_pd_target) * self.obs_scales.dof_pos,  # 23
            self.dof_vel * self.obs_scales.dof_vel,  # 23
            self.actions,  # 23
            diff,  # 23
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz * self.obs_scales.quat,  # 3
            self.rand_push_force[:, :2],  # 2
            self.rand_push_torque,  # 3
            self.env_frictions,  # 1
            self.body_mass / 30.,  # 1
            stance_mask,  # 2
            contact_mask,  # 2
        ), dim=-1)

        obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,#3
            self.projected_gravity, #3
            q,    # 23D
            dq,  # 23D
            self.actions,   # 23D
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.avg_yaw_vel.unsqueeze(1) * self.obs_scales.ang_vel, #1
            self.gait_phase,#2
            #self.base_euler_xyz * self.obs_scales.quat,  # 3
            ), dim=-1)

        #print(f"obs_buf 维度: {obs_buf.shape[-1]}")
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)

  

    # 应用噪声（如果配置）
        if self.add_noise:  
            obs_now = obs_buf.clone() + torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()
        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)
        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # 新增：辅助张量重置
        if env_ids is None:
            self.action_d.zero_()
            self.last_action_d.zero_()
            self.last_action_dot_d.zero_()
        else:
            self.action_d[env_ids].zero_()
            self.last_action_d[env_ids].zero_()
            self.last_action_dot_d[env_ids].zero_()

        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0


# ================================================ Rewards ================================================== #
    def _reward_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        diff = joint_pos - pos_target
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        return r

    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away.
        """
        foot_pos = self.rigid_state[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2


    def _reward_knee_distance(self):
        """
        Calculates the reward based on the distance between the knee of the humanoid.
        """
        foot_pos = self.rigid_state[:, self.knee_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist / 2
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2


    def _reward_foot_slip(self):
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces 
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact 
        with the ground. The speed of the foot is calculated and scaled by the contact condition.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 7:9], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)    

    def _reward_feet_air_time(self):
        """
        Calculates the reward for feet air time, promoting longer steps. This is achieved by
        checking the first contact with the ground after being in the air. The air time is
        limited to a maximum value for reward calculation.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        self.contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        self.feet_air_time *= ~self.contact_filt
        return air_time.sum(dim=1)

    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase. 
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == stance_mask, 1.0, -0.3)
        return torch.mean(reward, dim=1)

    def _reward_orientation(self):
        """
        使用角速度评估姿态稳定性，惩罚快速姿态变化和异常偏航
        """
        # 基础角速度奖励（惩罚快速旋转）
        base_ang_vel_norm = torch.norm(self.base_ang_vel * self.obs_scales.ang_vel, dim=1)
        ang_vel_reward = torch.exp(-base_ang_vel_norm * 10)  # 角速度越小奖励越高
    
        # 平均偏航速度奖励（惩罚异常偏航）
        avg_yaw_vel_scaled = self.avg_yaw_vel.unsqueeze(1) * self.obs_scales.ang_vel
        yaw_vel_reward = torch.exp(-torch.abs(avg_yaw_vel_scaled).squeeze(1) * 20)  # 偏航速度越小奖励越高
    
        # 投影重力向量奖励（确保直立）
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
    
        # 综合奖励（赋予角速度更高权重以快速抑制异常摆动）
        return (ang_vel_reward * 0.4 + yaw_vel_reward * 0.5 + orientation * 0.3)


    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(0, 400), dim=1)

    def _reward_default_joint_pos(self):
        """
        计算关节位置接近默认位置的奖励，特别关注 penalizing 偏航和滚动方向的偏差。
        排除腰部关节的偏航和滚动偏差。
        """
        joint_diff = self.dof_pos - self.default_joint_pd_target
    
        # 提取左右腿的偏航和滚动关节
        # 左: hipRoll_Left(1), hipYaw_Left(2), ankleRoll_Left(5)
        # 右: hipRoll_Right(7), hipYaw_Right(8), ankleRoll_Right(11)
        left_yaw_roll = torch.cat([
            joint_diff[:, 1:3],  # hipRoll_Left(1), hipYaw_Left(2)
            2*joint_diff[:, 5:6],   # ankleRoll_Left(5)
            2*joint_diff[:, 16:18]  # shoulderRoll_Left(16), shoulderYaw_Left(17)
        ], dim=1)
    
        right_yaw_roll = torch.cat([
            joint_diff[:, 7:9],  # hipRoll_Right(7), hipYaw_Right(8)
            2*joint_diff[:, 11:12], # ankleRoll_Right(11)
            2*joint_diff[:, 20:22] # shoulderRoll_Right(20), shoulderYaw_Right(21)
        ], dim=1)

    
        # 提取腰部关节的Roll和Yaw
        # waist_joints = torch.cat([ 
        #    joint_diff[:, 12:15],  # waistRoll(12)
        #], dim=1)
    
        # 计算偏航和滚动关节的范数（包括手部关节）
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
    
        # 计算所有关节的范数作为次要惩罚
        return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)
    
    def _reward_waist_stability(self):
        """
        计算腰部稳定性奖励，惩罚腰部向一侧偏移的情况
        """
        
        # 提取腰部关节的当前位置
        waist_roll = self.dof_pos[:, 12]  # waistRoll(12)
        waist_pitch = self.dof_pos[:, 13] # waistPitch(13)
        waist_yaw = self.dof_pos[:, 14]   # waistYaw(14)
        
    
        # 计算与默认位置的偏差（默认均为0）
        roll_diff = torch.abs(waist_roll)
        pitch_diff = torch.abs(waist_pitch)
        yaw_diff = torch.abs(waist_yaw)
    
        # 计算腰部整体偏移（可根据实际情况调整权重）
        waist_offset = 0.4 * roll_diff + 0.3 * yaw_diff + 0.3 * pitch_diff  # 给roll更高权重，防止向一侧偏移
    
        # 添加平滑因子，避免小偏差产生过大惩罚
        smooth_factor = 10.0
        waist_penalty = torch.exp(-smooth_factor * waist_offset)
    
        # 添加一个小的常数奖励，鼓励保持稳定
        stability_bonus = 0.5 * torch.ones_like(waist_penalty)
    
        return waist_penalty + stability_bonus


    def _reward_base_height(self):
        """
        Calculates the reward based on the robot's base height. Penalizes deviation from a target base height.
        The reward is computed based on the height difference between the robot's base and the average height 
        of its feet when they are in contact with the ground.
        """
        stance_mask = self._get_gait_phase()
        measured_heights = torch.sum(
        self.rigid_state[:, self.feet_indices, 2] * stance_mask, dim=1) / torch.sum(stance_mask, dim=1)
        base_height = self.root_states[:, 2] - (measured_heights - 0.05)
        return torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 100)

    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew


    def _reward_vel_mismatch_exp(self):
        """
        Computes a reward based on the mismatch in the robot's linear and angular velocities. 
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        """
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.)

        c_update = (lin_mismatch + ang_mismatch) / 2.

        return c_update

    def _reward_track_vel_hard(self):
        """
        Calculates a reward for accurately tracking both linear and angular velocity commands.
        Penalizes deviations from specified linear and angular velocity targets.
        """
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.norm(
            self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)

        linear_error = 0.2 * (lin_vel_error + ang_vel_error)

        return (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error

    def _reward_tracking_lin_vel(self):
        """
        Tracks linear velocity commands along the xy axes. 
        Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
        """
        lin_vel_error = torch.sum(torch.square(
            self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error * self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        """
        Tracks angular velocity commands for yaw rotation.
        Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
        """   
        
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma)
    
    def _reward_feet_clearance(self):
        """
        Calculates reward based on the clearance of the swing leg from the ground during movement.
        Encourages appropriate lift of the feet during the swing phase of the gait.
        """
        # Compute feet contact mask
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.

        # Get the z-position of the feet and compute the change in z-position
        feet_z = self.rigid_state[:, self.feet_indices, 2] - 0.05
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z

        # Compute swing mask
        swing_mask = 1 - self._get_gait_phase()

        # feet height should be closed to target feet height at the peak
        rew_pos = torch.abs(self.feet_height - self.cfg.rewards.target_feet_height) < 0.01
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_height *= ~contact
        return rew_pos

    def _reward_low_speed(self):
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed. 
        This function checks if the robot is moving too slow, too fast, or at the desired speed, 
        and if the movement direction matches the command.
        """
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(
            self.base_lin_vel[:, 0]) != torch.sign(self.commands[:, 0])

        # Initialize reward tensor
        reward = torch.zeros_like(self.base_lin_vel[:, 0])

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.
        # Speed within desired range
        reward[speed_desired] = 1.2
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.1)
    
    def _reward_torques(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        """
        Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages smoother and 
        more controlled movements.
        """
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_collision(self):
        """
        Penalizes collisions of the robot with the environment, specifically focusing on selected body parts.
        This encourages the robot to avoid undesired contact with objects or surfaces.
        """
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.15)

    def _reward_foot_orientation(self):
        """
        计算脚部朝向与运动方向的一致性奖励
        减少"外八"步态问题，使脚部yaw方向与运动方向对齐
        """
    # 获取左右脚的朝向（yaw角度）
        foot_quats_l = self.rigid_state[:, self.feet_indices[0], 3:7]  # 左脚四元数
        foot_quats_r = self.rigid_state[:, self.feet_indices[1], 3:7]  # 右脚四元数
    
    # 四元数转欧拉角，提取yaw分量(绕Z轴旋转角度)
        foot_yaws_l = get_euler_xyz(foot_quats_l)[2]  # 假设返回(x,y,z)，我们需要z(yaw)
        foot_yaws_r = get_euler_xyz(foot_quats_r)[2]
    
    # 计算运动方向（基于线速度）
        lin_vel_xy = self.base_lin_vel[:, :2]
    # 添加小量避免除零
        lin_vel_mag = torch.clamp_min(torch.norm(lin_vel_xy, dim=1, keepdim=True), 1e-6)
        movement_direction = torch.atan2(lin_vel_xy[:, 1], lin_vel_xy[:, 0])
    
    # 计算速度因子（用于平衡运动方向和默认朝向的影响）
        speed_factor = torch.clamp(lin_vel_mag / 0.5, 0.0, 1.0).squeeze()
    
    # 获取机器人整体朝向
        robot_yaw = self.base_euler_xyz[:, 2]
    
    # 计算期望的脚部朝向
    # 当速度较高时，参考运动方向；速度较低时，参考机器人整体朝向
        desired_orientation_l = speed_factor * movement_direction + (1 - speed_factor) * (robot_yaw - 0.1)
        desired_orientation_r = speed_factor * movement_direction + (1 - speed_factor) * (robot_yaw + 0.1)
    
    # 计算角度差（wrap到[-pi, pi]）
        angle_diff_l = torch.remainder(foot_yaws_l - desired_orientation_l + torch.pi, 2 * torch.pi) - torch.pi
        angle_diff_r = torch.remainder(foot_yaws_r - desired_orientation_r + torch.pi, 2 * torch.pi) - torch.pi
    
    # 将角度差转换为奖励（角度差越小，奖励越高）
        reward_l = torch.exp(-8.0 * torch.abs(angle_diff_l))  # 调整系数以控制敏感度
        reward_r = torch.exp(-8.0 * torch.abs(angle_diff_r))
    
    # 平均左右脚的奖励
        return (reward_l + reward_r) / 2.0


    

