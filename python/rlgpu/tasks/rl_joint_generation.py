import glob
import os

import numpy as np
import pandas as pd
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_apply, tensor_clamp, to_torch, torch_rand_float
from rlgpu.tasks.base.base_task import BaseTask
from tasks.pytorch3d import (
    matrix_to_rotation_6d,
    quaternion_to_matrix,
    random_quaternions,
)
import torch


class RLJointGeneration(BaseTask):
    def __init__(
        self,
        task_name,
        cfg,
        sim_params,
        physics_engine,
        device_type,
        device_id,
        headless,
    ):
        # parameters acquired from parse_task.py
        self.task_name = task_name.lower()
        self.cfg = cfg
        self.headless = headless
        self.sim_params = sim_params
        self.physics_engine = physics_engine

        # call up parameters from the yaml configuration file
        self.spacing = self.cfg["env"]["env_spacing"]
        self.max_episode_length = self.cfg["env"]["episode_length"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.dt = 1 / 60.0

        self.debug_vis = self.cfg["env"]["enable_debug_vis"]

        self.start_position_noise = self.cfg["env"]["start_position_noise"]
        self.start_rotation_noise = self.cfg["env"]["start_rotation_noise"]

        self.curr_modus = self.cfg["env"]["curr_modus"]
        self.curr_level = self.cfg["env"]["curr_level"]
        if self.curr_modus == 2:
            self.num_curr_level = sum(
                len(files)
                for _, _, files in os.walk(
                    f"../curriculum_data/{self.task_name}_curriculums/"
                )
            )
        else:
            self.num_curr_level = self.cfg["env"]["num_curr_level"]
        self.num_rolling_avg = 25

        self.distance_reward_scale = self.cfg["env"]["distance_reward_scale"]
        self.obstacle_reward_scale = self.cfg["env"]["obstacle_reward_scale"]
        self.obstacle_maximum_distance = self.cfg["env"]["obstacle_maximum_distance"]
        self.success_tolerance = self.cfg["env"]["success_tolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reach_goal_bonus"]

        self.control_type = self.cfg["robot"]["control_type"]
        self.action_scale = self.cfg["robot"]["action_scale"]
        self.dof_vel_scale = self.cfg["robot"]["dof_vel_scale"]
        self.visual_attachments = self.cfg["robot"]["visual_attachments"]

        self.dof_damping = self.cfg["robot"]["dof_damping"]
        self.dof_stiffness = self.cfg["robot"]["dof_stiffness"]
        self.dof_vel_lower_limits = self.cfg["robot"]["dof_vel_lower_limits"]
        self.dof_vel_upper_limits = self.cfg["robot"]["dof_vel_upper_limits"]

        self.obst_size = self.cfg["robot"]["obst_size"]
        self.link_safety_dist = self.cfg["robot"]["link_safety_dist"]

        self.spher_radius_low = self.cfg["robot"]["spher_radius_low"]
        self.spher_radius_up = self.cfg["robot"]["spher_radius_up"]
        self.spher_theta_low = np.deg2rad(self.cfg["robot"]["spher_theta_low"])
        self.spher_theta_up = np.deg2rad(self.cfg["robot"]["spher_theta_up"])
        self.spher_phi_low = np.deg2rad(self.cfg["robot"]["spher_phi_low"])
        self.spher_phi_up = np.deg2rad(self.cfg["robot"]["spher_phi_up"])

        self.torus_major_radius = self.cfg["robot"]["torus_major_radius"]
        self.torus_minor_radius_low = self.cfg["robot"]["torus_minor_radius_low"]
        self.torus_minor_radius_up = self.cfg["robot"]["torus_minor_radius_up"]
        self.torus_theta_low = np.deg2rad(self.cfg["robot"]["torus_theta_low"])
        self.torus_theta_up = np.deg2rad(self.cfg["robot"]["torus_theta_up"])
        self.torus_phi_low = np.deg2rad(self.cfg["robot"]["torus_phi_low"])
        self.torus_phi_up = np.deg2rad(self.cfg["robot"]["torus_phi_up"])

        self.num_obst = self.cfg["env"]["num_obst"]
        self.num_dofs = self.cfg["env"]["num_dofs"]
        self.goal_representation = self.cfg["env"]["goal_representation"]
        self.error_representation = self.cfg["env"]["error_representation"]
        num_observations = self.error_representation + (2 * self.goal_representation) + (2 * self.num_dofs) + (len(self.link_safety_dist) * self.num_obst)
        num_actions = self.cfg["env"]["num_actions"]

        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        self.cfg["env"]["num_actions"] = num_actions
        self.cfg["env"]["num_observations"] = num_observations
        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = self.headless

        # initialize the constructor inheritaged from BaseTask to initialize the simulation
        super().__init__(cfg=self.cfg)

        # specify where the camera is position and what the target it looks at
        if self.headless == False:
            cam_pos = gymapi.Vec3(np.sqrt(self.num_envs) * self.spacing * 2, np.sqrt(self.num_envs) * self.spacing * 2, 8.0)
            cam_target = gymapi.Vec3(np.sqrt(self.num_envs), np.sqrt(self.num_envs), -8.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors which holds the state of all bodies in the simulation
        # root state holds 3 floats for position, 4 floats for quaternions, 3 floats for linear velocity and 3 floats for angular velocity (13)
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)

        # update the contents of the state tensors from the physics engine
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        # 13 floats for each robot, obstacle and for every environment (num_env x 2+num_obst x 13)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)

        if self.num_obst > 0:
            self.obstacle_states = self.root_state_tensor[:, 2:]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        # create some wrappers for each state tensor to access the contents of it
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor) # a num_env*13 x 2 matrix
        self.robot_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_robot_dofs]
        self.robot_dof_pos = self.robot_dof_state[..., 0]
        self.robot_dof_vel = self.robot_dof_state[..., 1]

        # create memory allocation for the target
        self._num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.robot_dof_pos_targets = torch.zeros((self.num_envs, self._num_dofs), dtype=torch.float, device=self.device)
        self.robot_dof_vel_targets = torch.zeros((self.num_envs, self._num_dofs), dtype=torch.float, device=self.device)

        # unit vectors used for rotations
        self.x = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        # curriculum buffer is used to advance from one region to the next upon satisfying the criteria
        self.curriculum_buf = torch.zeros_like(self.reset_buf, dtype=torch.float, device=self.device)
        self.position_buf = torch.zeros_like(self.reset_buf, dtype=torch.float, device=self.device)
        self.orientation_buf = torch.zeros_like(self.reset_buf, dtype=torch.float, device=self.device)
        self.success_rate_buf = torch.zeros_like(self.reset_buf, dtype=torch.int, device=self.device)
        self.rolling_avg_success_rate = torch.tensor(0.0, dtype=torch.float, device=self.device)

        # reset the environment before start
        self.reset(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs(num_per_row=int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_per_row):
        lower = gymapi.Vec3(-self.spacing, -self.spacing, 0.0)
        upper = gymapi.Vec3(self.spacing, self.spacing, self.spacing)

        asset_root = "../../assets"
        robot_asset_file = f"urdf/{self.task_name}/{self.task_name}.urdf"

        # load robot asset
        robot_asset_options = gymapi.AssetOptions()
        robot_asset_options.fix_base_link = True
        robot_asset_options.flip_visual_attachments = self.visual_attachments
        # robot_asset_options.collapse_fixed_joints = True # breaks the end_effector_link and the corresponding end_effector_pos that is retrieved from refresh_rigid_body_state_tensor()
        robot_asset_options.disable_gravity = True
        robot_asset_options.thickness = 0.001
        robot_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        robot_asset_options.use_mesh_materials = True
        robot_asset = self.gym.load_asset(
            self.sim, asset_root, robot_asset_file, robot_asset_options
        )

        robot_dof_damping = to_torch(self.dof_damping, dtype=torch.float, device=self.device)
        robot_dof_stiffness = to_torch(self.dof_stiffness, dtype=torch.float, device=self.device)

        self.num_robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.num_robot_dofs = self.gym.get_asset_dof_count(robot_asset)

        print("Number of robot bodies: ", self.num_robot_bodies)
        print("Number of robot dofs: ", self.num_robot_dofs)

        # set robot dof properties
        robot_dof_properties = self.gym.get_asset_dof_properties(robot_asset)
        self.robot_dof_lower_limits = []
        self.robot_dof_upper_limits = []
        for i in range(self.num_robot_dofs):
            if self.control_type == 0:
                robot_dof_properties["driveMode"][i] = gymapi.DOF_MODE_POS
            elif self.control_type == 1:
                robot_dof_properties["driveMode"][i] = gymapi.DOF_MODE_VEL
            if self.physics_engine == gymapi.SIM_PHYSX:
                robot_dof_properties["damping"][i] = robot_dof_damping[i]
                robot_dof_properties["stiffness"][i] = robot_dof_stiffness[i]
            else:
                raise ValueError("*** Flex Engine is not implemented")

            self.robot_dof_lower_limits.append(robot_dof_properties["lower"][i])
            self.robot_dof_upper_limits.append(robot_dof_properties["upper"][i])

        self.robot_dof_lower_limits = to_torch(self.robot_dof_lower_limits, device=self.device)
        self.robot_dof_upper_limits = to_torch(self.robot_dof_upper_limits, device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)

        self.robot_vel_lower_limits = to_torch(self.dof_vel_lower_limits, device=self.device)
        self.robot_vel_upper_limits = to_torch(self.dof_vel_upper_limits, device=self.device)
        self.robot_vel_lower_limits = torch.deg2rad(self.robot_vel_lower_limits)
        self.robot_vel_upper_limits = torch.deg2rad(self.robot_vel_upper_limits)

        # create obstacle assets
        if self.num_obst > 0:
            ball_options = gymapi.AssetOptions()
            ball_options.density = 400
            ball_options.disable_gravity = True
            obstacle_asset = self.gym.create_sphere(self.sim, self.obst_size, ball_options)

        # placement of the robot, xyz and orientation determining if the robot stands upright
        robot_start_pose = gymapi.Transform()
        robot_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        robot_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 0.707107)

        # cache some common handles for later use
        self.robots = []
        self.robot_indices = []
        self.work_objects = []
        self.obstacles = []
        self.default_obstacle_states = []
        # self.obstacle_start = []
        self.obstacle_indices = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # spawn the robot in a fixed location
            robot_actor = self.gym.create_actor(env_ptr, robot_asset, robot_start_pose, "robot", i, 1)
            self.gym.set_actor_dof_properties(env_ptr, robot_actor, robot_dof_properties)
            robot_idx = self.gym.get_actor_index(env_ptr, robot_actor, gymapi.DOMAIN_SIM)

            # spawn obstacles
            if self.num_obst > 0:
                # self.obstacle_start.append(self.gym.get_sim_actor_count(self.sim))
                obst_count = 0
                obst_ids = torch.ones(self.num_obst, device=self.device)
                for j in range(self.num_obst):
                    obstacle_start_pose = gymapi.Transform()
                    p = randomize_spherical_position(self.spher_radius_low[4], self.spher_radius_up[4],
                                                     self.spher_theta_low[4], self.spher_theta_up[4],
                                                     self.spher_phi_low[4], self.spher_phi_up[4],
                                                     torch.ones(1, device=self.device), self.device).squeeze().tolist()
                    obstacle_start_pose.p = gymapi.Vec3(p[0], p[1], p[2])
                    obstacle_start_pose.r = gymapi.Quat(*randomize_quaternion(obst_ids, self.device)[j])

                    obstacle_actor = self.gym.create_actor(env_ptr, obstacle_asset, obstacle_start_pose, "obst_{}".format(obst_count), 0, 1)
                    obst_count += 1
                    self.default_obstacle_states.append([obstacle_start_pose.p.x, obstacle_start_pose.p.y, obstacle_start_pose.p.z,
                                                         obstacle_start_pose.r.x, obstacle_start_pose.r.y, obstacle_start_pose.r.z, obstacle_start_pose.r.w,
                                                         0, 0, 0, 0, 0, 0])
                    obstacle_idx = self.gym.get_actor_index(env_ptr, obstacle_actor, gymapi.DOMAIN_SIM)
                    self.obstacle_indices.append(obstacle_idx)

            # the handle is appended to the common handles stored in the cache
            self.envs.append(env_ptr)
            self.robots.append(robot_actor)
            self.robot_indices.append(robot_idx)
            # self.work_objects.append(work_object_actor)
            if self.num_obst > 0:
                self.obstacles.append(obstacle_actor)

        link_names = [f"{self.task_name}_link_{j}" for j in range(self.num_robot_dofs + 1)]
        link_names.append(f"{self.task_name}_link_ee")
        self.link_handles = []
        for link in link_names:
            globals()[link] = self.gym.find_actor_rigid_body_handle(env_ptr, robot_actor, link)
            self.link_handles.append(globals()[link])
        self.robot_indices = to_torch(self.robot_indices, dtype=torch.long, device=self.device)

        if self.num_obst > 0:
            self.obst_name = self.gym.find_actor_rigid_body_handle(env_ptr, obstacle_actor, "obst_0")

            self.default_obstacle_states = to_torch(self.default_obstacle_states, dtype=torch.float, device=self.device).view(self.num_envs, self.num_obst, 13) # 13 is the number of elements in the list
            self.obstacle_indices = to_torch(self.obstacle_indices, dtype=torch.long, device=self.device)

        self.init_data()

    def init_data(self):
        # a list containing the radius of each link to subtract it from the distance calculated between the link and the obstacle
        self.link_radius = to_torch(self.link_safety_dist, dtype=torch.float, device=self.device)

        # load csv file for generated curriculum
        if self.curr_modus == 2:
            self.csv_list = [
                name.split("/")[3].split(".")[0]
                for name in sorted(
                    glob.glob(f"../curriculum_data/{self.task_name}_curriculums/*.csv")
                )
            ]
            for name in self.csv_list:
                globals()[name] = torch.tensor(
                    pd.read_csv(
                        f"../curriculum_data/{self.task_name}_curriculums/{name}.csv"
                    ).to_numpy(),
                    dtype=torch.float,
                    device=self.device,
                )
            # assign which curriculum poses should be sampled from
            self.current_curriculum_poses = globals()[self.csv_list[self.curr_level]]

        # send the default joint positions to torch
        if self.curr_modus == 0:
            self.robot_default_dof_pos = to_torch([1.0, 1.0, 0.0, -0.3, 0.0, 0.5, 0.0], device=self.device)
        if self.curr_modus == 1:
            self.robot_default_dof_pos = to_torch([0.0, 1.0, 0.0, -1.0, 0.0, 0.5, 0.0], device=self.device)
        if self.curr_modus == 2:
            self.robot_default_dof_pos = torch.zeros((self.num_dofs), dtype=torch.float, device=self.device)

        self.position_error = torch.zeros((self.num_envs, 1), dtype=torch.float, device=self.device)
        self.orientation_error = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        self.randomized_goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.randomized_goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.randomized_goal_rot[..., -1] = -1

        self.obstacle_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.obstacle_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)

        self.flag = torch.tensor(0, dtype=torch.int, device=self.device)

    def update_curriculum(self):
        # curriculum buffer stores the distance/orientation error when an environment has finished
        # either by reaching the goal or resetting at the limit of 500 steps per episode
        if torch.mean(self.curriculum_buf) <= self.success_tolerance:
                # if the agents was successful and has not trained on all curriculums
                if self.curr_level < self.num_curr_level:
                    # advance to the next curriculum region
                    self.curr_level += 1
                    # reassign which curriculum the poses are sampled from
                    if self.curr_modus == 2 and self.curr_level < self.num_curr_level:
                        self.current_curriculum_poses = globals()[self.csv_list[self.curr_level]]
                # if the agent was successful and has trained on all curriculums
                if self.curr_level == self.num_curr_level and self.success_tolerance > 0.002:
                    if self.flag == 1:
                        # decrease the accuracy threshold to make it harder
                        self.success_tolerance /= 1.99
                        self.curr_level = 5
                    else:
                        self.flag = torch.tensor(1, dtype=torch.int, device=self.device)
                        self.success_tolerance = 0.04
                        # and reset to first curriculum
                        self.curr_level = self.cfg["env"]["curr_level"]
                    # reassign which curriculum the poses are sampled from
                    if self.curr_modus == 2:
                        self.current_curriculum_poses = globals()[self.csv_list[self.curr_level]]
                self.curriculum_buf = torch.zeros_like(self.curriculum_buf, dtype=torch.float, device=self.device)

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:], self.curriculum_buf, self.position_buf[:], self.orientation_buf[:], self.success_rate_buf[:] = compute_reward(
            self.reset_buf, self.progress_buf, self.curriculum_buf,
            self.position_buf, self.orientation_buf, self.success_rate_buf,
            self.position_error.squeeze(), self.distance_reward_scale, self.orientation_error.squeeze(), self.flag,
            self.link_obstacle_distance, self.obstacle_reward_scale, self.obstacle_maximum_distance,
            self.success_tolerance, self.reach_goal_bonus,
            self.num_envs, self.max_episode_length,
        )

        if torch.all(self.curriculum_buf > 0):
            self.rolling_avg_success_rate = ((self.num_rolling_avg - 1) * self.rolling_avg_success_rate + (torch.sum(self.success_rate_buf) / self.num_envs)) / self.num_rolling_avg

            self.update_curriculum()

            self.extras = [
                {"episode": {"curriculum_level": self.curr_level}},
                {"episode": {"success_tolerance": self.success_tolerance}},
                {"episode": {"distance_error": self.position_buf.cpu().numpy()}},
                {"episode": {"orientation_error": self.orientation_buf.cpu().numpy()}},
                {"episode": {"success_rate": self.success_rate_buf.cpu().numpy()}},
                {
                    "episode": {
                        "rolling_avg_success_rate": self.rolling_avg_success_rate.cpu().numpy()
                    }
                },
            ]

    def compute_observations(self):
        # update the contents of the state tensors from the physics engine
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # acquire the refreshed end-effector values from tensors
        self.end_effector_pos = self.rigid_body_states[:, self.link_handles[-1]][:, 0:3]
        self.end_effector_rot = self.rigid_body_states[:, self.link_handles[-1]][:, 3:7]

        # compute distance error
        if self.error_representation == 1:
            self.position_error = compute_position_error(self.end_effector_pos, self.randomized_goal_pos).unsqueeze(-1)
        elif self.error_representation == 3:
            self.position_error = compute_orientation_error(self.x, self.y, self.z,
                                                            self.end_effector_pos, self.end_effector_rot,
                                                            self.randomized_goal_pos, self.randomized_goal_rot)
        elif self.error_representation == 4:
            self.position_error = compute_position_error(self.end_effector_pos, self.randomized_goal_pos).unsqueeze(-1)
            self.orientation_error = compute_orientation_error(self.x, self.y, self.z,
                                                            self.end_effector_pos, self.end_effector_rot,
                                                            self.randomized_goal_pos, self.randomized_goal_rot)
        else:
            raise ValueError("*** Goal representation is ill-defined")

        # acquire joint position and velocity and scale it
        # self.dof_pos_scaled = unscale(self.robot_dof_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        self.dof_pos_scaled = (2.0 * (self.robot_dof_pos - self.robot_dof_lower_limits) / (self.robot_dof_upper_limits - self.robot_dof_lower_limits) - 1.0)
        self.obs_vel = self.robot_dof_vel * self.dof_vel_scale # velocity is scaled by * 0.1

        # run obstacel avoidance
        if self.num_obst > 0:
            link_pos = self.rigid_body_states[:, self.link_handles[3:]][:, :, 0:3]
            obst_pos = self.rigid_body_states[:, self.obst_name][:, 0:3]
            self.link_obstacle_distance = compute_position_error(link_pos, obst_pos[:, None, :]) - self.link_radius - self.obst_size
        else: # set the distance to some large number so the reward is not considered
            self.link_obstacle_distance = torch.ones((self.num_envs, 6), dtype=torch.float, device=self.device)

        self.end_effector_rota = matrix_to_rotation_6d(quaternion_to_matrix(self.end_effector_rot))
        self.randomized_goal_rota = matrix_to_rotation_6d(quaternion_to_matrix(self.randomized_goal_rot))

        self.obs_buf = torch.cat((self.position_error, self.orientation_error,
                                  self.end_effector_pos, self.randomized_goal_pos, # if the pos and rot is not provided in
                                  self.end_effector_rota, self.randomized_goal_rota, # the observation the agent doesn't learn
                                  self.dof_pos_scaled, self.obs_vel,
                                  self.link_obstacle_distance), dim=-1)

        self.debug_visualizer()

        return self.obs_buf

    def reset_robot(self, env_ids):
        # reset robot pos -> dof, vel
        # clamps the joint positions to be within the allowed lower and upper limits
        pos = tensor_clamp(self.robot_default_dof_pos.unsqueeze(0) + 0.25 * (torch.rand((len(env_ids), self.num_robot_dofs), device=self.device) -0.5),
                           self.robot_dof_lower_limits,
                           self.robot_dof_upper_limits)
        # set the position of joint angles as above
        self.robot_dof_pos[env_ids, :] = pos
        # set velocities to zero
        self.robot_dof_vel[env_ids, :] = torch.zeros_like(self.robot_dof_vel[env_ids])
        # set the target position to the same as above
        self.robot_dof_pos_targets[env_ids, :self.num_robot_dofs] = pos

        # reset the robot to its initial target position using an index function
        robot_indices = self.robot_indices[env_ids].to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.robot_dof_pos_targets),
                                                        gymtorch.unwrap_tensor(robot_indices), len(robot_indices))

    def reset_goal_pose(self, env_ids):
        # randomize position
        if self.curr_modus == 0: # spherical
            self.randomized_goal_pos[env_ids, 0:3] = randomize_spherical_position(self.spher_radius_low[self.curr_level], self.spher_radius_up[self.curr_level],
                                                                                  self.spher_theta_low[self.curr_level], self.spher_theta_up[self.curr_level],
                                                                                  self.spher_phi_low[self.curr_level], self.spher_phi_up[self.curr_level],
                                                                                  env_ids, self.device)

        if self.curr_modus == 1: # torus
            self.randomized_goal_pos[env_ids, 0:3] = randomize_torus_position(self.torus_major_radius,
                                                                              self.torus_minor_radius_low[self.curr_level], self.torus_minor_radius_up[self.curr_level],
                                                                              self.torus_theta_low[self.curr_level], self.torus_theta_up[self.curr_level],
                                                                              self.torus_phi_low[self.curr_level], self.torus_phi_up[self.curr_level],
                                                                              env_ids, self.device)

        # randomize orientation
        if self.curr_modus == 0 or self.curr_modus == 1:
            self.randomized_goal_rot[env_ids] = random_quaternions(len(env_ids), dtype=torch.float, device=self.device)

        if self.curr_modus == 2:
            rand_int = torch.randint(0, len(self.current_curriculum_poses), (len(env_ids),))
            poses = self.current_curriculum_poses[rand_int]
            self.randomized_goal_pos[env_ids, 0:3] = poses[:, 0:3]
            self.randomized_goal_rot[env_ids] = poses[:, 3:7]

    def reset_obstacle(self, env_ids):
        self.obstacle_pos[env_ids, 0:3] = randomize_spherical_position(self.spher_radius_low[4], self.spher_radius_up[4],
                                                                       self.spher_theta_low[4], self.spher_theta_up[4],
                                                                       self.spher_phi_low[4], self.spher_phi_up[4],
                                                                       env_ids, self.device)
        self.obstacle_rot[env_ids, 0:4] = randomize_quaternion(env_ids, self.device)
        # a buffer is created for those obstacle positions that are too close to the goal
        obs_ids = torch.where(compute_position_error(self.obstacle_pos[env_ids], self.randomized_goal_pos[env_ids]) <= self.obst_size + 0.2,
                              env_ids, torch.zeros_like(env_ids))
        # the loop is used because sometimes randomizing the position a second time is not enough
        while torch.sum(obs_ids, dim=-1) > 0:
            # generate new positions for the indices that are too close to the goal
            self.obstacle_pos[obs_ids, 0:3] = randomize_spherical_position(self.spher_radius_low[4], self.spher_radius_up[4],
                                                                           self.spher_theta_low[4], self.spher_theta_up[4],
                                                                           self.spher_phi_low[4], self.spher_phi_up[4],
                                                                           obs_ids, self.device)
            obs_ids = torch.where(compute_position_error(self.obstacle_pos[env_ids], self.randomized_goal_pos[env_ids]) <= self.obst_size + 0.2,
                                  obs_ids, torch.zeros_like(obs_ids))

        self.root_state_tensor[env_ids, 1, 0:3] = self.obstacle_pos[env_ids]
        self.root_state_tensor[env_ids, 1, 3:7] = self.obstacle_rot[env_ids]
        obstacle_indices = self.obstacle_indices[env_ids].to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(obstacle_indices), len(env_ids))

    def reset(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        self.reset_robot(env_ids)

        # calculate new poses for goal
        self.reset_goal_pose(env_ids)

        # reset position of obstacle
        if self.num_obst > 0:
            self.reset_obstacle(env_ids)

        # reset the progress and reset buffer
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        if self.control_type == 0:
            targets = self.robot_dof_pos_targets[:, :self.num_robot_dofs] + self.robot_dof_speed_scales * self.dt * self.actions * self.action_scale
            self.robot_dof_pos_targets[:, :self.num_robot_dofs] = tensor_clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.robot_dof_pos_targets))

        if self.control_type == 1:
            self.robot_dof_vel_targets = (self.robot_vel_upper_limits - self.robot_vel_lower_limits) / (1 - (-1)) * self.actions + (self.robot_vel_lower_limits * 1 - self.robot_vel_upper_limits * (-1)) / (1 - (-1))
            self.robot_dof_vel_targets /= 4

            self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(self.robot_dof_vel_targets))


    def post_physics_step(self):
        # progress buffer that counts upwards until reset() is called
        self.progress_buf += 1  # the counter is iterated for every 10 steps
        self.randomize_buf += 1

        # copy indices of environments that needs to be reset
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset(env_ids)

        self.compute_observations()
        self.compute_reward()

    def debug_visualizer(self):
        # debug visualizer
        if self.viewer and self.debug_vis:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                # draw frame for the end-effector
                px = (self.end_effector_pos[i] + quat_apply(self.end_effector_rot[i], self.x[0] * 0.2)).cpu().numpy()
                py = (self.end_effector_pos[i] + quat_apply(self.end_effector_rot[i], self.y[0] * 0.2)).cpu().numpy()
                pz = (self.end_effector_pos[i] + quat_apply(self.end_effector_rot[i], self.z[0] * 0.2)).cpu().numpy()
                p0 = self.end_effector_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                # draw frame for the goal
                px = (self.randomized_goal_pos[i] + quat_apply(self.randomized_goal_rot[i], self.x[0] * 0.2)).cpu().numpy()
                py = (self.randomized_goal_pos[i] + quat_apply(self.randomized_goal_rot[i], self.y[0] * 0.2)).cpu().numpy()
                pz = (self.randomized_goal_pos[i] + quat_apply(self.randomized_goal_rot[i], self.z[0] * 0.2)).cpu().numpy()
                p0 = self.randomized_goal_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])



#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_reward(  # any argument without any attribution is a tensor
    reset_buf,
    progress_buf,
    curriculum_buf,
    position_buf,
    orientation_buf,
    success_rate_buf,
    position_error,
    distance_reward_scale: float,
    orientation_error,
    flag,
    link_obstacle_distance,
    obstacle_reward_scale: float,
    obstacle_maximum_distance: float,
    success_tolerance: float,
    reach_goal_bonus: float,
    num_envs: int,
    max_episode_length: float,
):
    # sum the rows if and only if the corrent orientation is trying to be reached
    orientation_error = torch.sum(orientation_error, dim=-1)
    if flag == 1:
        error = orientation_error
    else:
        error = position_error
    dist_reward = torch.exp(distance_reward_scale * torch.abs(error) ** 2)

    # first any negative distance values are converted to zero, secondly, all values greater than the maximum distance that incurs a penalty
    # are converted to zero and then the reward is summed for each environment
    obstacle_unfix = torch.where(link_obstacle_distance <= 0, torch.zeros_like(link_obstacle_distance), link_obstacle_distance)
    obstacle_penalty = torch.maximum(torch.zeros_like(link_obstacle_distance), 1 - (obstacle_unfix / obstacle_maximum_distance))
    obst_reward = -obstacle_reward_scale * torch.sum(obstacle_penalty, dim=-1)

    rewards = dist_reward + obst_reward

    # adjust reward if inside success tolerance of goal
    rewards = torch.where(error <= success_tolerance, rewards + reach_goal_bonus, rewards)

    # reset agents
    # the reset buffer is used to reset only the environments which have reached the goal, collided with an obstacle or such
    reset_buf = torch.where(error <= success_tolerance, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length, torch.ones_like(reset_buf), reset_buf)

    # buffer is used to refresh when the curriculum is changed
    curriculum_buf = torch.where(reset_buf == 1, error, curriculum_buf)

    position_buf = torch.where(reset_buf == 1, position_error, position_buf)
    orientation_buf = torch.where(reset_buf == 1, orientation_error, orientation_buf)
    success_rate_buf = torch.where(error <= success_tolerance, torch.ones_like(success_rate_buf), success_rate_buf)
    success_rate_buf = torch.where(progress_buf >= max_episode_length, torch.zeros_like(success_rate_buf), success_rate_buf)

    return rewards, reset_buf, curriculum_buf, position_buf, orientation_buf, success_rate_buf

@torch.jit.script
def compute_position_error(end_effector, goal):
    # type: (Tensor, Tensor) -> Tensor
    # euclidean distance/L2 loss
    return torch.norm(end_effector - goal, p=2, dim=-1)

@torch.jit.script
def compute_orientation_error(x, y, z, ee_pos, ee_rot, goal_pos, goal_rot):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    # compute three orthogonal axes
    ee_x = ee_pos + quat_apply(ee_rot, x * 0.1)
    ee_y = ee_pos + quat_apply(ee_rot, y * 0.1)
    ee_z = ee_pos + quat_apply(ee_rot, z * 0.1)

    goal_x = goal_pos + quat_apply(goal_rot, x * 0.1)
    goal_y = goal_pos + quat_apply(goal_rot, y * 0.1)
    goal_z = goal_pos + quat_apply(goal_rot, z * 0.1)

    x_error = torch.norm(ee_x - goal_x, p=2, dim=-1)
    y_error = torch.norm(ee_y - goal_y, p=2, dim=-1)
    z_error = torch.norm(ee_z - goal_z, p=2, dim=-1)

    return torch.stack([x_error, y_error, z_error], dim=-1).squeeze(1)


@torch.jit.script
def randomize_spherical_position(radius_low: float, radius_up: float, theta_low: float,
                                 theta_up: float, phi_low: float, phi_up: float, env_ids, device: str):
    # offset above ground is added
    z_offset = 0.2
    # pi/2 is substracted to get the right placement in front of the robot
    rot_offset = np.pi / 2

    radius = torch_rand_float(radius_low, radius_up, (len(env_ids), 1), device=device)
    theta = torch_rand_float(theta_low, theta_up, (len(env_ids), 1), device=device)
    phi = torch_rand_float(phi_low, phi_up, (len(env_ids), 1), device=device) - rot_offset
    x = radius * torch.sin(theta) * torch.cos(phi)
    y = radius * torch.sin(theta) * torch.sin(phi)
    z = radius * torch.cos(theta) + z_offset

    return torch.stack([x, y, z], dim=-1).squeeze(1)


@torch.jit.script
def randomize_torus_position(major_radius: float, minor_radius_low: float, minor_radius_up: float, theta_low: float,
                             theta_up: float, phi_low: float, phi_up: float, env_ids, device: str):
    # offset above ground is added
    z_offset = 0.5
    # pi/2 is substracted to get the right placement in front of the robot
    rot_offset = np.pi / 2

    minor_radius = torch_rand_float(minor_radius_low, minor_radius_up, (len(env_ids), 1), device=device)
    theta = torch_rand_float(theta_low, theta_up, (len(env_ids), 1), device=device)
    phi = torch_rand_float(phi_low, phi_up, (len(env_ids), 1), device=device) - rot_offset
    x = (major_radius + minor_radius * torch.cos(theta)) * torch.cos(phi)
    y = (major_radius + minor_radius * torch.cos(theta)) * torch.sin(phi)
    z = minor_radius * torch.sin(theta) + z_offset

    return torch.stack([x, y, z], dim=-1).squeeze(1)


@torch.jit.script
def randomize_quaternion(env_ids, device: str):
    rand = torch_rand_float(0.0, 1.0, (len(env_ids), 3), device=device)

    q1 = torch.sqrt(1 - rand[:, 0]) * torch.sin(2 * np.pi * rand[:, 1])
    q2 = torch.sqrt(1 - rand[:, 0]) * torch.cos(2 * np.pi * rand[:, 1])
    q3 = torch.sqrt(rand[:, 1]) * torch.sin(2 * np.pi * rand[:, 2])
    q4 = torch.sqrt(rand[:, 1]) * torch.cos(2 * np.pi * rand[:, 2])

    return torch.stack([q1, q2, q3, q4], dim=-1)
