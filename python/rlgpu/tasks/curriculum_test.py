import glob
import math
import os

import numpy as np
import pandas as pd
import scipy.spatial as spatial
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import to_torch, torch_rand_float
from rlgpu.utils.config import (
    get_args,
    load_cfg,
    parse_sim_params,
    set_np_formatting,
    set_seed,
)
import torch


class CurriculumTest:
    def __init__(
        self, args, cfg, sim_params, physics_engine, device_type, device_id, headless
    ):
        self.gym = gymapi.acquire_gym()
        self.args = args
        self.cfg = cfg
        self.headless = headless
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.device_type = device_type
        self.device_id = device_id
        self.device = "cpu"
        if self.device_type == "cuda" or self.device_type == "GPU":
            self.device = "cuda" + ":" + str(self.device_id)
        self.task_name = self.args.task.lower()

        self.spacing = 0.75  # self.cfg["env"]["env_spacing"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.visual_attachments = self.cfg["robot"]["visual_attachments"]
        self.dof_damping = self.cfg["robot"]["dof_damping"]
        self.dof_stiffness = self.cfg["robot"]["dof_stiffness"]

        self.num_dofs = self.cfg["env"]["num_dofs"]

        self.num_envs = self.args.num_envs

        self.rot_offset = np.pi / 2

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

        self.workspace_size = self.cfg["robot"]["workspace_size"]
        self.curriculum_sizes = self.cfg["robot"]["curriculum_sizes"]

        self.iter = [50, 75, 100, 200, 300, 25, 60, 75, 100, 100]

        self.load_generated_poses()

        self.create_sim()

    def load_generated_poses(self):
        path = f"../curriculum_data/{self.task_name}_raw_data/"
        all_files = glob.glob(os.path.join(path, "*.csv"))
        df = pd.concat(
            (pd.read_csv(f, header=None, index_col=False) for f in all_files)
        ).to_numpy()

        raw_size = df.shape[0]
        point_tree = spatial.cKDTree(df[:, [0, 1, 2]])
        df_indices = point_tree.query_ball_point([0, 0, 0], self.workspace_size)
        df = pd.DataFrame(df[df_indices])

        raw_size = df.shape[0]
        df = df[df[2] >= 0.2]

        df = df.to_numpy()

        self.point = [np.max(df[:, 0]) / 2, 0, np.max(df[:, 2]) / 2]

    def create_sim(self):
        # allocates which device will simulate and which device will render the scene and simulation type to be used
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim_params.use_gpu_pipeline = False
        if args.use_gpu_pipeline:
            print("WARNING: Forcing CPU pipeline.")
        self.sim = self.gym.create_sim(
            self.device_id, self.device_id, self.physics_engine, self.sim_params
        )
        if self.sim is None:
            print("*** Failed to create sim")
            quit()
        self._create_ground_plane()
        self._create_envs(spacing=self.spacing, num_per_row=5)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

        # create viewer
        if self.headless == False:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            cam_pos = gymapi.Vec3(np.sqrt(self.num_envs) * self.spacing * 2, np.sqrt(self.num_envs) * self.spacing * 2, 8.0)
            cam_target = gymapi.Vec3(np.sqrt(self.num_envs), np.sqrt(self.num_envs), -8.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
            if self.viewer is None:
                print("*** Failed to create viewer")
                quit()

    def _create_envs(self, spacing, num_per_row):
        # set up the env grid (it is a 3-dimensional box constructed by the lower and upper)
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # load robot assets
        asset_root = "../../assets"
        robot_asset_file = f"urdf/{self.task_name}/{self.task_name}.urdf"

        robot_asset_options = gymapi.AssetOptions()
        robot_asset_options.fix_base_link = True
        robot_asset_options.flip_visual_attachments = self.visual_attachments
        robot_asset_options.collapse_fixed_joints = False
        robot_asset_options.disable_gravity = True
        robot_asset_options.thickness = 0.001
        robot_asset_options.use_mesh_materials = True

        print("Loading asset '%s' from '%s'" % (robot_asset_file, asset_root))
        robot_asset = self.gym.load_asset(
            self.sim, asset_root, robot_asset_file, robot_asset_options
        )

        robot_dof_stiffness = to_torch(self.dof_stiffness, dtype=torch.float)
        robot_dof_damping = to_torch(self.dof_damping, dtype=torch.float)
        num_robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        num_robot_dofs = self.gym.get_asset_dof_count(robot_asset)

        robot_dof_properties = self.gym.get_asset_dof_properties(robot_asset)

        self.robot_dof_lower_limits = []
        self.robot_dof_upper_limits = []

        for i in range(num_robot_dofs):
            robot_dof_properties["driveMode"][i] = gymapi.DOF_MODE_POS

            robot_dof_properties["stiffness"][i] = robot_dof_stiffness[i]
            robot_dof_properties["damping"][i] = robot_dof_damping[i]

            self.robot_dof_lower_limits.append(robot_dof_properties["lower"][i])
            self.robot_dof_upper_limits.append(robot_dof_properties["upper"][i])

        self.robot_dof_lower_limits = to_torch(self.robot_dof_lower_limits)
        self.robot_dof_upper_limits = to_torch(self.robot_dof_upper_limits)

        # placement of the robot, xyz and quaternions determining if the robot stands upright
        robot_start_pose = gymapi.Transform()
        robot_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        robot_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 0.707107)

        # configure ball properties and create asset
        ball_radius = 0.01
        ball_options = gymapi.AssetOptions()
        ball_options.density = 200
        ball_options.disable_gravity = True
        self.ball_asset = self.gym.create_sphere(self.sim, ball_radius, ball_options)

        # create the environment
        print("Creating %d environments" % self.num_envs)

        self.envs = []
        self.robots = []

        for i in range(self.num_envs):
            # create env
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env_ptr)

            # spawn the robot in a fixed location
            robot_actor = self.gym.create_actor(
                env_ptr, robot_asset, robot_start_pose, "robot", i, 1
            )
            self.gym.set_actor_dof_properties(
                env_ptr, robot_actor, robot_dof_properties
            )
            self.robots.append(robot_actor)

            if i == 0 or i == 1 or i == 2 or i == 3 or i == 4:
                self.spawn_torus(k=i)
            if i == 5 or i == 6 or i == 7 or i == 8 or i == 9:
                self.spawn_spherical(k=i)
            if i == 10 or i == 11 or i == 12 or i == 13 or i == 14 or i == 15:
                self.spawn_generated_poses(k=i)

        link_names = [f"{self.task_name}_link_{j}" for j in range(num_robot_dofs + 1)]
        link_names.append(f"{self.task_name}_link_ee")
        self.link_handles = []
        for i in link_names:
            globals()[i] = self.gym.find_actor_rigid_body_handle(
                env_ptr, robot_actor, i
            )
            self.link_handles.append(globals()[i])

        joint_names = [f"{self.task_name}_link_{j}" for j in range(num_robot_dofs)]
        self.joint_handles = []
        for i in joint_names:
            globals()[i] = self.gym.find_actor_dof_handle(env_ptr, robot_actor, i)
            self.joint_handles.append(globals()[i])

        self.init_data()

    def init_data(self):
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.robot_dof_pos_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float
        )

    def spawn_torus(self, k):
        # a function to spawn the torus curriculum
        if self.args.interior:
            minor_radius_c = np.random.uniform(self.torus_minor_radius_low[k], self.torus_minor_radius_up[k], self.iter[k])
            theta_c = np.random.uniform(self.torus_theta_low[k], self.torus_theta_up[k], self.iter[k])
            phi_c = np.random.uniform(self.torus_phi_low[k], self.torus_phi_up[k], self.iter[k]) - self.rot_offset
            x = (self.torus_major_radius + minor_radius_c * np.cos(theta_c)) * np.cos(phi_c)
            y = (self.torus_major_radius + minor_radius_c * np.cos(theta_c)) * np.sin(phi_c)
            z = minor_radius_c * np.sin(theta_c) + 0.5
            for h in range(self.iter[k]):
                ball_pose = gymapi.Transform()
                ball_pose.p = gymapi.Vec3(x[h], y[h], z[h])
                ball_handle = self.gym.create_actor(self.envs[k], self.ball_asset, ball_pose, "ball", 0, 1)
                self.gym.set_rigid_body_color(self.envs[k], ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1.0, 0.0, 0.0))

        elif self.args.boundary:
            num_phi = int(np.round(np.degrees(self.torus_phi_up[k] / 10)))
            num_theta = int(np.round(np.degrees(self.torus_theta_up[k] / 10)))
            phi = np.linspace(0, self.torus_phi_up[k], num=num_phi) - self.rot_offset
            theta = np.linspace(0, self.torus_theta_up[k], num=num_theta)
            for u in range(num_phi):
                for h in range(num_theta):
                    x = (self.torus_major_radius + self.torus_minor_radius_up[k] * np.cos(theta[h])) * np.cos(phi[u])
                    y = (self.torus_major_radius + self.torus_minor_radius_up[k] * np.cos(theta[h])) * np.sin(phi[u])
                    z = self.torus_minor_radius_up[k] * np.sin(theta[h]) + 0.5
                    ball_pose = gymapi.Transform()
                    ball_pose.p = gymapi.Vec3(x, y, z)
                    ball_handle = self.gym.create_actor(self.envs[k], self.ball_asset, ball_pose, "ball", 0, 1)
                    self.gym.set_rigid_body_color(self.envs[k], ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.0, 1.0, 0.0))

        else:
            raise ValueError("*** You must select either --interior or --boundary")

    def spawn_spherical(self, k):
        # a function to spawn the spherical curriculum
        if self.args.interior:
            radius_c = np.random.uniform(self.spher_radius_low[k-5], self.spher_radius_up[k-5], self.iter[k])
            phi_c = np.random.uniform(self.spher_phi_low[k-5], self.spher_phi_up[k-5], self.iter[k]) - self.rot_offset
            theta_c = np.random.uniform(self.spher_theta_low[k-5], self.spher_theta_up[k-5], self.iter[k])
            x = radius_c * np.sin(theta_c) * np.cos(phi_c)
            y = radius_c * np.sin(theta_c) * np.sin(phi_c)
            z = radius_c * np.cos(theta_c) + 0.1
            for h in range(self.iter[k - 5]):
                ball_pose = gymapi.Transform()
                ball_pose.p = gymapi.Vec3(x[h], y[h], z[h])
                ball_handle = self.gym.create_actor(self.envs[k], self.ball_asset, ball_pose, "ball", 0, 1)
                self.gym.set_rigid_body_color(self.envs[k], ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1.0, 0.0, 0.0))

        elif self.args.boundary:
            num_phi = int(np.round(np.degrees((2 * self.spher_phi_up[k - 5]) / 10)))
            num_theta = int(np.round(np.degrees((2 * self.spher_theta_up[k - 5]) / 10)))
            radius = np.linspace(self.spher_radius_low[k - 5], self.spher_radius_up[k - 5], 2)
            phi = np.linspace(self.spher_phi_low[k - 5], self.spher_phi_up[k - 5], num=num_phi) - self.rot_offset
            theta = np.linspace(self.spher_theta_low[k - 5], self.spher_theta_up[k - 5], num=num_theta)
            for w in range(2):
                for u in range(num_phi):
                    for h in range(num_theta):
                        x = radius[w] * np.sin(theta[h]) * np.cos(phi[u])
                        y = radius[w] * np.sin(theta[h]) * np.sin(phi[u])
                        z = radius[w] * np.cos(theta[h]) + 0.1
                        ball_pose = gymapi.Transform()
                        ball_pose.p = gymapi.Vec3(x, y, z)
                        ball_handle = self.gym.create_actor(self.envs[k], self.ball_asset, ball_pose, "ball", 0, 1)
                        self.gym.set_rigid_body_color(self.envs[k], ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.0, 1.0, 0.0))

        else:
            raise ValueError("*** You must select either --interior or --boundary")

    def spawn_generated_poses(self, k):
        # a function to spawn the generated poses curriculum
        sphere_num = [10, 16, 20, 24, 28, 36]

        if self.args.interior:
            radius_c = np.random.uniform(0, self.curriculum_sizes[k-10], self.iter[k-5])
            phi_c = np.random.uniform(0, np.pi*2, self.iter[k-5]) - self.rot_offset
            theta_c = np.random.uniform(0, np.pi*2, self.iter[k-5])
            x = radius_c * np.sin(theta_c) * np.cos(phi_c) + self.point[0]
            y = radius_c * np.sin(theta_c) * np.sin(phi_c) + self.point[1]
            z = radius_c * np.cos(theta_c) + self.point[2]
            for h in range(self.iter[k-5]):
                ball_pose = gymapi.Transform()
                ball_pose.p = gymapi.Vec3(x[h], y[h], z[h])
                ball_handle = self.gym.create_actor(self.envs[k], self.ball_asset, ball_pose, "ball", 0, 1)
                self.gym.set_rigid_body_color(self.envs[k], ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1.0, 0.0, 0.0))

        elif self.args.boundary:
            num_phi = sphere_num[k - 10]
            num_theta = sphere_num[k - 10]
            radius = self.curriculum_sizes[k - 10]
            phi = np.linspace(0, np.pi * 2, num=num_phi) - self.rot_offset
            theta = np.linspace(0, np.pi * 2, num=num_theta)
            for u in range(num_phi):
                for h in range(num_theta):
                    x = radius * np.sin(theta[h]) * np.cos(phi[u]) + self.point[0]
                    y = radius * np.sin(theta[h]) * np.sin(phi[u]) + self.point[1]
                    z = radius * np.cos(theta[h]) + self.point[2]
                    ball_pose = gymapi.Transform()
                    ball_pose.p = gymapi.Vec3(x, y, z)
                    ball_handle = self.gym.create_actor(self.envs[k], self.ball_asset, ball_pose, "ball", 0, 1)
                    self.gym.set_rigid_body_color(self.envs[k], ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.0, 1.0, 0.0))

        else:
            raise ValueError("*** You msut select either --inerior or --boundary")

    def joint_control(self):
        for i in range(self.num_dofs):
            self.robot_dof_pos_targets[:, i] = torch_rand_float(
                self.robot_dof_lower_limits[i],
                self.robot_dof_upper_limits[i],
                (self.num_envs, 1),
                device=self.device,
            ).squeeze()

        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.robot_dof_pos_targets)
        )

    def run_curriculum_test(self):
        next_update_time = 1
        frame = 0

        for k in range(1000000):
            self.gym.clear_lines(self.viewer)

            # check if we should update
            t = self.gym.get_sim_time(self.sim)

            if t >= next_update_time:
                # move joints
                self.joint_control()

                next_update_time += 4

            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # step the rendering
            if not self.args.headless:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, False)
                self.gym.sync_frame_time(self.sim)

            frame += 1

        print("Done")

        if not self.args.headless:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy.sim(self.sim)


if __name__ == "__main__":
    # configures the print options for numpy objects in the terminal
    set_np_formatting()
    # gets the command line arguments and parses them
    args = get_args()
    # manual overwrite because it only runs on the CPU and not GPU
    args.device = "CPU"
    # manual overwrite to always spawn 10 environments for all the different curriculums
    # args.num_envs = 10
    # device used for pytorch to store tensors
    sim_device = "cpu"
    # grabs information from the two yaml files defining properties such as num_envs, physics engine, neural network properties
    cfg, cfg_train, logdir = load_cfg(args)
    # used for initializing the simulation parameters
    sim_params = parse_sim_params(args, cfg, cfg_train)
    # seed for random.seed, np.random.seed, and torch.manual_seed
    set_seed(cfg_train["seed"])

    cur = CurriculumTest(
        args=args,
        cfg=cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        device_type=args.device,
        device_id=args.device_id,
        headless=args.headless,
    )

    cur.run_curriculum_test()
