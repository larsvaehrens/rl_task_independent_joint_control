import numpy as np
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_apply, to_torch, torch_rand_float
from rlgpu.utils.config import (
    get_args,
    load_cfg,
    parse_sim_params,
    set_np_formatting,
    set_seed,
)
import torch


class CollisionTest:
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

        self.spacing = self.cfg["env"]["env_spacing"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.visual_attachments = self.cfg["robot"]["visual_attachments"]
        self.dof_damping = self.cfg["robot"]["dof_damping"]
        self.dof_stiffness = self.cfg["robot"]["dof_stiffness"]

        self.link_safety_distance = to_torch(self.cfg["robot"]["link_safety_dist"])

        self.num_dofs = self.cfg["env"]["num_dofs"]

        self.num_envs = self.args.num_envs

        # unit vectors used for rotations
        self.x = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.y = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.z = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat(
            (self.num_envs, 1)
        )

        self.create_sim()

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
        self._create_envs(spacing=self.spacing, num_per_row=int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

        # create viewer
        if self.headless == False:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            cam_pos = gymapi.Vec3(
                np.sqrt(self.num_envs) * self.spacing * 2,
                np.sqrt(self.num_envs) * self.spacing * 2,
                8.0,
            )
            cam_target = gymapi.Vec3(
                np.sqrt(self.num_envs), np.sqrt(self.num_envs), -8.0
            )
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
        ball_radius = 0.001
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

            # draw the visualization for how collision is checked
            # self.draw_spheres_around_joint(i, env_ptr)

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
        # acquire the rigid body tensor, wrap the tensor, and
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(
            self.num_envs, -1, 13
        )

        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # acquire the refreshed end-effector values from tensors
        self.end_effector_pos = self.rigid_body_states[:, self.link_handles[-1]][:, 0:3]
        self.end_effector_rot = self.rigid_body_states[:, self.link_handles[-1]][:, 3:7]

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.robot_dof_pos_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float
        )

    def draw_spheres_around_joint(self):
        joints = self.rigid_body_states[:, 3:9][:, 0:3]
        num_balls = 10
        i = 0

        for n in range(len(self.link_safety_distance)):
            radius = to_torch(self.link_safety_distance[n], device=self.device)
            phi = torch.linspace(0, 2 * np.pi, num_balls, device=self.device)
            theta = torch.linspace(0, np.pi, num_balls, device=self.device)

            for h in range(num_balls):
                for k in range(num_balls):
                    x = radius * torch.sin(theta[k]) * torch.cos(phi[h]) + joints[:, 0, n + 3]
                    y = radius * torch.sin(theta[k]) * torch.sin(phi[h]) + joints[:, 1, n + 3]
                    z = radius * torch.cos(theta[k]) + joints[:, 2, n + 3]
                    px = (x + quat_apply(to_torch([1, 0, 0, 0], device=self.device), self.x[0] * 0.01)).cpu().numpy()
                    py = (y + quat_apply(to_torch([0, 1, 0, 0], device=self.device), self.y[0] * 0.01)).cpu().numpy()
                    pz = (z + quat_apply(to_torch([0, 0, 1, 0], device=self.device), self.z[0] * 0.01)).cpu().numpy()
                    p0 = [x, y, z]
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])



    def joint_control(self):
        for i in range(self.num_dofs):
            self.robot_dof_pos_targets[:, i] = torch_rand_float(self.robot_dof_lower_limits[i], self.robot_dof_upper_limits[i], (self.num_envs, 1), device=self.device).squeeze()

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.robot_dof_pos_targets))

    def check_collision(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        for i in range(self.num_envs):
            # we skip the first joint (which is the second body in the urdf and rigid_body_states)
            for j in range(6):
                end_effector_pos = self.rigid_body_states[:, j + 3][:, 0:3]
                end_effector_rot = self.rigid_body_states[:, j + 3][:, 3:7]

                radius = to_torch(0.2, device=self.device)
                phi = torch.linspace(0, 2 * np.pi, 25, device=self.device)
                theta = torch.linspace(0, np.pi, 25, device=self.device)

                x = radius * torch.sin(theta) * torch.cos(phi)
                y = radius * torch.sin(theta) * torch.sin(phi)
                z = radius * torch.cos(theta)

                x += end_effector_pos[i, 0]
                y += end_effector_pos[i, 1]
                z += end_effector_pos[i, 2]

    def draw_coordinate_frames(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # draw frames for each joint 2 to 7
        for i in range(self.num_envs):
            for j in range(6):
                end_effector_pos = self.rigid_body_states[:, j + 3][:, 0:3]
                end_effector_rot = self.rigid_body_states[:, j + 3][:, 3:7]
                px = (end_effector_pos[i] + quat_apply(end_effector_rot[i], self.x[0] * 0.2)).cpu().numpy()
                py = (end_effector_pos[i] + quat_apply(end_effector_rot[i], self.y[0] * 0.2)).cpu().numpy()
                pz = (end_effector_pos[i] + quat_apply(end_effector_rot[i], self.z[0] * 0.2)).cpu().numpy()
                p0 = end_effector_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

    def run_collision_test(self):
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

            # read end-effector pos and rot
            self.check_collision()

            # draw coordinate frames for visualization
            if not self.args.headless:
                # self.draw_coordinate_frames()
                self.draw_spheres_around_joint()

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
    # device used for pytorch to store tensors
    sim_device = "cpu"
    # grabs information from the two yaml files defining properties such as num_envs, physics engine, neural network properties
    cfg, cfg_train, logdir = load_cfg(args)
    # used for initializing the simulation parameters
    sim_params = parse_sim_params(args, cfg, cfg_train)
    # seed for random.seed, np.random.seed, and torch.manual_seed
    set_seed(cfg_train["seed"])

    col = CollisionTest(
        args=args,
        cfg=cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        device_type=args.device,
        device_id=args.device_id,
        headless=args.headless,
    )

    col.run_collision_test()
