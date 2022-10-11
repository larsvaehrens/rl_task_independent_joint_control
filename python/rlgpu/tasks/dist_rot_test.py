import math

import numpy as np
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import (
    quat_apply,
    quat_conjugate,
    quat_mul,
    to_torch,
    torch_rand_float,
)
from rlgpu.utils.config import (
    get_args,
    load_cfg,
    parse_sim_params,
    set_np_formatting,
    set_seed,
)
import torch


class DistRotTest:
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
        self.x = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

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

    def get_pos(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.end_effector_pos = self.rigid_body_states[:, self.link_handles[-1]][:, 0:3]
        self.end_effector_rot = self.rigid_body_states[:, self.link_handles[-1]][:, 3:7]

        # set position of the randomized goal
        self.goal_pos = to_torch([-0.0007,  0.4547,  0.4003], dtype=torch.float, device=self.device)
        # set rotation of the goal
        if self.args.random_goal == 1:
            self.goal_rot = to_torch([0.4999, -0.5, 0.5001, 0.5], dtype=torch.float, device=self.device)
        if self.args.random_goal == 2:
            self.goal_rot = to_torch([0.4999, -0.75, 0.5001, 0.25], dtype=torch.float, device=self.device)
        if self.args.random_goal == 3:
            self.goal_rot = to_torch([0.4999, -1.0, 0.5001, 0.0], dtype=torch.float, device=self.device)
        if self.args.random_goal == 4:
            self.goal_rot = to_torch([1, 0, 0.0, 0.0], dtype=torch.float, device=self.device)

    def draw_coordinate_frames(self):
        for i in range(self.num_envs):
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            # draw frame for the end-effector
            px = (self.end_effector_pos[i] + quat_apply(self.end_effector_rot[i], self.x[0] * 0.2)).cpu().numpy()
            py = (self.end_effector_pos[i] + quat_apply(self.end_effector_rot[i], self.y[0] * 0.2)).cpu().numpy()
            pz = (self.end_effector_pos[i] + quat_apply(self.end_effector_rot[i], self.z[0] * 0.2)).cpu().numpy()
            p0 = self.end_effector_pos[i].cpu().numpy()
            self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
            self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
            self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

            # draw frame for the goal
            px = (self.goal_pos + quat_apply(self.goal_rot, self.x[0] * 0.2)).cpu().numpy()
            py = (self.goal_pos + quat_apply(self.goal_rot, self.y[0] * 0.2)).cpu().numpy()
            pz = (self.goal_pos + quat_apply(self.goal_rot, self.z[0] * 0.2)).cpu().numpy()
            p0 = self.goal_pos.cpu().numpy()
            self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
            self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
            self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

    def calculate_error_val(self):
        # euclidean distance torch function (L2 loss)
        pos_dist_1 = torch.norm(self.end_effector_pos - self.goal_pos, p=2, dim=-1)

        # orientation error
        quat_diff = quat_mul(self.end_effector_rot[0], quat_conjugate(self.goal_rot))
        rot_dist = 2.0 * torch.asin(
            torch.clamp(torch.norm(quat_diff[0:3], p=2, dim=-1), max=1.0)
        )

        print(f"position: {pos_dist_1[0]} centimeters, rotation: {rot_dist}")

    def run_dist_rot_test(self):
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
            self.get_pos()

            # draw coordinate frames for visualization
            if not self.args.headless:
                self.draw_coordinate_frames()

            # calculate error value of current and desired position and rotation
            self.calculate_error_val()

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

    dist_rot = DistRotTest(
        args=args,
        cfg=cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        device_type=args.device,
        device_id=args.device_id,
        headless=args.headless,
    )

    dist_rot.run_dist_rot_test()
