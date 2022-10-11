import glob
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


class GeneratePoses:
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

        self.num_dofs = self.cfg["env"]["num_dofs"]

        self.workspace_size = self.cfg["robot"]["workspace_size"]
        self.curriculum_sizes = self.cfg["robot"]["curriculum_sizes"]

        if not self.args.record and not self.args.split:
            raise ValueError(" *** Need to select either --record or --split")

        if self.args.record and not self.args.num_samples:
            raise ValueError(
                " *** Need to specify the --num_samples to record (1 = 1 million samples)"
            )

        if self.args.split and not self.args.headless:
            raise ValueError(
                " *** Need to be in headless mode to split the data into curriculums"
            )

        self.num_envs = self.args.num_envs

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

        envs = []
        robots = []

        for i in range(self.num_envs):
            # create env
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            envs.append(env_ptr)

            # spawn the robot in a fixed location
            robot_actor = self.gym.create_actor(
                env_ptr, robot_asset, robot_start_pose, "robot", i, 1
            )
            self.gym.set_actor_dof_properties(
                env_ptr, robot_actor, robot_dof_properties
            )
            robots.append(robot_actor)

        link_names = [f"{self.task_name}_link_{j}" for j in range(num_robot_dofs + 1)]
        link_names.append(f"{self.task_name}_link_ee")
        self.link_handles = []
        for i in link_names:
            globals()[i] = self.gym.find_actor_rigid_body_handle(
                env_ptr, robot_actor, i
            )
            self.link_handles.append(globals()[i])

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

    def update_ee(self):
        # define the target and ensure it is within the kinematic constraints
        for i in range(self.num_dofs):
            self.robot_dof_pos_targets[:, i] = torch_rand_float(
                self.robot_dof_lower_limits[i],
                self.robot_dof_upper_limits[i],
                (self.num_envs, 1),
                device=self.device,
            ).squeeze()
        # print(robot_dof_pos_targets)
        # move the robot to the position
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.robot_dof_pos_targets)
        )

        # try if this method moves the joints instantaneously
        # might need some velocity also according to file:///home/vrt/Downloads/RL_joint_generation/docs/api/python/gym_py.html?highlight=indexed#isaacgym.gymapi.Gym.set_dof_state_tensor
        # self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.robot_dof_pos_targets))

    def get_ee_pose(self, loop_iter):
        # refresh the tensor data to acquire the pose of the end-effector
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.end_effector_pose = self.rigid_body_states[:, self.link_handles[-1]][:, 0:7]
        # append this to the matrix that is to be written to the csv
        # print(self.end_effector_pose)

        self.storage = torch.cat([self.storage, self.end_effector_pose], dim=0)
        print(
            f"Recording number {loop_iter} out of {self.args.num_samples} - Current length of vector: {len(self.storage)}"
        )
        # print(f'Time: {timeit.default_timer() - self.start:.2f} seconds')
        # self.start = timeit.default_timer()

    def record(self):
        next_update_time = 1
        frame = 0

        for i in range(self.args.num_samples):
            # self.start = timeit.default_timer()
            self.storage = torch.zeros((1, 7), dtype=torch.float)

            # while not self.gym.query_viewer_has_closed(self.viewer):
            while len(self.storage) < 1000000:
                # check if we should update
                t = self.gym.get_sim_time(self.sim)

                if t >= next_update_time:
                    self.update_ee()
                    next_update_time += 4

                if t > 5:
                    self.get_ee_pose(i)

                # step the physics
                self.gym.simulate(self.sim)
                self.gym.fetch_results(self.sim, True)

                # step rendering
                if not self.args.headless:
                    self.gym.step_graphics(self.sim)
                    self.gym.draw_viewer(self.viewer, self.sim, False)
                    self.gym.sync_frame_time(self.sim)

                frame += 1

            data_cp = self.storage.cpu()
            data_np = data_cp.numpy()
            data_pd = pd.DataFrame(data_np[1:])
            data_pd.to_csv(
                f"../curriculum_data/{self.task_name}_raw_data/data_{i}.csv",
                index=False,
                header=False,
            )

        print("Done")

        if not self.args.headless:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def split(self):
        # implement something to collect all data_i.csv files into one
        path = f"../curriculum_data/{self.task_name}_raw_data/"
        all_files = glob.glob(os.path.join(path, "*.csv"))

        print(f'Path of all .csv files to be loaded:')
        print('\n'.join(map(str, sorted(all_files, key = lambda x: int(x.split("/")[-1].split("_")[1].split(".")[0])))))

        df = pd.concat(
            (pd.read_csv(f, header=None, index_col=False) for f in all_files)
        )
        print(f"Dataframe shape after loading all .csv files: {df.shape}")

        df = df.to_numpy()

        # remove points that are on the boundary of the workspace
        raw_size = df.shape[0]
        point_tree = spatial.cKDTree(df[:, [0, 1, 2]])
        df_indices = point_tree.query_ball_point([0, 0, 0], self.workspace_size)
        df = pd.DataFrame(df[df_indices])
        print(f"Number of out-of-range point discarded: {raw_size - df.shape[0]}")

        # remove points that are too close to the ground
        raw_size = df.shape[0]
        df = df[df[2] >= 0.2]
        print(f'Number of point too close to the ground discarded: {raw_size - df.shape[0]}')

        df = df.to_numpy()

        # use nearest neighbourhood algorithm to separate curriculums
        point = [np.max(df[:, 0]) / 2, 0, np.max(df[:, 2]) / 2]
        for i, value in enumerate(self.curriculum_sizes):
            # create point tree of the first three columns of the dataframe
            point_tree = spatial.cKDTree(df[:, [0, 1, 2]])
            # obtains indices of the points within the neighborhood
            df_indices = point_tree.query_ball_point(point, value)
            # create a curriculum dataframe that contains only the points inside the specified radius
            curr = pd.DataFrame(df[df_indices])

            print(f"Curriculum: {i} with {curr.shape} points")
            curr.to_csv(
                f"../curriculum_data/{self.task_name}_curriculums/curr_{i}.csv",
                index=False,
                header=False,
            )


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

    gen = GeneratePoses(
        args=args,
        cfg=cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        device_type=args.device,
        device_id=args.device_id,
        headless=args.headless,
    )

    if gen.args.record:
        if not os.path.exists(f"../curriculum_data/{gen.task_name}_raw_data/"):
            os.makedirs(f"../curriculum_data/{gen.task_name}_raw_data/")
        gen.record()
    if gen.args.split:
        if not os.path.exists(f"../curriculum_data/{gen.task_name}_curriculums/"):
            os.makedirs(f"../curriculum_data/{gen.task_name}_curriculums/")
        gen.split()
