# Learning Task-independent Joint Control for Robotic Manipulators with Reinforcement Learning and Curriculum Learning

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Weights & Biases monitoring" height=20>](https://wandb.ai/larsv)

This repository implements a Reinforcement Learning (RL) framework based on the NVIDIA Isaac Gym simulator. The RL framework learns policies to control robots using the actuators of the robot, e.g., joints using position or velocity control. The work aims to learn control strategies that make it possible to generate trajectories automatically for environments that change and, in the future, make it possible to operate in a dynamic environment, e.g., collaboration with humans.

<p align="center" float="middle">
<img width="100.0%" src="/docs/iiwa_franka_ur5_jaco.webp"/>
</p>

This work uses the [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym) simulator which makes it possible to train many robots in parallel, massively decreasing the time required to train successful RL agents.

- [1. Overview](#1-project-details)
- [2. Setup](#2-setup)
  - 2.1. Hardware
  - 2.2. Prerequisites
  - 2.3. Installation
    - 2.3.1. Docker
    - 2.3.2. Building Isaac Gym
- [3. Usage](#3-usage)
  - 3.1. Training a new model
    - 3.1.1. Tensorboard
    - 3.1.2. Wandb
  - 3.2. Tests
  - 3.3. Plotting
- [4. Citing](#4-citing)
- [5. Licensing](#5-licensing)

## 1. Overview

Five different manipulators are available in the simulation:

- KUKA LWR iiwa 14 R820
- Doosan H2017
- Franka Emika Panda
- Universal Robots UR5
- Kinova Robotics JACO (6 DOF)

The real-world testing was conducted with the KUKA LWR iiwa and can be found in the videos uploaded to [YouTube](https://www.youtube.com/playlist?list=PLU-jyLdl836XsZYOkd6cANcgW-erx_RDo). For a complete list of all the reward shaping, hyperparameters, and ablation studies that were carried out, navigate to [<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Weights & Biases monitoring" height=20>](https://wandb.ai/larsv) and scroll down to the **Reports** section.

## 2. Setup

This work was carried out using the **NVIDIA Isaac Gym Preview 2 release** and <u>not the latest</u> Preview 4. Refer to the [NVIDIA forum](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/isaac-gym/322) for updates.

**You must possess a copy of the NVIDIA Isaac Gym Preview 2 source code to be able to run this version of the repository**. The binary files for the Isaac Gym simulator are not included with this repository because of license reasons. If you wish to run the simulator, you must copy the `python/isaacgym` directory from the NVIDIA Isaac Gym Preview 2 release and place it in the empty directory `python/isaacgym` of this repository.

<details>
<summary><b>2.1. Hardware</b></summary>
<p>

All development and experiments in this work were carried out on the following system:

- Intel i9-9820X (20 cores at 3.3GHz)
- Quadro GV100 (32 GB video memory)
- 125 GB DDR4 memory

</p>
</details>

<details>
<summary><b>2.2. Prerequisites</b></summary>
<p>

The work was carried out using the following OS, Python and graphical driver versions:

- Ubunut 18.04
- Python 3.7
- NVIDIA driver version: 440.82

However, Ubuntu 20.04 should also work in theory.

</p>
</details>

<details>
<summary><b>2.3. Installation</b></summary>
<p>

Start by cloning the repository:

```sh
mkdir -p rl_task_independent_joint_control && cd rl_task_independent_joint_control
git clone --recursive -b master git@github.com:LarsVaehrens/rl_task_independent_joint_control.git
```

</p>

<summary><b>2.3.1. Docker</b></summary>
<p>

The preferred method of installation is to use the Docker image in the `/docker` directory which was provided by NVIDIA.

Go to the root of the cloned directory and run:

```sh
bash docker/build.sh
bash docker/run.sh
```

</p>

<summary><b>2.3.2. Build Isaac Gym</b></summary>
<p>

For development purposes, using a [Conda virtual environment](https://www.anaconda.com/products/indiviudal) is preferred. A script has been made available by NVIDIA to easily set up this development environment.

Go to the root of the cloned directory and run:

```sh
./create_conda_env_rlenv.sh
conda activate rlenv
python -m pip show isaacgym
```

</p>
</details>

## 3. Usage

The source files for this project are located in `python/`. For training and tests, the scripts in `python/rlgpu/tasks/` can be used for different purposes as described below, and each python script is made modular such that every robot can be used in the simulator with a single script. This is made possible by storing all robot and task-specific information in `.yaml` configuration files in the `python/rlgpu/cfg/` directory.

<details>
<summary><b>3.1. Training a new model</b></summary>
<p>

Additional configuration files are also used for training new models. The hyperparameters used for training are stored in the `python/rlgpu/cfg/train/` directory.

Before a new model can be trained, a decision on how the training should be carried out must be considered. All training is carried out using curriculum learning, and three methods are implemented for this purpose. However, as we have shown in our ablation studies, the third curriculum method provides better training performance than the other. Selection of the curriculum method can be found in the `python/rlgpu/cfg/*.yaml` along with the control mode, e.g., position or velocity control.

If the third method is selected for training, a dataset of poses must first be generated using the `python/rlgpu/tasks/generate_poses.py` script. A sample of how to use this script can be found below.

```sh
cd /RL_joint_generation/python/rlgpu/
python tasks/generate_poses.py --task=Iiwa14 \
                               --num_envs=128 \
                               --record=True \
                               --split=True \
                               --num_samples=1 \
                               --headless
```

Here the `--task` flag is the namespace of the robot that is to be used in the simulation, `--num_envs` is the number of parallel robots that are spawned in the simulation, where 128 is a good number for generating poses, `--record` and `--split` flags determine what operation to carry out, e.g., generating poses with the manipulator stored in `python/rlgpu/curriculum_data/*_raw_data/` and split the generated dataset into appropriate curriculums in `python/rlgpu/curriculum_data/*_curriculums/` that are to be used for training. `--num_samples` determine how many runs are made, where each run generates 1 million samples. The `--headless` flag will reduce the computational load but is also necessary when using the `--split` flag; otherwise, the simulation will crash.

After generating poses and selecting the right configurations, training a new model can be done with the following command:

```sh
python train.py --task=Iiwa14  \
                --num_envs=2048  \
                --headless
```

Here 2048 parallel environments is preferable for fast training. As previously mentioned it is necessary to use the `--headless` flag to disable the graphical user interface for better performance.

</p>

<details>
<summary><b>3.1.1. Tensorboard</b></summary>
<p>

To view the training progress in TensorBoard, run the following command in the root of the repository:

```sh
tensorboard --logdir python/rlgpu/logs/
```

</p>
</details>

<details>
<summary><b>3.1.2. Weights and Biases</b></summary>
<p>

W&B was used for logging data from the training process as it provides a better interface for keeping track of the hundreds of experiments that were carried out in the parameter search and ablation studies.

To use W&B, [create an account](https://wandb.ai/login?signup=true) and generate an API token to start logging data when using the `python/rlgpu/train.py` script. See their [quickstart guide](https://docs.wandb.ai/quickstart).

</p>
</details>
</details>

<details>
<summary><b>3.2. Tests</b></summary>
<p>

A number of test scripts were created in order to develop features for the training script. These scripts include tests of joint and cartesian control, end-effector sensing, curriculum method 1 and 2 visualizations, collision testing, and distance-to-goal measurements. See the [README](python/rlgpu/tasks/README.md) file in `python/rlgpu/tasks/` directory for a full explanation of how to use these scripts.

</p>
</details>

<details>
<summary><b>3.3. Plotting</b></summary>
<p>

The `python/plotting` directory contains a number of scripts that were used to plot data from training, experiments, or reward functions. Please find the [README](python/plotting/README.md) file in the `python/plotting` directory for a full explanation of how to use these scripts.

</p>
</details>

## 4. Citing

<p>

If you use this software, please cite it as below:

```tex
@inproceedings{vaehrens2022learning,
  author    = {Lars V{\ae}hrens and  Daniel D{\'i}ez {\'A}lvarez and Ulrich Berger and Simon B{\o}gh},
  title     = {{Learning} {Task}-independent {Joint} {Control} for {Robotic} {Manipulators} with {Reinforcement} {Learning} and {Curriculum} {Learning}},
  year      = {2022},
  booktitle = {2022 IEEE International Conference on Machine Learning and Applications (ICMLA)},
  month     = dec
}
```

</p>

## 5. Licensing

<p>

This repository hosts code that is part of the Learning Task-independent Joint Control for Robotic Manipulators with Reinforcement Learning and Curriculum Learning paper accepted at ICMLA2022. Copyright © 2022, project contributors: Lars Væhrens. Licensed under the BSD 3-Clause License, which can be found in [`LICENSE`](/LICENSE) file.

This repository is based off code written by NVIDIA and Meta Platforms, Inc. all files that are originally written by NVIDIA or Meta Platforms, Inc. maintain a copyright header. All files that do not contain such a header are original to the aforementioned authors of this repository. All derivative works of this repository must preserve these headers under the terms of the license.

</p>
