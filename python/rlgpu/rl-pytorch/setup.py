# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from setuptools import setup, find_packages

print(find_packages())

setup(name='rl_pytorch',
      packages=[package for package in find_packages()
                if package.startswith('rl_pytorch')],
      version='1.0.1',
      description='Simple PPO implementation for pytorch',
      author='NVIDIA CORPORATION',
      author_email='',
      url='',
      license='Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.',
      python_requires='>=3.6',
      install_requires=[
          "torch>=1.4.0",
          "torchvision>=0.5.0",
          "numpy>=1.16.4",
          "gym>=0.17.1",
          "wandb>=0.10.3",
          "matplotlib>=3.3.4",
          "seaborn>=0.8.1",
      ],
      )
