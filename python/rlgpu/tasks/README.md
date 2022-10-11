## Tests

<summary><b>1. Collision</b></summary>
<p>

The collision testing script visualizes the primitive geometric shapes that are used to carry out collision avoidance. The values for the primitives are stored in `python/rlgpu/cfg/*.yaml` files and should be manually measured and specified for the robot that is used. To run the collision test, run the following from the `python/` directory:

```sh
python tasks/collision_test.py --task=Iiwa14 \
                               --num_envs=1
```

</p>

<summary><b>2. Curriculum</b></summary>
<p>

The curriculum testing script visualizes the five or six curriculums spawned by the torus, spherical coordinates, or generated poses methods. The values that determines the size for each curriculum are stored in `python/rlgpu/cfg/*.yaml`. To run the curriculum test, run the following from the `python/` directory:

```sh
python tasks/curriculum_test.py --task=Iiwa14 \
                                --num_envs=16 \
                                --boundary
```

Here the last flag can be either `--boundary` or `--interior` to either visualize random points inside the boundary or the boundary itself.

</p>

<summary><b>3. Distance and orientation </b></summary>
<p>

The distance and orientation testing script visualizes the end-effector location and the goal positions with different orientation. The manipulator moves randomly and the distance and orientation error is calculated and printed.  To run the distance and orientation test, run the following from the `python/` directory:

```sh
python tasks/dist_rot_test.py --task=DoosanH2017 \
                              --num_envs=1 \
                              --random_goal=1
```

Here the `--random_goal` flag is a number between 1-4 that decides the orientation of the goal.

</p>
