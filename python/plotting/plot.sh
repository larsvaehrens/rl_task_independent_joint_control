#!/bin/sh
clear

SMOOTH=10
XAXIS=timestep
XLABEL=Timesteps

python python/plotting/tri_plot.py --config=curr_level --xaxis=$XAXIS -s=$SMOOTH --condition='Curriculum Level' --xlabel=$XLABEL
python python/plotting/tri_plot.py --config=curr_modus --xaxis=$XAXIS -s=$SMOOTH --condition='Curriculum Modus' --xlabel=$XLABEL
python python/plotting/tri_plot.py --config=pos_vel --xaxis=$XAXIS -s=$SMOOTH --condition='Different Control' --xlabel=$XLABEL
python python/plotting/tri_plot.py --config=manip_reset --xaxis=$XAXIS -s=$SMOOTH --condition='Manipulator Reset' --xlabel=$XLABEL
python python/plotting/tri_plot.py --config=start_dist_orien --xaxis=$XAXIS -s=$SMOOTH --condition='Distance Orientation' --xlabel=$XLABEL

python python/plotting/tri_plot.py --config=ent_coef --xaxis=$XAXIS -s=$SMOOTH --condition='Entropy Coefficient' --xlabel=$XLABEL
python python/plotting/tri_plot.py --config=gamma --xaxis=$XAXIS -s=$SMOOTH --condition='Discount Factor' --xlabel=$XLABEL
python python/plotting/tri_plot.py --config=lam --xaxis=$XAXIS -s=$SMOOTH --condition='Lambda GAE' --xlabel=$XLABEL
python python/plotting/tri_plot.py --config=eps_len --xaxis=$XAXIS -s=$SMOOTH --condition='Episode Length' --xlabel=$XLABEL
python python/plotting/tri_plot.py --config=num_envs --xaxis=episode -s=$SMOOTH --condition='Number of Environments' --xlabel=Episodes

python python/plotting/tri_plot.py --config=dist_rew --xaxis=$XAXIS -s=$SMOOTH --condition='Distance Reward' --xlabel=$XLABEL
python python/plotting/tri_plot.py --config=obst_rew --xaxis=$XAXIS -s=$SMOOTH --condition='Collision Reward' --xlabel=$XLABEL
python python/plotting/tri_plot.py --config=bonus_rew --xaxis=$XAXIS -s=$SMOOTH --condition='Bonus Reward' --xlabel=$XLABEL

python python/plotting/tri_plot.py --config=robot_test --xaxis=$XAXIS -s=$SMOOTH --condition='Different Robots' --xlabel=$XLABEL
