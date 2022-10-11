## Plotting

<summary><b>1. Plotting reward functions</b></summary>
<p>

The script `reward_func_plot.py` was created with the intention to visualize how different scalar values affect the reward given to the agent when it comes near the goal or moves close to an obstacle. Run the following command in `python/plotting/` directory:

```sh
python reward_func_plot.py
```

The flag `--reward_type` can be set equal to either `distance` or `collision` to plot the respective functions.

</p>

<summary><b>2. Plotting data from W&B</b></summary>
<p>

The script `wandb_plot.py` was created to evaluate sets of different hyperparameters and ablations. The script is modular such that it does not require any modifications to plot different runs uploaded to W&B. Information for different hyperparameter tuning and ablations experiments is stored in config files in the directory `/wandb_runs/*.yaml`. The information that is required is the run `id`s over a number of seeds (stored in a list for each experiment), secondly, another list containing the `legend` which has information about the parameter that was changed, the legend is supplementary information for each id list.

Run the following command in `python/plotting` directory:

```sh
python wandb_plot.py --config=curr_level --condition='Starting curriculum level'
```

This will pull the data from the wandb API for each of the run ids stored in the `/wandb_runs/curr_level.yaml` file. Given that there were 6 different experiments conducted for this, it will plot six different graphs in the same plot with a solid mean value line and standard deviation with a hue. To customize the plot, a number of options are available:

- `--config`: Name of the config file in `/wandb_runs/`
- `--xaxis`: The data to be plotted on the x-axis:
  - `timestep`
  - `episode`
  - `wall_time`
- `--value`: The data to be plotted on the y-axis:
  - `Train1/mean_reward`
  - `Train1/mean_episode_length`
  - `success_rate`
  - `rolling_avg_success_rate`
  - `Loss/value_function`
  - `Loss/surrogate`
  - `curriculum_level`
  - `distance_error`
  - `orientation_error`
  - `Policy/mean_noise_std`
  - `Train2/mean_episode_length`
  - `Train2/mean_reward`
- `--condition`: The name of the experiment that is being run, e.g. `'Starting curriculum level'`
- `--xlabel`: The label for the x-axis
- `--ylabel`: The label for the y-axis
- `--smooth`: Smooth data by averaging it over a fixed window size, e.g. `10`
- `--est`: Choose how the data is plotted:
  - `mean`
  - `max`
  - `min`

There are other plotting scripts such as `tri_plot.py` and `duo_plot.py` that were created for some specific purpose. They all follow the same implementation as `wandb_plot.py` and use the same command line arguments.

Alternatively, to plot many figures in one go. Use the `plot.sh` script and make any modifications necessary.

</p>

<summary><b>3. Plotting Sim2Real Comparison</b></summary>
<p>

The script `sim2real_comparison.py` was created to compare data collected from the simulation and the real world. The script requires two `.csv` files with joint position data for two trajectories that are to be compared, each joint data should be in a column for itself and with no header or index column. The script looks for these two files (`real_joint_pos.csv` and `sim_joint_pos.csv`) stored in `plotting/csv_data`.

Run the following command in `python/plotting` directory:

```sh
python sim2real_comparison.py
```

</p>

<summary><b>4. Additional dependencies for plotting with LaTeX font</b></summary>
<p>

Depending on the system, it may necessary to install LaTeX to plot the figures with the current `rcParams`. This can be done with the following command:

```sh
sudo apt-get install texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super
pip install latex
```

</p>
