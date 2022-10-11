import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PATH = "csv_data/"
LABELS = [
    "Real J1",
    "Real J2",
    "Real J3",
    "Real J4",
    "Real J5",
    "Real J6",
    "Real J7",
    "Sim J1",
    "Sim J2",
    "Sim J3",
    "Sim J4",
    "Sim J5",
    "Sim J6",
    "Sim J7",
]
NCOLS = 7

params = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"],
    "pgf.texsystem": "pdflatex",
    "font.size": 10,
    "text.usetex": True,
    "pgf.rcfonts": False,
}

sns.set(style="darkgrid", rc=params)

df_real = pd.read_csv(PATH + "real_joint_pos.csv", header=None, index_col=False)
df_sim = pd.read_csv(PATH + "sim_joint_pos.csv", header=None, index_col=False)

print(df_real.shape)
print(df_sim.shape)

# the real and sim data does not have equal amounts of data due to sampling rate
# so the sim data is divided by 5 to match the real data
df_real_clipped = df_real[0:108]
df_sim_clipped = df_sim.groupby(pd.RangeIndex(len(df_sim)) // 5).first()

print(df_real_clipped.shape)
print(df_sim_clipped.shape)

plt.rcParams.update(params)
fig = plt.figure()
axes = plt.gca()
fig.set_size_inches(w=4.960629, h=3.6)
plt.plot(df_real_clipped, linewidth=0.75)
plt.plot(df_sim_clipped, linewidth=0.75, linestyle="dashed")
plt.legend(
    labels=LABELS,
    handlelength=1.25,
    handleheight=0.0,
    bbox_to_anchor=(0.5, 1.175),
    loc="upper center",
    ncol=NCOLS,
    columnspacing=0.9,
).set_draggable(True)
axes.set_ylim([-np.pi, np.pi])
plt.rcParams.update(params)
plt.xlabel("Timesteps")
plt.rcParams.update(params)
plt.ylabel("Joint position (in radians)")

plt.show()
fig.savefig("sim2real.pdf", format="pdf", bbox_inches="tight")
