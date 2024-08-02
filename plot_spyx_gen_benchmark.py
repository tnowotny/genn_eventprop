import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gs

from itertools import chain
from pandas import read_csv

data = read_csv("spyx_genn.csv", delimiter=",")

data_1024_hidden = data[data["Num hidden"] == 1024]
data_256_hidden = data[data["Num hidden"] == 256]

fig, axes = plt.subplots(1, 3, figsize=(6.3, 3.2))


# Plot memory
actor_1024 = axes[0].plot(1000.0 / data_1024_hidden["Timestep [ms]"], 
                              data_1024_hidden["Thomas peak GPU [mb]"] - data_1024_hidden["Thomas dataset GPU [mb]"],
                              marker="o")
actor_256 = axes[0].plot(1000.0 / data_256_hidden["Timestep [ms]"], 
                             data_256_hidden["Thomas peak GPU [mb]"] - data_256_hidden["Thomas dataset GPU [mb]"], 
                             marker="o")
axes[0].plot(1000.0 / data_1024_hidden["Timestep [ms]"], 
                 data_1024_hidden["Spyx platform peak GPU [mb]"] - data_1024_hidden["Spyx dataset GPU [mb]"],
                 marker="o", color=actor_1024[0].get_color(), linestyle="--")
axes[0].plot(1000.0 / data_256_hidden["Timestep [ms]"], 
                 data_256_hidden["Spyx platform peak GPU [mb]"] - data_256_hidden["Spyx dataset GPU [mb]"],
                 marker="o", color=actor_256[0].get_color(), linestyle="--")


axes[1].plot(1000.0 / data_1024_hidden["Timestep [ms]"], data_1024_hidden["Thomas time [s]"] + data_1024_hidden["Thomas build load time [s]"], marker="o", color=actor_1024[0].get_color())
axes[1].plot(1000.0 / data_256_hidden["Timestep [ms]"], data_256_hidden["Thomas time [s]"] + data_256_hidden["Thomas build load time [s]"], marker="o", color=actor_256[0].get_color())
axes[1].plot(1000.0 / data_1024_hidden["Timestep [ms]"], data_1024_hidden["Spyx time default [s]"], marker="o", color=actor_1024[0].get_color(), linestyle="--")
axes[1].plot(1000.0 / data_256_hidden["Timestep [ms]"], data_256_hidden["Spyx time default [s]"], marker="o", color=actor_256[0].get_color(), linestyle="--")

axes[0].set_ylabel("GPU memory [MiB]")
axes[1].set_ylabel("Training time [s]")
axes[0].set_ylim((0, 3000))
axes[1].set_ylim((0, 4000))

axes[0].set_title("A", loc="left")
axes[1].set_title("B", loc="left")

for a in axes[:2]:
    a.set_xlabel("Num timesteps")
    a.grid(axis="y")
    a.grid(which='minor', alpha=0.3)
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)

line_legend = fig.legend([actor_256[0],  mlines.Line2D([],[], color="black"), actor_1024[0],mlines.Line2D([],[], linestyle="--", color="black")], 
                         ["256 hidden neurons", "GeNN", "1024 hidden neurons",  "Spyx"], 
                         loc="lower center", bbox_to_anchor=(2.0 / 6.0, 0.0), ncol=2, columnspacing=1.0)


def plot_bars(df, bar_x, axis, actors=None):
    column_names = ["Thomas build load time [s]", "Thomas synapse time [s]", "Thomas neuron time [s]"]
    colours = [a[0].get_facecolor() for a in actors] if actors is not None else [None] * (len(column_names) + 1)
    bottom = np.zeros(len(df))
    actors = []
    for n, c in zip(column_names, colours):
        height = df[n]
        actors.append(axis.bar(bar_x, height, width=0.3, bottom=bottom, color=c))
        bottom += df[n]

    overhead = (df["Thomas build load time [s]"] + df["Thomas time [s]"]) - bottom
    actors.append(axis.bar(bar_x, overhead, width=0.3, bottom=bottom, color=colours[-1]))
    
    return actors


# Plot groups of bars
assert len(data_256_hidden) == len(data_1024_hidden)
group_bar_x = np.arange(0.0, 1.0, 1.0 / len(data_256_hidden))
actors = plot_bars(data_256_hidden, group_bar_x, axes[2])
plot_bars(data_1024_hidden, group_bar_x + 1.2, axes[2], actors)
    
axes[2].set_ylabel("Training time [s]")
axes[2].set_title("C", loc="left")
axes[2].grid(axis="y")
axes[2].set_axisbelow(True)
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)

axes[2].set_ylim((0, 4000))
axes[2].set_xticks(np.concatenate((group_bar_x, group_bar_x + 1.2)))
axes[2].set_xticklabels([str(1000 // t) for t in chain(data_256_hidden["Timestep [ms]"], data_1024_hidden["Timestep [ms]"])], 
                        rotation="vertical")
axes[2].set_xlabel("Num timesteps")


axes[2].text(0.33, 4500.0, "256", ha="center", va="top")
axes[2].text(1.53, 4500.0, "1024", ha="center", va="top")

fig.legend(actors, ["Compile", "Synapse", "Neuron", "Other"],
           loc="lower center", bbox_to_anchor=(5.0 / 6.0, 0.0), ncol=2, columnspacing=0.8)
fig.add_artist(line_legend)

fig.tight_layout(pad=0, rect=[0.0, 0.2, 1.0, 1.0])
fig.savefig("spyx_genn_benchmark.pdf")

plt.show()
