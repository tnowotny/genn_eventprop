import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gs

from itertools import chain
from pandas import read_csv

data = read_csv("spyx_genn.csv", delimiter=",")

data_1024_hidden = data[data["Num hidden"] == 1024]
data_256_hidden = data[data["Num hidden"] == 256]

fig = plt.figure(figsize=(6.3, 3.2))

gsp = gs.GridSpec(1, 6)

line_gs = gs.GridSpecFromSubplotSpec(2, 1, subplot_spec=gsp[0:4])
bar_gs = gs.GridSpecFromSubplotSpec(2, 1, subplot_spec=gsp[4:6])

mem_line_ax = plt.Subplot(fig, line_gs[0])
time_line_ax = plt.Subplot(fig, line_gs[1])
bar_256_ax = plt.Subplot(fig, bar_gs[0])
bar_1024_ax = plt.Subplot(fig, bar_gs[0])


line_fig, line_axes = plt.subplots(1, 2, figsize=(6.3 * 2.0 / 3.0,3.2))

# Plot memory
actor_1024 = line_axes[0].plot(1000.0 / data_1024_hidden["Timestep [ms]"], 
                               data_1024_hidden["Thomas peak GPU [mb]"] - data_1024_hidden["Thomas dataset GPU [mb]"],
                               marker="o")
actor_256 = line_axes[0].plot(1000.0 / data_256_hidden["Timestep [ms]"], 
                              data_256_hidden["Thomas peak GPU [mb]"] - data_256_hidden["Thomas dataset GPU [mb]"], 
                              marker="o")
line_axes[0].plot(1000.0 / data_1024_hidden["Timestep [ms]"], 
                  data_1024_hidden["Spyx platform peak GPU [mb]"] - data_1024_hidden["Spyx dataset GPU [mb]"],
                  marker="o", color=actor_1024[0].get_color(), linestyle="--")
line_axes[0].plot(1000.0 / data_256_hidden["Timestep [ms]"], 
                  data_256_hidden["Spyx platform peak GPU [mb]"] - data_256_hidden["Spyx dataset GPU [mb]"],
                  marker="o", color=actor_256[0].get_color(), linestyle="--")


line_axes[1].plot(1000.0 / data_1024_hidden["Timestep [ms]"], data_1024_hidden["Thomas time [s]"] + data_1024_hidden["Thomas build load time [s]"], marker="o", color=actor_1024[0].get_color())
line_axes[1].plot(1000.0 / data_256_hidden["Timestep [ms]"], data_256_hidden["Thomas time [s]"] + data_256_hidden["Thomas build load time [s]"], marker="o", color=actor_256[0].get_color())
line_axes[1].plot(1000.0 / data_1024_hidden["Timestep [ms]"], data_1024_hidden["Spyx time default [s]"], marker="o", color=actor_1024[0].get_color(), linestyle="--")
line_axes[1].plot(1000.0 / data_256_hidden["Timestep [ms]"], data_256_hidden["Spyx time default [s]"], marker="o", color=actor_256[0].get_color(), linestyle="--")

line_axes[0].set_ylabel("GPU memory [MiB]")
line_axes[1].set_ylabel("Training time [s]")

line_axes[0].set_title("A", loc="left")
line_axes[1].set_title("B", loc="left")

for a in line_axes:
    a.set_xlabel("Num timesteps")
    a.grid(axis="y")
    a.grid(which='minor', alpha=0.3)
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)

line_fig.legend([actor_256[0],  mlines.Line2D([],[], color="black"), actor_1024[0],mlines.Line2D([],[], linestyle="--", color="black")], 
                ["256 hidden neurons", "GeNN", "1024 hidden neurons",  "Spyx"], loc="lower center", ncol=2)
line_fig.tight_layout(pad=0, rect=[0.0, 0.2, 1.0, 1.0])
line_fig.savefig("spyx_genn_benchmark.pdf")




def plot_bars(df, axis, actors=None):
    bar_x = np.arange(0.0, 1.0, 1.0 / len(df))
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
    
    axis.grid(axis="y")
    axis.set_axisbelow(True)
    axis.set_xticks(bar_x)
    axis.set_xticklabels([str(1000 // t) for t in df["Timestep [ms]"]], rotation="vertical")
    axis.set_xlabel(" ")    # Hack so tight-layout leaves matching space
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    
    return actors

bar_fig, bar_axes = plt.subplots(1, 2, sharey=True, figsize=(6.3 / 3.0,3.2))


# Plot groups of bars
actors = plot_bars(data_256_hidden, bar_axes[0])
plot_bars(data_1024_hidden, bar_axes[1], actors)

bar_axes[1].spines['left'].set_visible(False)
bar_axes[1].tick_params("y", left=False)
    
bar_axes[0].set_ylabel("Training time [s]")
bar_axes[0].set_title("256")
bar_axes[1].set_title("1024")
bar_fig.text(0.65, 0.21, "Num timesteps", ha="center")

bar_fig.legend(actors, ["Compile", "Synapse", "Neuron", "Other"],
               loc="lower center", ncol=2, columnspacing=1.5)

bar_fig.tight_layout(pad=0, rect=[0.0, 0.2, 1.0, 1.0])
bar_fig.savefig("genn_benchmark_bars.pdf")

plt.show()
