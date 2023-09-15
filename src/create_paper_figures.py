import contextlib
import os

import cloudpickle as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scienceplots  # noqa: F401
import sympy as sm
from cycler import cycler
from matplotlib.gridspec import GridSpec

from utils import create_time_lapse, get_solution_statistics

plt.style.use(["science", "no-latex"])
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["xtick.major.pad"] += 3.0
plt.rcParams["xtick.minor.pad"] += 3.0
plt.rcParams["ytick.major.pad"] += 3.0
plt.rcParams["ytick.minor.pad"] += 3.0
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.prop_cycle'] = cycler(
    'color', ['#0C5DA5', '#00B945', '#FF2C00', '#FF9500', '#845B97', '#474747',
              '#9e9e9e', '#008080'])

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
PAGE_WIDTH = 7.275454  # LaTeX \textwidth in inches
markevery = 6
OPTIMIZATION_STYLES = [
    {"color": "C0", "linestyle": "-"},  # "marker": "o", "markevery": (0, markevery)},
    {"color": "C1", "linestyle": "--"},  # "marker": "v", "markevery": (5, markevery)},
    {"color": "C2", "linestyle": ":"},  # "marker": "^", "markevery": (2, markevery)},
    {"color": "C3", "linestyle": "-"},  # "marker": "<", "markevery": (4, markevery)},
    {"color": "C4", "linestyle": "--"},  # "marker": ">", "markevery": (1, markevery)},
    {"color": "C5", "linestyle": ":"},  # "marker": "s", "markevery": (3, markevery)},
]
legend_optimization = (
    [plt.Line2D([0], [0], **OPTIMIZATION_STYLES[i - 1]) for i in range(1, 7)],
    [fr"\#{i}" for i in range(1, 7)],
)


def get_x(data, xi) -> npt.NDArray[np.float64]:
    if isinstance(xi, str):
        x_names = [x.name for x in data.x]
        idx_mapping = [(0, "x"), (1, "y"), (2, "yaw"), (3, "roll"), (6, "steer"),
                       (7, "drive")]
        qs = {name: data.bicycle.q[i] for i, name in idx_mapping}
        us = {name: data.bicycle.u[i] for i, name in idx_mapping}
        if xi in x_names:
            xi = data.x[x_names.index(xi)]
        if xi.startswith("q_") and xi[2:] in qs:
            xi = qs[xi[2:]]
        elif xi.startswith("u_") and xi[2:] in us:
            xi = us[xi[2:]]
        else:
            raise ValueError(f"Unknown state variable: {xi}")
    return data.solution_state[data.x[:].index(xi), :]


def get_r(data, ri) -> npt.NDArray[np.float64]:
    if isinstance(ri, str):
        r_names = [r.name for r in data.r]
        if ri in r_names:
            ri = data.r[r_names.index(ri)]
        else:
            raise ValueError(f"Unknown input variable: {ri}")
    return data.solution_input[data.r[:].index(ri), :]


def savefig(fig: plt.Figure, name: str) -> None:
    fig.savefig(os.path.join(OUTPUT_DIR, name + ".svg"), dpi=300, bbox_inches="tight")


data_lst = []
statistics = {"optimization": list(range(1, 7))}
print("Loading data...")
for i in statistics["optimization"]:
    result_dir = os.path.join(OUTPUT_DIR, f"optimization{i}")
    with open(os.path.join(result_dir, "data.pkl"), "rb") as f:
        data_lst.append(cp.load(f))
    for key, value in get_solution_statistics(result_dir, data_lst[-1]).items():
        if key not in statistics:
            statistics[key] = []
        statistics[key].append(value)
q1_path = np.linspace(0, data_lst[0].metadata.longitudinal_displacement, 100)
q2_path = sm.lambdify(
    (data_lst[0].bicycle.q[0],),
    sm.solve(data_lst[0].target, data_lst[0].bicycle.q[1])[0], cse=True)(q1_path)

with contextlib.suppress(ImportError):
    import pandas as pd

    pd.set_option('display.max_columns', None)
    statistics = pd.DataFrame(data=statistics, index=statistics["optimization"])
print(statistics)

print("Plotting...")
optimization = 1
fig_time_lapse, ax = create_time_lapse(data_lst[optimization - 1], 6)
ax.plot(q1_path, q2_path, np.zeros_like(q1_path), label="Target", color="r")
ax.plot(get_x(data_lst[optimization - 1], "q_x"),
        get_x(data_lst[optimization - 1], "q_y"), np.zeros_like(q1_path),
        label="Trajectory", **OPTIMIZATION_STYLES[optimization - 1])
ax.legend(loc="upper left", bbox_to_anchor=(0.16, 0.77))
savefig(fig_time_lapse, f"time_lapse_{optimization}")

fig = plt.figure(figsize=(10, 5))
gs = GridSpec(2, 2, figure=fig)
axs = [fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]
axs[1].sharex(axs[2])
axs[0].plot(q1_path, q2_path, label="Target", color="C6", linestyle="-.")
for i, data in enumerate(data_lst, 1):
    axs[0].plot(get_x(data, "q_x"), get_x(data, "q_y"), **OPTIMIZATION_STYLES[i - 1])
axs[0].set_xlabel("Longitudinal displacement (m)")
axs[0].set_ylabel("Lateral displacement (m)")
axs[0].legend()
for j, xi_name in enumerate(["steer", "roll"], 1):
    for i, data in enumerate(data_lst, 1):
        axs[j].plot(data.time_array, np.rad2deg(get_x(data, f"q_{xi_name}")),
                    **OPTIMIZATION_STYLES[i - 1])
    axs[j].set_ylabel(f"{xi_name.capitalize()} angle (deg)")
axs[-1].set_xlabel("Time (s)")
fig.legend(*legend_optimization, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.05))
fig.tight_layout()
savefig(fig, "states_all")

fig, axs = plt.subplots(2, 2, figsize=(10, 5), sharex=True)
for i, data in enumerate(data_lst[:-1], 1):
    axs[0, 0].plot(data.time_array, get_r(data, "T_s"),
                   **OPTIMIZATION_STYLES[i - 1])
    axs[1, 0].plot(data.time_array, get_r(data, "T_p"),
                   **OPTIMIZATION_STYLES[i - 1])
axs[1, 0].plot(data_lst[-1].time_array, get_r(data_lst[-1], "T_p"),
               **OPTIMIZATION_STYLES[-1])
axs[0, 1].plot(data_lst[-1].time_array, get_r(data_lst[-1], "T_l"), color="C6",
               label="left")
axs[0, 1].plot(data_lst[-1].time_array, get_r(data_lst[-1], "T_r"), color="C7",
               label="right")
for i, data in enumerate(data_lst, 1):
    tracking_error = np.abs(sm.lambdify((data.x,), data.target)(data.solution_state))
    axs[1, 1].plot(data.time_array, tracking_error * 1E3, **OPTIMIZATION_STYLES[i - 1])
for i in range(2):
    axs[-1, i].set_xlabel("Time (s)")
axs[0, 0].set_ylabel("Steering torque (Nm)")
axs[1, 0].set_ylabel("Pedaling torque (Nm)")
axs[0, 1].set_ylabel("Elbow torque (Nm)")
axs[1, 1].set_ylabel("Tracking error (mm)")
axs[0, 1].legend()
fig.legend(*legend_optimization, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.05))
fig.align_labels()
fig.tight_layout()
savefig(fig, "torques_all")

fig_state, axs = plt.subplots(2, 1, figsize=(5, 3.7), sharex=True)
data = data_lst[optimization - 1]
for xi_name in ("steer", "roll", "yaw"):
    axs[0].plot(data.time_array, np.rad2deg(get_x(data, f"q_{xi_name}")), label=xi_name)
    axs[1].plot(data.time_array, np.rad2deg(get_x(data, f"u_{xi_name}")), label=xi_name)
axs[-1].set_xlabel("Time (s)")
axs[0].set_ylabel("Angle (deg)")
axs[1].set_ylabel("Angular velocity (deg/s)")
axs[0].legend()
fig_state.align_labels()
fig_state.tight_layout()
savefig(fig_state, f"states_{optimization}")

if __name__ == "__main__":
    plt.show()
