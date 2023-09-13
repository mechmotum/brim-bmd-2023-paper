import os

import cloudpickle as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scienceplots  # noqa: F401
import sympy as sm

from utils import create_time_lapse

plt.style.use(["science", "no-latex"])
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["xtick.major.pad"] += 2.0
plt.rcParams["xtick.minor.pad"] += 2.0
plt.rcParams["ytick.major.pad"] += 2.0
plt.rcParams["ytick.minor.pad"] += 2.0
plt.rcParams['svg.fonttype'] = 'none'

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
PAGE_WIDTH = 7.275454  # LaTeX \textwidth in inches
markevery = 6
OPTIMIZATION_STYLES = [
    {"color": "C0", "linestyle": "-"},  # "marker": "o", "markevery": (0, markevery)},
    {"color": "C1", "linestyle": "--"},  # "marker": "v", "markevery": (5, markevery)},
    {"color": "C3", "linestyle": ":"},  # "marker": "^", "markevery": (2, markevery)},
    {"color": "C2", "linestyle": "-"},  # "marker": "<", "markevery": (4, markevery)},
    {"color": "C6", "linestyle": "--"},  # "marker": ">", "markevery": (1, markevery)},
    {"color": "C5", "linestyle": ":"},  # "marker": "s", "markevery": (3, markevery)},
]


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
print("Loading data...")
for i in range(1, 7):
    result_dir = os.path.join(OUTPUT_DIR, f"optimization{i}")
    with open(os.path.join(result_dir, "data.pkl"), "rb") as f:
        data_lst.append(cp.load(f))

q1_path = np.linspace(0, data_lst[0].metadata.longitudinal_displacement, 100)
q2_path = sm.lambdify(
    (data_lst[0].bicycle.q[0],),
    sm.solve(data_lst[0].target, data_lst[0].bicycle.q[1])[0], cse=True)(q1_path)

print("Plotting...")
optimization = 1
fig_time_lapse, ax = create_time_lapse(data_lst[optimization - 1], 6)
savefig(fig_time_lapse, f"time_lapse_{optimization}")

fig_trajectory, ax = plt.subplots(1, 1, figsize=(10, 3))
ax.plot(q1_path, q2_path, label="Target", color="C4", linestyle="-.")
for i, data in enumerate(data_lst, 1):
    ax.plot(get_x(data, "q_x"), get_x(data, "q_y"), label=fr"\#{i}",
            **OPTIMIZATION_STYLES[i - 1])
ax.set_xlabel("Longitudinal displacement (m)")
ax.set_ylabel("Lateral displacement (m)")
ax.legend(ncol=2)
# ax.set_aspect("equal")
fig_trajectory.tight_layout()
savefig(fig_trajectory, "trajectory_all")

fig, axs = plt.subplots(2, 2, figsize=(10, 3.5), sharex=True)
for i, data in enumerate(data_lst[:-1], 1):
    axs[0, 1].plot(data.time_array, get_r(data, "steer_torque"),
                   **OPTIMIZATION_STYLES[i - 1])
    axs[1, 1].plot(data.time_array, get_r(data, "pedal_torque"),
                   **OPTIMIZATION_STYLES[i - 1])
axs[0, 1].set_ylabel("Steer torque (Nm)")
axs[1, 1].set_ylabel("Pedal torque (Nm)")
for i, data in enumerate(data_lst, 1):
    for j, xi_name in enumerate(["steer", "roll"]):
        axs[j, 0].plot(data.time_array, get_x(data, f"q_{xi_name}"),
                       **OPTIMIZATION_STYLES[i - 1])
        if i == 1:  # Only done once
            axs[j, 0].set_ylabel(f"{xi_name.capitalize()} angle (rad)")
for i in range(2):
    axs[-1, i].set_xlabel("Time (s)")
fig.legend([plt.Line2D([0], [0], **OPTIMIZATION_STYLES[i - 1]) for i in range(1, 7)],
           [fr"\#{i}" for i in range(1, 7)],
           loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.05))
fig.align_labels()
fig.tight_layout()
savefig(fig, f"angles_torques_all")

fig_state, axs = plt.subplots(2, 1, figsize=(5, 3.5), sharex=True)
data = data_lst[optimization - 1]
for xi_name in ("steer", "roll", "yaw"):
    axs[0].plot(data.time_array, get_x(data, f"q_{xi_name}"), label=xi_name)
    axs[1].plot(data.time_array, get_x(data, f"u_{xi_name}"), label=xi_name)
axs[-1].set_xlabel("Time (s)")
axs[0].set_ylabel("Angle (rad)")
axs[1].set_ylabel("Angular velocity (rad/s)")
axs[0].legend()
fig_state.align_labels()
fig_state.tight_layout()
savefig(fig_state, f"states_{optimization}")

if __name__ == "__main__":
    plt.show()
