import os

import cloudpickle as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scienceplots  # noqa: F401
import sympy as sm

from utils import create_time_lapse

plt.style.use(["science", "no-latex"])
plt.rcParams['svg.fonttype'] = 'none'

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def get_x(data, xi) -> npt.NDArray[np.float64]:
    if isinstance(xi, str):
        x_names = [x.name for x in data.x]
        idx_mapping = [(0, "x"), (1, "y"), (2, "yaw"), (3, "roll"), (6, "steer")]
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
fig_time_lapse, ax = create_time_lapse(data_lst[0], 6)
fig_time_lapse.savefig(os.path.join(OUTPUT_DIR, "time_lapse.svg"))

fig_trajectory, ax = plt.subplots(1, 1, figsize=(10, 3))
ax.plot(q1_path, q2_path, label="Target")
for i, data in enumerate(data_lst, 1):
    ax.plot(get_x(data, "q_x"), get_x(data, "q_y"), label=fr"#{i}")
ax.set_xlabel("Longitudinal displacement (m)")
ax.set_ylabel("Lateral displacement (m)")
ax.legend()
fig_trajectory.tight_layout()
fig_trajectory.savefig(os.path.join(OUTPUT_DIR, "trajectory.svg"))

fig_torques, axs = plt.subplots(2, 1, figsize=(5, 3.5), sharex=True)
for i, data in enumerate(data_lst[:-1], 1):
    axs[0].plot(data.time_array, get_r(data, "steer_torque"), label=fr"#{i}",
                color=f"C{i}")
    axs[1].plot(data.time_array, get_r(data, "pedal_torque"), label=fr"#{i}",
                color=f"C{i}")
axs[-1].set_xlabel("Time (s)")
axs[0].set_ylabel("Steer torque (Nm)")
axs[1].set_ylabel("Pedal torque (Nm)")
axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig_torques.align_labels()
fig_torques.tight_layout()
fig_torques.savefig(os.path.join(OUTPUT_DIR, "torques.svg"))

fig_state, axs = plt.subplots(2, 1, figsize=(5, 3.5), sharex=True)
data = data_lst[0]
for xi_name in ("yaw", "roll", "steer"):
    axs[0].plot(data.time_array, get_x(data, f"q_{xi_name}"), label=xi_name)
    axs[1].plot(data.time_array, get_x(data, f"u_{xi_name}"), label=xi_name)
axs[-1].set_xlabel("Time (s)")
axs[0].set_ylabel("Angle (rad)")
axs[1].set_ylabel("Angular velocity (rad/s)")
axs[0].legend()
fig_state.align_labels()
fig_state.tight_layout()
fig_state.savefig(os.path.join(OUTPUT_DIR, "states.svg"))

if __name__ == "__main__":
    plt.show()
