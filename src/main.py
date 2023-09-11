from __future__ import annotations

import os

import cloudpickle as cp
import matplotlib.pyplot as plt

from container import DataStorage, Metadata, SteerWith
from model import set_bicycle_model
from problem import set_problem
from utils import create_time_lapse, create_animation, create_plots

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data")
output_dir = os.path.join(current_dir, "output")
i = 0
while os.path.exists(os.path.join(output_dir, f"result{i}")):
    i += 1
result_dir = os.path.join(output_dir, f"result{i}")
os.mkdir(result_dir)

mean_tracking_error = 0.01
control_weight = 3 * mean_tracking_error ** 2
path_weight = 3 * 0.5 ** 2
weight = path_weight / (control_weight + path_weight)
METADATA = Metadata(
    front_frame_suspension=False,
    upper_body_bicycle_rider=False,
    steer_with=SteerWith.PEDAL_STEER_TORQUE,
    parameter_data_dir=data_dir,
    bicycle_parametrization="Fisher",
    duration=2.0,
    longitudinal_displacement=10.0,
    lateral_displacement=1.0,
    straight_lengths=2.5,
    num_nodes=100,
    weight=weight,
)
with open(os.path.join(result_dir, "README.md"), "w") as f:
    f.write(f"""\
    # Result {i}
    ## Metadata
    {METADATA}
    """)
REUSE_SOLUTION_AS_GUESS = False
data = DataStorage(METADATA)
set_bicycle_model(data)
set_problem(data)
if REUSE_SOLUTION_AS_GUESS:
    with open("data.pkl", "rb") as f:
        data_old = cp.load(f)
        N = METADATA.num_nodes
        get_x = lambda d, xi: d.solution_state[d.x[:].index(xi), :]
        get_r = lambda d, ri: d.solution_input[d.r[:].index(ri), :]
        for i, xi in enumerate(data.x):
            if xi in data_old.x:
                data.initial_guess[i * N:(i + 1) * N] = get_x(data_old, xi)
        for i, ri in enumerate(data.input_vars):
            if ri in data_old.input_vars:
                data.initial_guess[(len(data.x) + i) * N:(len(data.x) + i + 1) * N] = \
                    get_r(data_old, ri)
data.solution, info = data.problem.solve(data.initial_guess)
data.problem.plot_objective_value()
data.problem.plot_trajectories(data.initial_guess)
data.problem.plot_constraint_violations(data.solution)
with open(os.path.join(result_dir, "data.pkl"), "wb") as f:
    cp.dump(data, f)
create_plots(data)
create_time_lapse(data, n_frames=7)
create_animation(data, os.path.join(result_dir, "animation.gif"))

for i in plt.get_fignums():
    plt.figure(i).savefig(os.path.join(result_dir, f"figure{i}.png"))

if __name__ == "__main__":
    plt.show()