from __future__ import annotations

import cloudpickle as cp
import matplotlib.pyplot as plt

from container import DataStorage, Metadata, SteerWith
from model import set_bicycle_model
from problem import set_problem
from utils import create_time_lapse, create_animation, create_plots

data_dir = "data"
mean_tracking_error = 0.1
control_weight = 2 * mean_tracking_error ** 2
path_weight = 2 * 0.5 ** 2
weight = path_weight / (control_weight + path_weight)
METADATA = Metadata(
    front_frame_suspension=False,
    upper_body_bicycle_rider=False,
    steer_with=SteerWith.PEDAL_STEER_TORQUE,
    parameter_data_dir=data_dir,
    bicycle_parametrization="Fisher",
    duration=3.0,
    longitudinal_displacement=15.0,
    lateral_displacement=1.0,
    straight_lengths=2.5,
    num_nodes=100,
    weight=weight,
)
REUSE_SOLUTION_AS_GUESS = False
data = DataStorage(METADATA)
set_bicycle_model(data)
set_problem(data)
if REUSE_SOLUTION_AS_GUESS:
    with open("data.pkl", "rb") as f:
        data_old = cp.load(f)
        get_x = lambda d, xi: d.solution_state[d.solution_state[:].index(xi), :]
        get_r = lambda d, ri: d.solution_input[d.solution_input[:].index(ri), :]
        for i, xi in enumerate(data.x):
            if xi in data_old.x:
                data.initial_guess[i, :] = get_x(data_old, xi)
        for i, ri in enumerate(data.input_vars):
            if ri in data_old.input_vars:
                data.initial_guess[len(data.x) + i, :] = get_r(data_old, ri)
data.solution, info = data.problem.solve(data.initial_guess)
data.problem.plot_objective_value()
data.problem.plot_trajectories(data.solution)
data.problem.plot_constraint_violations(data.solution)
with open("data.pkl", "wb") as f:
    cp.dump(data, f)
create_plots(data)
create_time_lapse(data, n_frames=6)
create_animation(data)

if __name__ == "__main__":
    plt.show()
