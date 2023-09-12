from __future__ import annotations

import json
import os

import cloudpickle as cp
import matplotlib.pyplot as plt
import numpy as np

from .container import DataStorage, Metadata, SteerWith, ShoulderJointType
from .model import set_bicycle_model, set_simulator
from .problem import set_problem, set_constraints, set_initial_guess
from .utils import NumpyEncoder, create_time_lapse, create_animation, create_plots

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data")
output_dir = os.path.join(current_dir, "output")
i = 0
while os.path.exists(os.path.join(output_dir, f"result{i}")):
    i += 1
result_dir = os.path.join(output_dir, f"result{i}")

LONGITUDINAL_DISPLACEMENT = 10.0
LATERAL_DISPLACEMENT = 1.0
STRAIGHT_LENGTHS = 2.5
NUM_NODES = 100
DURATION = 2.0
mean_tracking_error = 0.01
mean_torque = 1
control_weight = DURATION * mean_tracking_error ** 2
path_weight = DURATION * mean_torque ** 2
WEIGHT = path_weight / (control_weight + path_weight)

METADATA = Metadata(
    bicycle_only=False,
    model_upper_body=True,
    front_frame_suspension=False,
    shoulder_type=ShoulderJointType.FLEX_ROT,
    steer_with=SteerWith.HUMAN_TORQUE,
    parameter_data_dir=data_dir,
    bicycle_parametrization="Browser",
    rider_parametrization="Jason",
    duration=DURATION,
    longitudinal_displacement=LONGITUDINAL_DISPLACEMENT,
    lateral_displacement=LATERAL_DISPLACEMENT,
    straight_lengths=STRAIGHT_LENGTHS,
    num_nodes=NUM_NODES,
    weight=WEIGHT,
)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
with open(os.path.join(result_dir, "README.md"), "w") as f:
    f.write(f"# Result {i}\n## Metadata\n{METADATA}\n")
data = DataStorage(METADATA)
REUSE_LAST_MODEL = False
if REUSE_LAST_MODEL and os.path.exists("last_model.pkl"):
    print("Reloading last model...")
    with open("last_model.pkl", "rb") as f:
        data = cp.load(f)
else:
    print("Computing the equations of motion...")
    set_bicycle_model(data)
    print("Initializing the simulator...")
    set_simulator(data)
    with open("last_model.pkl", "wb") as f:
        cp.dump(data, f)
print("Defining the constraints and objective...")
set_constraints(data)
print("Making an initial guess...")
set_initial_guess(data)
print("Initializing the Problem object...")
set_problem(data)
print("Solving the problem...")
data.solution, info = data.problem.solve(data.initial_guess)
print("Mean torque:", np.abs(data.solution_input).mean())
print("Plotting the results...")
data.problem.plot_objective_value()
data.problem.plot_trajectories(data.solution)
data.problem.plot_constraint_violations(data.solution)
with open(os.path.join(result_dir, "solution_info.txt"), "w", encoding="utf-8") as f:
    json.dump(info, f, cls=NumpyEncoder)
with open(os.path.join(result_dir, "data.pkl"), "wb") as f:
    cp.dump(data, f)
create_plots(data)
create_time_lapse(data, n_frames=7)
create_animation(data, os.path.join(result_dir, "animation.gif"))

for i in plt.get_fignums():
    plt.figure(i).savefig(os.path.join(result_dir, f"figure{i}.png"))

if __name__ == "__main__":
    plt.show()
