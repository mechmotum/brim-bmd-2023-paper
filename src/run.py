from __future__ import annotations

import argparse
import json
import os

import cloudpickle as cp
import matplotlib.pyplot as plt
import numpy as np

from container import DataStorage, Metadata, ShoulderJointType, SteerWith
from main import (
    LONGITUDINAL_DISPLACEMENT, LATERAL_DISPLACEMENT, STRAIGHT_LENGTHS,
    NUM_NODES, DURATION, WEIGHT, DATA_DIR, DEFAULT_RESULT_DIR
)
from model import set_bicycle_model, set_simulator
from problem import set_constraints, set_initial_guess, set_problem
from utils import (
    EnumAction, NumpyEncoder, create_animation, create_plots, create_time_lapse)

parser = argparse.ArgumentParser(description="Run an trajectory tracking problem.")
parser.add_argument("--bicycle-only", action="store_true",
                    help="Use a bicycle-only model.")
parser.add_argument("--model-upper-body", action="store_true",
                    help="Use a model with an upper body.")
parser.add_argument("--front-frame-suspension", action="store_true",
                    help="Use a model with a front frame suspension.")
parser.add_argument("--shoulder-type", type=ShoulderJointType,
                    default=ShoulderJointType.NONE, action=EnumAction)
parser.add_argument("--steer-with", type=SteerWith,
                    default=SteerWith.PEDAL_STEER_TORQUE, action=EnumAction)
parser.add_argument("--bicycle-parametrization", type=str, default="Browser",
                    help="The parametrization of the bicycle model.")
parser.add_argument("--rider-parametrization", type=str, default="Jason",
                    help="The parametrization of the rider model.")
parser.add_argument("--duration", type=float, default=DURATION,
                    help="The time duration of the trajectory.")
parser.add_argument("--longitudinal-displacement", type=float,
                    default=LONGITUDINAL_DISPLACEMENT,
                    help="The longitudinal displacement of the trajectory.")
parser.add_argument("--lateral-displacement", type=float,
                    default=LATERAL_DISPLACEMENT,
                    help="The lateral displacement of the trajectory.")
parser.add_argument("--straight-lengths", type=float, default=STRAIGHT_LENGTHS,
                    help="The length of the straight sections at the start and end "
                         "of the trajectory.")
parser.add_argument("--num-nodes", type=int, default=NUM_NODES,
                    help="The number of nodes in the optimization problem.")
parser.add_argument("--weight", type=float, default=WEIGHT,
                    help="The weight of the path objective [0, 1].")
parser.add_argument("--parameter-data-dir", type=str, default=DATA_DIR,
                    help="The directory containing the parameter data.")
parser.add_argument('-o', '--output', type=str, default=DEFAULT_RESULT_DIR,
                    help="The directory to save the results in.")

result_dir = parser.parse_args().output
METADATA = Metadata(**{
    k: v for k, v in vars(parser.parse_args()).items()
    if k in Metadata.__dataclass_fields__})
print("Running optimization with the following metadata:", METADATA, sep="\n")
print("Saving results to:", result_dir)

if not os.path.exists(result_dir):
    os.mkdir(result_dir)
with open(os.path.join(result_dir, "README.md"), "w") as f:
    f.write(f"# Result\n## Metadata\n{METADATA}\n")
data = DataStorage(METADATA)

print("Computing the equations of motion...")
set_bicycle_model(data)
print("Initializing the simulator...")
set_simulator(data)
print("Defining the constraints and objective...")
set_constraints(data)
print("Making an initial guess...")
set_initial_guess(data)
print("Initializing the Problem object...")
set_problem(data)
data.problem.add_option("output_file", os.path.join(result_dir, "output.txt"))
print("Solving the problem...")
data.solution, info = data.problem.solve(data.initial_guess)
print("Mean torque:",
      np.sqrt(data.metadata.interval_value * (data.solution_input ** 2).sum() /
              data.metadata.duration))

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
