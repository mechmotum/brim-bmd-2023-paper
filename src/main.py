from __future__ import annotations

import json
import os

import cloudpickle as cp
import matplotlib.pyplot as plt
import numpy as np

from container import DataStorage, Metadata, SteerWith, ShoulderJointType
from model import set_bicycle_model, set_simulator
from problem import set_problem, set_constraints, set_initial_guess
from utils import NumpyEncoder, Timer, create_time_lapse, create_animation, create_plots

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SRC_DIR, "data")
OUTPUT_DIR = os.path.join(SRC_DIR, "output")
i = 0
while os.path.exists(os.path.join(OUTPUT_DIR, f"result{i}")):
    i += 1
DEFAULT_RESULT_DIR = os.path.join(OUTPUT_DIR, f"result{i}")

LONGITUDINAL_DISPLACEMENT = 10.0
LATERAL_DISPLACEMENT = 1.0
STRAIGHT_LENGTHS = 2.5
NUM_NODES = 100
DURATION = 2.0
mean_tracking_error = 0.025
estimated_torque = 2.5
control_weight = DURATION * mean_tracking_error ** 2  # Aimed path cost
path_weight = DURATION * estimated_torque ** 2  # Estimated input cost
WEIGHT = path_weight / (control_weight + path_weight)
WEIGHT = 1 - 1E-4  # Overwrite the above to match the paper.

METADATA = Metadata(
    bicycle_only=False,
    model_upper_body=True,
    front_frame_suspension=False,
    shoulder_type=ShoulderJointType.FLEX_ROT,
    steer_with=SteerWith.PEDAL_STEER_TORQUE,
    parameter_data_dir=DATA_DIR,
    bicycle_parametrization="Browser",
    rider_parametrization="Jason",
    duration=DURATION,
    longitudinal_displacement=LONGITUDINAL_DISPLACEMENT,
    lateral_displacement=LATERAL_DISPLACEMENT,
    straight_lengths=STRAIGHT_LENGTHS,
    num_nodes=NUM_NODES,
    weight=WEIGHT,
)

if __name__ == "__main__":
    timer = Timer()
    if not os.path.exists(DEFAULT_RESULT_DIR):
        os.mkdir(DEFAULT_RESULT_DIR)
    with open(os.path.join(DEFAULT_RESULT_DIR, "README.md"), "w") as f:
        f.write(f"# Result {i}\n## Metadata\n{METADATA}\n")
    data = DataStorage(METADATA)
    REUSE_LAST_MODEL = True
    if REUSE_LAST_MODEL and os.path.exists("last_model.pkl"):
        with timer("Reloading last model"):
            with open("last_model.pkl", "rb") as f:
                data = cp.load(f)
    else:
        with timer("Computing the equations of motion"):
            set_bicycle_model(data)
        with timer("Initializing the simulator"):
            set_simulator(data)
        with open("last_model.pkl", "wb") as f:
            cp.dump(data, f)
    with timer("Defining the constraints and objective"):
        set_constraints(data)
    with timer("Making an initial guess"):
        set_initial_guess(data)
    with timer("Initializing the Problem object"):
        set_problem(data)
    data.problem.add_option("output_file",
                            os.path.join(DEFAULT_RESULT_DIR, "ipopt.txt"))
    with timer("Solving the problem"):
        data.solution, info = data.problem.solve(data.initial_guess)
    timer.to_file(os.path.join(DEFAULT_RESULT_DIR, "timings.txt"))
    print("Estimated torque:",
          np.sqrt(data.metadata.interval_value * (data.solution_input ** 2).sum() /
                  data.metadata.duration))
    print("Plotting the results...")
    data.problem.plot_objective_value()
    data.problem.plot_trajectories(data.solution)
    data.problem.plot_constraint_violations(data.solution)
    with open(os.path.join(DEFAULT_RESULT_DIR, "solution_info.txt"), "w",
              encoding="utf-8") as f:
        json.dump(info, f, cls=NumpyEncoder)
    with open(os.path.join(DEFAULT_RESULT_DIR, "data.pkl"), "wb") as f:
        cp.dump(data, f)
    create_plots(data)
    create_time_lapse(data, n_frames=7)
    create_animation(data, os.path.join(DEFAULT_RESULT_DIR, "animation.gif"))

    for i in plt.get_fignums():
        plt.figure(i).savefig(os.path.join(DEFAULT_RESULT_DIR, f"figure{i}.png"))

    plt.show()
