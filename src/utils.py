from __future__ import annotations

import argparse
import enum
import json
import os
import re
from copy import copy
from time import perf_counter

import cloudpickle as cp
import matplotlib.pyplot as plt
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from brim.core.base_classes import BrimBase
from brim.utilities.plotting import Plotter
from matplotlib.animation import FuncAnimation
from scipy.interpolate import CubicSpline
from symmeplot.matplotlib import PlotBody, PlotVector

from container import DataStorage


class Timer:
    def __init__(self):
        self.current_description = None
        self.readout = []

    def __call__(self, current_description: str) -> "Timer":
        self.current_description = current_description
        return self

    def __enter__(self):
        print(f"{self.current_description}...")
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start
        self.readout.append((self.current_description, self.time))
        print(f'{self.current_description} ran in {self.time:.3f} seconds')

    def to_file(self, file) -> None:
        with open(file, "w") as f:
            for description, time in self.readout:
                f.write(f"{description}: {time} seconds\n")


class EnumAction(argparse.Action):
    """Argparse action for handling Enums.

    Source: https://stackoverflow.com/a/60750535/20185124
    """

    def __init__(self, **kwargs):
        # Pop off the type value
        enum_type = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum_type is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum_type, enum.Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum
        kwargs.setdefault("choices", tuple(e.name for e in enum_type))

        super(EnumAction, self).__init__(**kwargs)

        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        value = self._enum[values]
        setattr(namespace, self.dest, value)


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types.

    Source: https://stackoverflow.com/a/49677241/20185124
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        try:
            return json.JSONEncoder.default(self, obj)
        except TypeError:
            return str(obj)


def get_all_symbols_from_model(brim_obj: BrimBase) -> set[sm.Symbol]:
    """Get all the symbols from a model."""
    syms = set(brim_obj.symbols.values())
    if hasattr(brim_obj, "submodels"):
        for submodel in brim_obj.submodels:
            syms.update(get_all_symbols_from_model(submodel))
    if hasattr(brim_obj, "connections"):
        for connection in brim_obj.connections:
            syms.update(get_all_symbols_from_model(connection))
    if hasattr(brim_obj, "load_groups"):
        for load_group in brim_obj.load_groups:
            syms.update(get_all_symbols_from_model(load_group))
    return syms


def get_ipopt_statistics(result_dir):
    with open(os.path.join(result_dir, "ipopt.txt"), "r", encoding="utf-8") as f:
        ipopt_output = f.read()
    objective = float(re.search(
        re.compile(r"Objective.*?:\s+(.*?)\s+(.*)"), ipopt_output).group(2))
    nlp_iterations = int(re.search(
        re.compile(r"Number of Iterations\.\.\.\.: (\d+)"), ipopt_output).group(1))
    ipopt_time = float(re.search(
        re.compile(r"Total seconds in IPOPT[ ]+= (\d+\.\d+)"), ipopt_output).group(1))
    ipopt_exit = re.search(re.compile(f"EXIT: (.*)"), ipopt_output).group(1)
    return {
        "Objective": objective,
        "#NLP iterations": nlp_iterations,
        "Time in Ipopt": ipopt_time,
        "Ipopt exit status": ipopt_exit,
    }

def get_solution_statistics(result_dir, data: DataStorage = None):
    if data is None:
        with open(os.path.join(result_dir, "data.pkl"), "rb") as f:
            data = cp.load(f)
    tracking_cost = data.target ** 2
    input_cost = sum(i ** 2 for i in data.input_vars)
    tracking_cost_val = create_objective_function(data, tracking_cost)[0](data.solution)
    input_cost_val = create_objective_function(data, input_cost)[0](data.solution)
    mean_tracking_error = np.sqrt(tracking_cost_val / data.metadata.duration)
    estimated_torque = np.sqrt(input_cost_val / data.metadata.duration)
    ipopt_stats = get_ipopt_statistics(result_dir)
    return {
        "Objective": ipopt_stats["Objective"],
        "Tracking cost": tracking_cost_val,
        "Input cost": input_cost_val,
        "Mean tracking error": mean_tracking_error,
        "Estimated torque": estimated_torque,
        "#NLP iterations": ipopt_stats["#NLP iterations"],
        "Time in Ipopt": ipopt_stats["Time in Ipopt"],
        "Ipopt exit status": ipopt_stats["Ipopt exit status"],
    }


def create_objective_function(data: DataStorage, objective: sm.Expr):
    nx = data.x.shape[0]
    nr = data.input_vars.shape[0]
    N, interval = data.metadata.num_nodes, data.metadata.interval_value
    split_N = nx * N

    objective = me.msubs(objective, data.constants)
    objective_grad = sm.ImmutableMatrix([objective]).jacobian(
        data.x.col_join(data.input_vars))
    objective_grad = [
        np.zeros(N) if grad == 0 else grad for grad in objective_grad
    ]

    eval_objective = sm.lambdify((data.x, data.input_vars), objective, cse=True)
    eval_objective_grad = sm.lambdify((data.x, data.input_vars), objective_grad,
                                      cse=True)

    def obj(free):
        return interval * eval_objective(
            free[:split_N].reshape((nx, N)), free[split_N:].reshape((nr, N))).sum()

    def obj_grad(free):
        return interval * np.hstack(
            eval_objective_grad(free[:split_N].reshape((nx, N)),
                                free[split_N:].reshape((nr, N))))

    return obj, obj_grad


def plot_constraint_violations(self, vector):
    """Improved version of opty's ``Problem.plot_constraint_violations``."""

    con_violations = self.con(vector)
    con_nodes = range(self.collocator.num_states,
                      self.collocator.num_collocation_nodes + 1)
    N = len(con_nodes)
    fig, axes = plt.subplots(2)

    for i, symbol in enumerate(self.collocator.state_symbols):
        axes[0].plot(con_nodes, con_violations[i * N:i * N + N],
                     label=sm.latex(symbol, mode='inline'))
    axes[0].legend()

    axes[0].set_title('Constraint Violations')
    axes[0].set_xlabel('Node Number')

    left = range(len(self.collocator.instance_constraints))
    axes[-1].bar(left, con_violations[-len(self.collocator.instance_constraints):],
                 tick_label=[sm.latex(s, mode='inline')
                             for s in self.collocator.instance_constraints])
    axes[-1].set_ylabel('Instance')
    axes[-1].set_xticklabels(axes[-1].get_xticklabels(), rotation=-10)

    return axes


def create_plots(data: DataStorage) -> tuple[plt.Figure, plt.Axes]:
    t_arr = data.time_array
    x_arr = data.solution_state
    r_arr = data.solution_input
    q1_path = np.linspace(0, data.metadata.longitudinal_displacement, 100)
    q2_path = sm.lambdify(
        (data.bicycle.q[0],), sm.solve(data.target, data.bicycle.q[1])[0], cse=True
    )(q1_path)
    idx_mapping = [(0, "x"), (1, "y"), (2, "yaw"), (3, "roll"), (6, "steer")]
    qs = {name: data.bicycle.q[i] for i, name in idx_mapping}
    us = {name: data.bicycle.u[i] for i, name in idx_mapping}
    get_q = lambda q_name: x_arr[data.system.q[:].index(qs[q_name]), :]  # noqa: E731
    get_u = lambda u_name: x_arr[len(data.system.q) +  # noqa: E731
                                 data.system.u[:].index(us[u_name]), :]
    fig, axs = plt.subplots(2, 2, figsize=(8, 4))
    axs[0, 1].sharex(axs[1, 1])
    axs[0, 0].plot(q1_path, q2_path, label="target")
    axs[0, 0].plot(get_q("x"), get_q("y"), label="trajectory")
    axs[0, 0].set_xlabel("Longitudinal displacement (m)")
    axs[0, 0].set_ylabel("Lateral displacement (m)")
    axs[0, 0].legend()
    # axs[0, 0].axis('equal')
    name_mapping = {
        "T_p": "pedal",
        "T_s": "steer",
        "T_l": "left elbow",
        "T_r": "right elbow",
    }
    for i, ri in enumerate(data.r):
        axs[1, 0].plot(t_arr, r_arr[i, :], label=name_mapping.get(ri.name, ri.name))
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Torque (Nm)")
    axs[1, 0].legend()
    for name in ("yaw", "roll", "steer"):
        axs[0, 1].plot(t_arr, get_q(name), label=name)
    axs[0, 1].set_ylabel("Angle (rad)")
    axs[0, 1].legend()
    for name in ("yaw", "roll", "steer"):
        axs[1, 1].plot(t_arr, get_u(name), label=name)
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Angular velocity (rad/s)")
    fig.align_labels()
    fig.tight_layout()
    return fig, axs


def create_animation(data: DataStorage, output: str
                     ) -> tuple[plt.Figure, plt.Axes, FuncAnimation]:
    x_eval = CubicSpline(data.time_array, data.solution_state.T)
    r_eval = CubicSpline(data.time_array, data.solution_input.T)
    p, p_vals = zip(*data.constants.items())

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 10))
    plotter = Plotter.from_model(data.bicycle_rider, ax=ax)
    plotter.lambdify_system((data.x[:], data.input_vars[:], p))
    plotter.evaluate_system(x_eval(0), r_eval(0), p_vals)
    plotter.plot()
    _plot_ground(data, plotter)
    ax.invert_zaxis()
    ax.invert_yaxis()
    ax.view_init(19, 14)
    ax.set_aspect("equal")
    ax.axis("off")

    fps = 30
    ani = plotter.animate(
        lambda ti: (x_eval(ti), r_eval(ti), p_vals),
        frames=np.arange(0, data.time_array[-1], 1 / fps),
        blit=False)
    return fig, ax, ani


def create_time_lapse(data: DataStorage, n_frames: int = 7
                      ) -> tuple[plt.Figure, plt.Axes]:
    x_eval = CubicSpline(data.time_array, data.solution_state.T)
    r_eval = CubicSpline(data.time_array, data.solution_input.T)
    p, p_vals = zip(*data.constants.items())

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 10))
    plotter = Plotter.from_model(data.bicycle_rider, ax=ax)
    plotter.lambdify_system((data.x[:], data.input_vars[:], p))
    queue = [plotter]
    while queue:
        parent = queue.pop()
        if isinstance(parent, PlotBody):
            parent.plot_frame.visible = False
            parent.plot_masscenter.visible = False
        elif isinstance(parent, PlotVector):
            parent.visible = False
        else:
            queue.extend(parent.children)
    plotter.evaluate_system(x_eval(0), r_eval(0), p_vals)
    for i in range(n_frames):
        for artist in plotter.artists:
            artist.set_alpha(i * 1 / (n_frames + 1) + 1 / n_frames)
            ax.add_artist(copy(artist))
        time = i / (n_frames - 1) * data.time_array[-1]
        plotter.evaluate_system(x_eval(time), r_eval(time), p_vals)
        plotter.update()
    for artist in plotter.artists:
        artist.set_alpha(1)
        ax.add_artist(copy(artist))
    _plot_ground(data, plotter)
    ax.invert_zaxis()
    ax.invert_yaxis()
    ax.view_init(24, 29)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _plot_ground(data: DataStorage, plotter: Plotter):
    q1_arr = data.solution_state[data.system.q[:].index(data.bicycle.q[0]), :]
    q2_arr = data.solution_state[data.system.q[:].index(data.bicycle.q[1]), :]
    p, p_vals = zip(*data.constants.items())

    front_contact_coord = data.bicycle.front_tire.contact_point.pos_from(
        plotter.zero_point).to_matrix(plotter.inertial_frame)[:2]
    eval_fc = sm.lambdify((data.system.q[:] + data.system.u[:], p), front_contact_coord,
                          cse=True)
    fc_arr = np.array(eval_fc(data.solution_state, p_vals))
    x_min = min((fc_arr[0, :].min(), q1_arr.min()))
    x_max = max((fc_arr[0, :].max(), q1_arr.max()))
    y_min = min((fc_arr[1, :].min(), q2_arr.min()))
    y_max = max((fc_arr[1, :].max(), q2_arr.max()))
    X, Y = np.meshgrid(np.arange(x_min - 1, x_max + 1, 0.5),
                       np.arange(y_min - 1, y_max + 1, 0.5))
    plotter.axes.plot_wireframe(X, Y, np.zeros_like(X), color="k", alpha=0.3, rstride=1,
                                cstride=1)
    plotter.axes.set_xlim(X.min(), X.max())
    plotter.axes.set_ylim(Y.min(), Y.max())
