from __future__ import annotations

import numpy as np
import numpy.typing as npt
import sympy.physics.mechanics as me
from opty.direct_collocation import Problem

from container import DataStorage, SteerWith
from simulator import Simulator
from utils import create_objective_function, plot_constraint_violations

# Corrected version from opty's ``Problem.plot_constraint_violations``.
plot_constraint_violations.__doc__ = Problem.plot_constraint_violations.__doc__
Problem.plot_constraint_violations = plot_constraint_violations


def set_problem(data: DataStorage) -> None:
    t = me.dynamicsymbols._t  # Time symbol.
    t0, tf = 0.0, data.metadata.duration  # Initial and final time.
    bicycle = data.bicycle

    initial_state_constraints = {
        bicycle.q[0]: 0.0,
        bicycle.q[1]: 0.0,
        bicycle.q[2]: 0.0,
        bicycle.q[3]: 0.0,
        bicycle.q[5]: 0.0,
        bicycle.q[6]: 0.0,
        bicycle.q[7]: 0.0,
    }
    final_state_constraints = {
        bicycle.q[0]: data.metadata.longitudinal_displacement,
        bicycle.q[1]: data.metadata.lateral_displacement,
        bicycle.q[2]: 0.0,
        bicycle.q[3]: 0.0,
        bicycle.q[6]: 0.0,
    }
    if data.metadata.front_frame_suspension:
        initial_state_constraints[bicycle.front_frame.q[0]] = 0.0
        initial_state_constraints[bicycle.front_frame.u[0]] = 0.0

    instance_constraints = (
        bicycle.q[4].replace(t, t0) - bicycle.q[4].replace(t, tf),  # Periodic pitch.
        # Periodic velocities.
        bicycle.u[0].replace(t, t0) - bicycle.u[0].replace(t, tf),
        bicycle.u[1].replace(t, t0) + bicycle.u[1].replace(t, tf),
        bicycle.u[2].replace(t, t0) + bicycle.u[2].replace(t, tf),
        bicycle.u[3].replace(t, t0) + bicycle.u[3].replace(t, tf),
        bicycle.u[4].replace(t, t0) - bicycle.u[4].replace(t, tf),
        bicycle.u[5].replace(t, t0) - bicycle.u[5].replace(t, tf),
        bicycle.u[6].replace(t, t0) + bicycle.u[6].replace(t, tf),
        bicycle.u[7].replace(t, t0) - bicycle.u[7].replace(t, tf),
    )
    instance_constraints += tuple(
        xi.replace(t, t0) - xi_val for xi, xi_val in initial_state_constraints.items()
    ) + tuple(
        xi.replace(t, tf) - xi_val for xi, xi_val in final_state_constraints.items()
    )

    bounds = {
        bicycle.q[0]: (-0.1, data.metadata.longitudinal_displacement + 0.1),
        bicycle.q[1]: (-0.1, data.metadata.lateral_displacement + 0.1),
        bicycle.q[2]: (-1.0, 1.0),
        bicycle.q[3]: (-1.0, 1.0),
        bicycle.q[4]: (-1.0, 1.0),
        bicycle.q[5]: (-100.0, 100.0),
        bicycle.q[6]: (-1.0, 1.0),
        bicycle.q[7]: (-100.0, 100.0),
        bicycle.u[0]: (0.0, 10.0),
        bicycle.u[1]: (-5.0, 5.0),
        bicycle.u[2]: (-2.0, 2.0),
        bicycle.u[3]: (-2.0, 2.0),
        bicycle.u[4]: (-2.0, 2.0),
        bicycle.u[5]: (-20.0, 0.0),
        bicycle.u[6]: (-2.0, 2.0),
        bicycle.u[7]: (-20.0, 0.0),
    }

    if data.metadata.front_frame_suspension:
        bounds.update({
            bicycle.front_frame.q[0]: (-0.1, 0.1),
            bicycle.front_frame.u[0]: (-5.0, 5.0),
        })
    if data.metadata.steer_with is SteerWith.PEDAL_STEER_TORQUE:
        bounds.update({
            data.input_vars[0]: (-10.0, 10.0),
            data.input_vars[1]: (-1.0, 1.0),  # Limit pedal torque for persistence.
        })

    objective_expr = (data.metadata.weight * (data.target) ** 2 +
                      (1 - data.metadata.weight) * sum(i ** 2 for i in data.input_vars))
    print("Objective function:", objective_expr)

    data.initial_guess = generate_initial_guess(data, initial_state_constraints,
                                                final_state_constraints)

    obj, obj_grad = create_objective_function(data, objective_expr)

    problem = Problem(
        obj,
        obj_grad,
        data.eoms,
        data.x,
        data.metadata.num_nodes,
        data.metadata.interval_value,
        known_parameter_map=data.constants,
        instance_constraints=instance_constraints,
        bounds=bounds,
        integration_method='midpoint',
    )

    problem.add_option('nlp_scaling_method', 'gradient-based')

    data.problem = problem


def generate_initial_guess(data: DataStorage, initial_state_constraints,
                           final_state_constraints) -> npt.NDArray[np.float64]:
    d_long = data.metadata.longitudinal_displacement
    d_lat = data.metadata.lateral_displacement
    d_tot = np.sqrt(d_long ** 2 + d_lat ** 2)
    diagonal = False
    if diagonal:
        vel_mean = d_tot / data.metadata.duration
        angle = np.arctan2(d_lat, d_long)
        q20 = 0.0
    else:
        vel_mean = d_long / data.metadata.duration
        angle = 0.0
        q20 = d_lat / 2
    rr = data.constants[data.bicycle.rear_wheel.radius]

    simulator = Simulator(data.system)
    simulator.initial_conditions = {
        **{xi: initial_state_constraints.get(xi, 0.0)
           for xi in data.x},
        data.bicycle.u[0]: vel_mean * np.cos(angle),
        data.bicycle.u[1]: vel_mean * np.sin(angle),
        data.bicycle.q[1]: q20,
        data.bicycle.q[2]: angle,
        data.bicycle.u[5]: -vel_mean / rr,
        data.bicycle.u[7]: -vel_mean / rr,
    }
    simulator.initial_conditions.update(
        {xi: np.random.rand() * 1E-8
         for xi, x_val in simulator.initial_conditions.items() if x_val == 0.0})
    simulator.constants = data.constants
    simulator.inputs = {ri: lambda t, x: 0.0 for ri in data.input_vars}
    simulator.initialize(False)
    t_arr, x_arr = simulator.solve(
        np.linspace(0, data.metadata.duration, data.metadata.num_nodes), "dae",
        rtol=1e-3, atol=1e-6)

    return np.concatenate(
        (x_arr.ravel(), np.zeros(len(data.input_vars) * data.metadata.num_nodes)))
