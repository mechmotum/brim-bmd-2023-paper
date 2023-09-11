from __future__ import annotations

import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from opty.direct_collocation import Problem
from scipy.optimize import fsolve

from container import DataStorage, SteerWith, ConstraintStorage, ShoulderJointType
from utils import create_objective_function, plot_constraint_violations

# Corrected version from opty's ``Problem.plot_constraint_violations``.
plot_constraint_violations.__doc__ = Problem.plot_constraint_violations.__doc__
Problem.plot_constraint_violations = plot_constraint_violations


def set_constraints(data: DataStorage) -> None:
    t = me.dynamicsymbols._t  # Time symbol.
    t0, tf = 0.0, data.metadata.duration  # Initial and final time.
    bicycle, rider = data.bicycle, data.rider
    spherical_shoulders = (data.metadata.upper_body_bicycle_rider and
                           data.metadata.shoulder_type is ShoulderJointType.SPHERICAL)

    initial_state_constraints = {
        bicycle.q[0]: 0.0,
        bicycle.q[1]: 0.0,
        bicycle.q[2]: 0.0,
        bicycle.q[3]: 0.0,
        bicycle.q[5]: 0.0,
        bicycle.q[6]: 0.0,
        bicycle.q[7]: 0.0,
    }
    if data.metadata.front_frame_suspension:
        initial_state_constraints[bicycle.front_frame.q[0]] = 0.01  # Small compression.
        initial_state_constraints[bicycle.front_frame.u[0]] = 0.0

    final_state_constraints = {
        bicycle.q[0]: data.metadata.longitudinal_displacement,
        bicycle.q[1]: data.metadata.lateral_displacement,
        bicycle.q[2]: 0.0,
        bicycle.q[3]: 0.0,
        bicycle.q[6]: 0.0,
    }

    instance_constraints = (
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
        bicycle.q[2]: (-1.5, 1.5),
        bicycle.q[3]: (-1.5, 1.5),
        bicycle.q[4]: (-1.5, 1.5),
        bicycle.q[5]: (-100.0, 100.0),
        bicycle.q[6]: (-1.5, 1.5),
        bicycle.q[7]: (-100.0, 100.0),
        bicycle.u[0]: (0.0, 10.0),
        bicycle.u[1]: (-10.0, 10.0),
        bicycle.u[2]: (-10.0, 10.0),
        bicycle.u[3]: (-10.0, 10.0),
        bicycle.u[4]: (-10.0, 10.0),
        bicycle.u[5]: (-20.0, 0.0),
        bicycle.u[6]: (-10.0, 10.0),
        bicycle.u[7]: (-20.0, 0.0),
    }

    if data.metadata.front_frame_suspension:
        bounds.update({
            bicycle.front_frame.q[0]: (-0.2, 0.2),
            bicycle.front_frame.u[0]: (-10.0, 10.0),
        })
    if data.metadata.upper_body_bicycle_rider:
        bounds.update({
            rider.right_shoulder.q[0]: (-1.5, 1.5),
            rider.right_shoulder.q[1]: (-1.5, 1.5),
            rider.right_arm.q[0]: (0.0, 3.0),
            rider.left_shoulder.q[0]: (-1.5, 1.5),
            rider.left_shoulder.q[1]: (-1.5, 1.5),
            rider.left_arm.q[0]: (0.0, 3.0),
            rider.right_shoulder.u[0]: (-10.0, 10.0),
            rider.right_shoulder.u[1]: (-10.0, 10.0),
            rider.right_arm.u[0]: (-10.0, 10.0),
            rider.left_shoulder.u[0]: (-10.0, 10.0),
            rider.left_shoulder.u[1]: (-10.0, 10.0),
            rider.left_arm.u[0]: (-10.0, 10.0),
        })
        if spherical_shoulders:
            bounds.update({
                rider.right_shoulder.q[2]: (-1.5, 1.5),
                rider.left_shoulder.q[2]: (-1.5, 1.5),
                rider.right_shoulder.u[2]: (-10.0, 10.0),
                rider.left_shoulder.u[2]: (-10.0, 10.0),
            })
    if data.metadata.steer_with is SteerWith.PEDAL_STEER_TORQUE:
        bounds.update({
            data.input_vars[0]: (-10.0, 10.0),
            data.input_vars[1]: (-10.0, 10.0),
        })
    elif data.metadata.steer_with is SteerWith.HUMAN_TORQUE:
        bounds.update({
            data.input_vars[0]: (-10.0, 10.0),
            data.input_vars[1]: (-10.0, 10.0),
            data.input_vars[2]: (-10.0, 10.0),
        })

    data.objective_expr = (
            data.metadata.weight * (data.target) ** 2 +
            (1 - data.metadata.weight) * sum(i ** 2 for i in data.input_vars))
    data.constraints = ConstraintStorage(
        initial_state_constraints, final_state_constraints, instance_constraints, bounds
    )


def set_problem(data: DataStorage) -> None:
    obj, obj_grad = create_objective_function(data, data.objective_expr)

    problem = Problem(
        obj,
        obj_grad,
        data.eoms,
        data.x,
        data.metadata.num_nodes,
        data.metadata.interval_value,
        known_parameter_map=data.constants,
        instance_constraints=data.constraints.instance_constraints,
        bounds=data.constraints.bounds,
        integration_method='backward euler',
    )

    problem.add_option('nlp_scaling_method', 'gradient-based')

    data.problem = problem


def set_initial_guess(data: DataStorage) -> None:
    d_long = data.metadata.longitudinal_displacement
    d_lat = data.metadata.lateral_displacement
    d_tot = np.sqrt(d_long ** 2 + d_lat ** 2)
    diagonal = False
    if diagonal:
        vel_mean = d_tot / data.metadata.duration
        angle = np.arctan2(d_lat, d_long)
        q2_0 = 0.0
    else:
        vel_mean = d_long / data.metadata.duration
        angle = 0.0
        q2_0 = d_lat / 2
    rr = data.constants[data.bicycle.rear_wheel.radius]

    data.simulator.initial_conditions = {
        **{xi: data.constraints.initial_state_constraints.get(xi, 0.0)
           for xi in data.x},
        data.bicycle.u[0]: vel_mean * np.cos(angle),
        data.bicycle.u[1]: vel_mean * np.sin(angle),
        data.bicycle.q[1]: q2_0,
        data.bicycle.q[2]: angle,
        data.bicycle.u[5]: -vel_mean / rr,
        data.bicycle.u[7]: -vel_mean / rr,
    }
    if data.metadata.upper_body_bicycle_rider:
        data.simulator.initial_conditions = {
            **data.simulator.initial_conditions,
            # Some initial guesses for the arm angles.
            data.rider.left_arm.q[0]: 0.7,
            data.rider.right_arm.q[0]: 0.7,
            data.rider.left_shoulder.q[0]: 0.5,
            data.rider.left_shoulder.q[1]: -0.6,
            data.rider.right_shoulder.q[0]: 0.5,
            data.rider.right_shoulder.q[1]: -0.6,
        }
    t_arr, x_arr = data.simulator.solve(
        np.linspace(0, data.metadata.duration, data.metadata.num_nodes), "dae",
        rtol=1e-3, atol=1e-6)
    if t_arr[-1] != data.metadata.duration:
        print("DAE integration failed, integrating with solve_ivp.")
        t_arr, x_arr = data.simulator.solve(
            (0, data.metadata.duration), "solve_ivp",
            t_eval=np.linspace(0, data.metadata.duration, data.metadata.num_nodes))

    data.initial_guess = np.concatenate(
        (x_arr.ravel(), np.zeros(len(data.input_vars) * data.metadata.num_nodes)))
