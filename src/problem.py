from __future__ import annotations

import numpy as np
import numpy.typing as npt
import sympy as sm
import sympy.physics.mechanics as me
from opty.direct_collocation import Problem
from scipy.optimize import fsolve

from container import DataStorage, SteerWith
from utils import create_objective_function, plot_constraint_violations

# Corrected version from opty's ``Problem.plot_constraint_violations``.
plot_constraint_violations.__doc__ = Problem.plot_constraint_violations.__doc__
Problem.plot_constraint_violations = plot_constraint_violations


def set_problem(data: DataStorage) -> None:
    t = me.dynamicsymbols._t  # Time symbol.
    t0, tf = 0.0, data.metadata.duration  # Initial and final time.
    vel_mean = data.metadata.longitudinal_displacement / tf
    rr = data.constants[data.bicycle.rear_wheel.radius]
    rf = data.constants[data.bicycle.front_wheel.radius]

    bicycle = data.bicycle

    initial_state_constraints = {
        bicycle.q[0]: 0.0,
        bicycle.q[1]: 0.0,
        bicycle.q[5]: 0.0,
        bicycle.q[7]: 0.0,
        bicycle.u[0]: vel_mean,
        bicycle.u[5]: -vel_mean / rr,
        bicycle.u[7]: -vel_mean / rf,
        bicycle.front_frame.q[0]: 0.0,  # Rough estimate.
    }
    final_state_constraints = {
        bicycle.q[0]: data.metadata.longitudinal_displacement,
        bicycle.q[1]: data.metadata.lateral_displacement,
        bicycle.u[0]: vel_mean,
        bicycle.u[5]: - vel_mean / rr,
        bicycle.u[7]: -vel_mean / rf,
    }

    instance_constraints = (
        bicycle.q[2].replace(t, t0) - bicycle.q[2].replace(t, tf),  # Periodic yaw.
        bicycle.q[3].replace(t, t0) - bicycle.q[3].replace(t, tf),  # Periodic roll.
        bicycle.q[4].replace(t, t0) - bicycle.q[4].replace(t, tf),  # Periodic pitch.
        # Periodic steering rotation angle.
        bicycle.q[6].replace(t, t0) - bicycle.q[6].replace(t, tf),
        # Periodic velocities.
        bicycle.u[1].replace(t, t0) - bicycle.u[1].replace(t, tf),
        bicycle.u[2].replace(t, t0) - bicycle.u[2].replace(t, tf),
        bicycle.u[3].replace(t, t0) - bicycle.u[3].replace(t, tf),
        bicycle.u[4].replace(t, t0) - bicycle.u[4].replace(t, tf),
        bicycle.u[6].replace(t, t0) - bicycle.u[6].replace(t, tf),
    )
    instance_constraints += tuple(
        xi.replace(t, t0) - xi_val for xi, xi_val in initial_state_constraints.items()
    ) + tuple(
        xi.replace(t, tf) - xi_val for xi, xi_val in final_state_constraints.items()
    )

    if data.metadata.front_frame_suspension:
        instance_constraints += (bicycle.front_frame.q[0].replace(t, t0) - 0.0,)

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
            bicycle.front_frame.q[0]: (-0.3, 0.3),
            bicycle.front_frame.u[0]: (-50.0, 50.0),
        })
    if data.metadata.steer_with is SteerWith.PEDAL_STEER_TORQUE:
        bounds.update({
            data.input_vars[0]: (-100.0, 100.0),
            data.input_vars[1]: (-100.0, 100.0),
        })

    objective_expr = (data.metadata.weight * (data.target) ** 2 +
                      (1 - data.metadata.weight) * sum(i ** 2 for i in data.input_vars))

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
    data.initial_guess = generate_initial_guess(data, initial_state_constraints,
                                                final_state_constraints)


def generate_initial_guess(data: DataStorage, initial_state_constraints,
                           final_state_constraints) -> npt.NDArray[np.float64]:
    system, bicycle, constants = data.system, data.bicycle, data.constants
    q_fix_idx = np.argpartition([
        5 * (xi in initial_state_constraints) +
        5 * (xi in final_state_constraints) +
        1 * (xi in system.q_ind)
        for xi in system.q
    ], -len(system.q_ind))[-len(system.q_ind):]
    u_fix_idx = np.argpartition([
        5 * (xi in initial_state_constraints) +
        5 * (xi in final_state_constraints) +
        1 * (xi in system.u_ind)
        for xi in system.u
    ], -len(system.u_ind))[-len(system.u_ind):]
    q_fix_idx = range(7)
    u_fix_idx = [system.u[:].index(bicycle.u[0]),
                 system.u[:].index(bicycle.u[3]),
                 system.u[:].index(bicycle.u[6])]
    if data.metadata.front_frame_suspension:
        q_fix_idx = range(8)
        u_fix_idx += [system.u[:].index(bicycle.u[2])]
    q_free_idx = [xi for xi in range(len(system.q)) if xi not in q_fix_idx]
    u_free_idx = [xi for xi in range(len(system.u)) if xi not in u_fix_idx]
    q_fixed = [system.q[i] for i in q_fix_idx]
    q_free = [system.q[i] for i in q_free_idx]
    u_fixed = [system.u[i] for i in u_fix_idx]
    u_free = [system.u[i] for i in u_free_idx]
    p, p_vals = zip(*constants.items())
    nq = len(system.q)

    qdot_to_u = system.eom_method.kindiffdict()
    velocity_constraints = me.msubs(
        system.holonomic_constraints.diff(me.dynamicsymbols._t).col_join(
            system.nonholonomic_constraints), qdot_to_u)
    eval_configuration_constraints = sm.lambdify(
        (q_free, q_fixed, p), system.holonomic_constraints[:], cse=True)
    eval_velocity_constraints = sm.lambdify(
        (u_free, u_fixed, system.q, p),
        velocity_constraints[:], cse=True)

    x0 = np.array([initial_state_constraints.get(xi, 0.0) for xi in data.x])
    xf = np.array([final_state_constraints.get(xi, 0.0) for xi in data.x])
    x_arr = np.zeros([len(data.x), data.metadata.num_nodes])
    for i in range(x_arr.shape[0]):
        x_arr[i, :] = (np.linspace(x0[i], xf[i], data.metadata.num_nodes) +
                       np.random.normal(0.0, 0.01, data.metadata.num_nodes))
    for i in range(x_arr.shape[1]):
        x_arr[q_free_idx, i] = fsolve(
            eval_configuration_constraints, x_arr[q_free_idx, i],
            args=(x_arr[q_fix_idx, i], p_vals))
        x_arr[u_free_idx, i] = fsolve(
            eval_velocity_constraints, x_arr[u_free_idx, i],
            args=(x_arr[u_fix_idx, i], x_arr[:nq, i], p_vals))
    return np.concatenate(
        (x_arr.flatten(),
         0.01 * np.ones(len(data.input_vars) * data.metadata.num_nodes)))
