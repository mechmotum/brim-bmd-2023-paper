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

    d_long = data.metadata.longitudinal_displacement
    d_lat = data.metadata.lateral_displacement
    d_tot = np.sqrt(d_long ** 2 + d_lat ** 2)
    p, p_vals = zip(*constants.items())
    vel_mean = d_tot / data.metadata.duration
    rr = data.constants[data.bicycle.rear_wheel.radius]

    #####################
    # Solve initial state
    #####################
    x0 = np.array([initial_state_constraints.get(xi, 0.0) for xi in data.x])
    x0[data.x[:].index(bicycle.q[2])] = np.arctan2(d_lat, d_long)
    x0[data.x[:].index(bicycle.u[0])] = vel_mean
    x0[data.x[:].index(bicycle.u[5])] = -vel_mean / rr

    qdot_to_u = system.eom_method.kindiffdict()
    velocity_constraints = me.msubs(
        system.holonomic_constraints.diff(me.dynamicsymbols._t).col_join(
            system.nonholonomic_constraints), qdot_to_u)
    eval_configuration_constraints = sm.lambdify(
        (system.q_dep, system.q_ind, p), system.holonomic_constraints[:], cse=True)
    eval_velocity_constraints = sm.lambdify(
        (system.u_dep, system.u_ind, system.q, p),
        velocity_constraints[:], cse=True)

    q0 = x0[:len(system.q)]
    q0_ind = q0[:len(system.q_ind)]
    q0_dep_guess = q0[len(system.q_ind):]
    u0_ind = x0[len(system.q):-len(system.u_dep)]
    u0_dep_guess = x0[-len(system.u_dep):]
    q0_dep = fsolve(
        eval_configuration_constraints, q0_dep_guess, args=(q0_ind, p_vals))

    eval_velocity_constraints(u0_dep_guess, u0_ind, q0, p_vals)
    u0_dep = fsolve(
        eval_velocity_constraints, u0_dep_guess, args=(u0_ind, q0, p_vals))
    x0 = np.concatenate((q0_ind, q0_dep, u0_ind, u0_dep))

    # TODO Perform simulation
    # eval_eoms = sm.lambdify((data.x.diff(), data.x, p), data.eoms[:], cse=True)
    # def eqsres(t, x, xd, residual):
    #     residual[:] = eval_eoms(xd, x, p_vals)
    # xd0 = fsolve(eval_eoms, x0, args=(x0, p_vals))
    # dae_solver = dae('ida', eqsres,
    #                  algebraic_vars_idx=range(len(q0_ind) + len(u0_ind), len(x0)),
    #                  old_api=False)
    # sol = dae_solver.solve(
    #     np.linspace(0, data.metadata.duration, data.metadata.num_nodes), x0, xd0)

    # Extrapolate initial state guess.
    xf = np.concatenate((q0_ind, q0_dep, u0_ind, u0_dep))
    xf[data.x[:].index(bicycle.q[0])] = d_long
    xf[data.x[:].index(bicycle.q[1])] = d_lat
    x_arr = np.zeros([len(data.x), data.metadata.num_nodes])
    for i in range(x_arr.shape[0]):
        x_arr[i, :] = (np.linspace(x0[i], xf[i], data.metadata.num_nodes)
                       )  # + np.random.normal(0.0, 0.01, data.metadata.num_nodes))
    return np.concatenate(
        (x_arr.flatten(),
         0.0 * np.ones(len(data.input_vars) * data.metadata.num_nodes)))
