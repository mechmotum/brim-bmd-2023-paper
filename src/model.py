import bicycleparameters as bp
import brim as bm
import brim.rider as bmr
import sympy as sm
import sympy.physics.mechanics as me
from brim.utilities.parametrize import get_inertia_vals
from dtk.bicycle import benchmark_to_moore
from scipy.optimize import fsolve

from .brim_extra import (FlexRotLeftShoulder, FlexRotRightShoulder,
                         FlexAddLeftShoulder, FlexAddRightShoulder)
from .container import SteerWith, DataStorage, ShoulderJointType
from .simulator import Simulator
from .utils import get_all_symbols_from_model


def set_bicycle_model(data: DataStorage):
    input_vars = sm.ImmutableMatrix()  # Input variables.

    # Configure the bicycle model.
    bicycle = bm.WhippleBicycle("bicycle")
    if data.metadata.front_frame_suspension:
        bicycle.front_frame = bm.SuspensionRigidFrontFrame("front_frame")
    else:
        bicycle.front_frame = bm.RigidFrontFrame("front_frame")
    bicycle.rear_frame = bm.RigidRearFrame("rear_frame")
    bicycle.front_wheel = bm.KnifeEdgeWheel("front_wheel")
    bicycle.rear_wheel = bm.KnifeEdgeWheel("rear_wheel")
    bicycle.front_tyre = bm.NonHolonomicTyre("front_tyre")
    bicycle.rear_tyre = bm.NonHolonomicTyre("rear_tyre")
    bicycle.ground = bm.FlatGround("ground")

    bicycle_rider = bm.BicycleRider("bicycle_rider")
    bicycle_rider.bicycle = bicycle

    if data.metadata.model_upper_body:
        rider = bm.Rider("rider")
        rider.pelvis = bm.PlanarPelvis("pelvis")
        rider.torso = bm.PlanarTorso("torso")
        rider.sacrum = bm.FixedSacrum("sacrum")
        rider.left_arm = bm.PinElbowStickLeftArm("left_arm")
        rider.right_arm = bm.PinElbowStickRightArm("right_arm")
        if data.metadata.shoulder_type == ShoulderJointType.FLEX_ROT:
            rider.left_shoulder = FlexRotLeftShoulder("left_shoulder")
            rider.right_shoulder = FlexRotRightShoulder("right_shoulder")
        elif data.metadata.shoulder_type == ShoulderJointType.FLEX_ADD:
            rider.left_shoulder = FlexAddLeftShoulder("left_shoulder")
            rider.right_shoulder = FlexAddRightShoulder("right_shoulder")
        elif data.metadata.shoulder_type == ShoulderJointType.SPHERICAL:
            rider.left_shoulder = bm.SphericalLeftShoulder("left_shoulder")
            rider.right_shoulder = bm.SphericalRightShoulder("right_shoulder")
        bicycle_rider.rider = rider
        bicycle_rider.seat = bm.FixedSeat("seat")
        bicycle_rider.hand_grips = bm.HolonomicHandGrips("hand_grips")
        if data.metadata.steer_with == SteerWith.HUMAN_TORQUE:
            let = bmr.PinElbowTorque("left_elbow_torque")
            ret = bmr.PinElbowTorque("right_elbow_torque")
            rider.left_arm.add_load_groups(let)
            rider.right_arm.add_load_groups(ret)

    # Define the model.
    bicycle_rider.define_connections()
    bicycle_rider.define_objects()
    if data.metadata.model_upper_body:
        alpha = sm.Symbol("alpha")
        int_frame = me.ReferenceFrame("int_frame")
        int_frame.orient_axis(bicycle.rear_frame.saddle.frame, alpha,
                              bicycle.rear_frame.wheel_hub.axis)
        bicycle_rider.seat.rear_interframe = int_frame
    bicycle_rider.define_kinematics()
    bicycle_rider.define_loads()
    bicycle_rider.define_constraints()

    # Export model to a system object.
    system = bicycle_rider.to_system()

    # Apply additional forces and torques to the system.
    g = sm.Symbol("g")
    system.apply_gravity(-g * bicycle.ground.get_normal(bicycle.ground.origin))
    pedal_torque = me.dynamicsymbols("pedal_torque")
    system.add_loads(
        me.Torque(bicycle.rear_wheel.body,
                  pedal_torque * bicycle.rear_wheel.rotation_axis)
    )
    input_vars = input_vars.col_join(sm.Matrix([pedal_torque]))
    if data.metadata.steer_with == SteerWith.PEDAL_STEER_TORQUE:
        steer_torque = me.dynamicsymbols("steer_torque")
        system.add_actuators(me.TorqueActuator(
            steer_torque, bicycle.rear_frame.steer_hub.axis,
            bicycle.front_frame.steer_hub.frame, bicycle.rear_frame.steer_hub.frame))
        input_vars = input_vars.col_join(sm.Matrix([steer_torque]))
    elif data.metadata.steer_with == SteerWith.HUMAN_TORQUE:
        input_vars = input_vars.col_join(
            sm.Matrix([let.symbols["T"], ret.symbols["T"]]))

    # Specify the independent and dependent generalized coordinates and speeds.
    system.q_ind = [*bicycle.q[:4], *bicycle.q[5:]]
    system.q_dep = [bicycle.q[4]]
    system.u_ind = [bicycle.u[3], *bicycle.u[5:7]]
    system.u_dep = [*bicycle.u[:3], bicycle.u[4], bicycle.u[7]]
    if data.metadata.front_frame_suspension:
        system.add_coordinates(bicycle.front_frame.q[0], independent=True)
        system.add_speeds(bicycle.front_frame.u[0], independent=True)
    if data.metadata.model_upper_body:
        if data.metadata.shoulder_type == ShoulderJointType.SPHERICAL:
            system.add_coordinates(rider.left_shoulder.q[1], rider.right_shoulder.q[1],
                                   independent=True)
            system.add_speeds(rider.left_shoulder.u[1], rider.right_shoulder.u[1],
                              independent=True)
            system.add_coordinates(rider.left_shoulder.q[0], rider.left_shoulder.q[2],
                                   rider.right_shoulder.q[0], rider.right_shoulder.q[2],
                                   rider.left_arm.q[0], rider.right_arm.q[0],
                                   independent=False)
            system.add_speeds(rider.left_shoulder.u[0], rider.left_shoulder.u[2],
                              rider.right_shoulder.u[0], rider.right_shoulder.u[2],
                              rider.left_arm.u[0], rider.right_arm.u[0],
                              independent=False)
        else:
            system.add_coordinates(*rider.left_shoulder.q, *rider.right_shoulder.q,
                                   *rider.left_arm.q, *rider.right_arm.q,
                                   independent=False)
            system.add_speeds(*rider.left_shoulder.u, *rider.right_shoulder.u,
                              *rider.left_arm.u, *rider.right_arm.u,
                              independent=False)

    # Simple check to see if the system is valid.
    system.validate_system()
    # Form the equations of motion. Note: LU solve may lead to zero divisions.
    essential_eoms = system.form_eoms(constraint_solver="CRAMER")
    eoms = system.kdes.col_join(essential_eoms).col_join(
        system.holonomic_constraints).col_join(system.nonholonomic_constraints)

    # Obtain constant parameters.
    bicycle_params = bp.Bicycle(
        data.metadata.bicycle_parametrization,
        pathToData=data.metadata.parameter_data_dir)
    if data.metadata.bicycle_parametrization == "Fisher":
        bicycle_params.parameters["Measured"]["hbb"] = 0.3
        bicycle_params.parameters["Measured"]["lcs"] = 0.44
        bicycle_params.parameters["Measured"]["lsp"] = 0.22
        bicycle_params.parameters["Measured"]["lst"] = 0.53
        bicycle_params.parameters["Measured"]["lamst"] = 1.29
        bicycle_params.parameters["Measured"]["whb"] = 0.6
        bicycle_params.parameters["Measured"]["LhbR"] = 1.11
        bicycle_params.parameters["Measured"]["LhbF"] = 0.65
    if not data.metadata.bicycle_only:
        bicycle_params.add_rider(data.metadata.rider_parametrization, reCalc=True)
    constants = bicycle_rider.get_param_values(bicycle_params)
    constants[g] = 9.81
    if data.metadata.front_frame_suspension:
        constants[bicycle.front_frame.symbols["d9"]] = \
            constants[bicycle.front_frame.symbols["d3"]] / 2
        # Suspension spring and damper constants are the softest settings provided in:
        # http://dx.doi.org/10.13140/RG.2.2.26063.64162
        constants[bicycle.front_frame.symbols["k"]] = 19.4E3  # 42.6E3
        constants[bicycle.front_frame.symbols["c"]] = 9E3
    if data.metadata.model_upper_body:
        approximate_alpha = -(bicycle_params.parameters["Benchmark"]["lam"] +
                              bicycle_params.human.CFG["somersault"]).nominal_value
        # The somersault angle is generally based on straight arms, therefore the value
        # should be a bit higher to account for the bent arms and steering.
        if data.metadata.bicycle_parametrization == "Browser":
            offset = 0.125
        else:
            offset = 0.1
        constants[alpha] = approximate_alpha - offset

    syms = get_all_symbols_from_model(bicycle_rider)
    missing_constants = syms.difference(constants.keys()).difference({
        bicycle.symbols["gear_ratio"], 0, *input_vars})
    if data.metadata.model_upper_body:
        missing_constants = missing_constants.difference(
            bicycle_rider.seat.symbols.values())
    if missing_constants:
        rear_constants_estimates = {
            bicycle.rear_frame.symbols["d4"]: 0.42,
            bicycle.rear_frame.symbols["d5"]: -0.55,
            bicycle.rear_frame.symbols["l_bbx"]: 0.40,
            bicycle.rear_frame.symbols["l_bbz"]: 0.22,
        }
        front_constants_estimates = {
            bicycle.front_frame.symbols["d6"]: -0.17,
            bicycle.front_frame.symbols["d7"]: 0.29,
            bicycle.front_frame.symbols["d8"]: -0.37,
        }

        if (data.metadata.model_upper_body and
                missing_constants.difference(rear_constants_estimates.keys())):
            raise ValueError(f"Missing constants: {missing_constants}")
        elif missing_constants.difference(rear_constants_estimates.keys()).difference(
                front_constants_estimates.keys()):
            raise ValueError(f"Missing constants: {missing_constants}")
        estimated_constants = {
            sym: rear_constants_estimates.get(sym, front_constants_estimates.get(sym))
            for sym in missing_constants
        }
        print(f"Estimated constants, which are used for visualization purposes only: "
              f"{estimated_constants}.")
        constants.update(estimated_constants)

    # Include missing rider mass into the rear frame
    rear_body = bicycle.rear_frame.body
    if data.metadata.model_upper_body:
        # Add the inertia of the legs to the rear frame
        leg = bm.TwoPinStickLeftLeg("left_leg")
        q_hip = me.dynamicsymbols("q_hip")
        leg.define_all()
        leg.hip_interframe.orient_axis(bicycle.rear_frame.saddle.frame, q_hip,
                                       bicycle.rear_frame.wheel_hub.axis)
        offset = rider.pelvis.symbols["hip_width"] * bicycle.rear_frame.saddle.frame.y
        leg.hip_interpoint.set_pos(
            rider.pelvis.left_hip_point,
            rider.pelvis.right_hip_point.pos_from(rider.pelvis.left_hip_point) / 2)
        val_dict = {leg.q[1]: 0, **leg.get_param_values(bicycle_params), **constants}
        v = leg.foot_interpoint.pos_from(bicycle.rear_frame.bottom_bracket).to_matrix(
            bicycle.rear_frame.wheel_hub.frame).xreplace(val_dict).simplify()
        val_dict[q_hip], val_dict[leg.q[0]] = fsolve(
            sm.lambdify([(q_hip, leg.q[0])], [v[0], v[2]]), (0.6, 1.5))
        additional_inertia = me.Dyadic(0)
        additional_mass = sm.S.Zero
        for body in leg.system.bodies:
            additional_inertia += 2 * body.parallel_axis(rear_body.masscenter)
            additional_inertia += 2 * body.mass * (me.inertia(
                rear_body.frame, 1, 1, 1) * offset.dot(offset) - offset.outer(offset))
            additional_mass += 2 * body.mass
        extra_i_vals = sm.lambdify(
            val_dict.keys(), additional_inertia.to_matrix(rear_body.frame),
            cse=True)(*val_dict.values())
        i_rear = rear_body.central_inertia.to_matrix(rear_body.frame)
        constants[rear_body.mass] += float(additional_mass.xreplace(val_dict))
        for idx in [(0, 0), (1, 1), (2, 2), (2, 0)]:
            constants[i_rear[idx]] += float(extra_i_vals[idx])
    elif not data.metadata.bicycle_only:
        params = bp.io.remove_uncertainties(bicycle_params.parameters["Benchmark"])
        mop = benchmark_to_moore(params)
        constants[rear_body.mass] = mop["mc"]
        constants.update(get_inertia_vals(
            rear_body, mop["ic11"], mop["ic22"], mop["ic33"], mop["ic12"],
            mop["ic23"], mop["ic31"]))
        constants[bicycle.rear_frame.symbols["l1"]] = mop["l1"]
        constants[bicycle.rear_frame.symbols["l2"]] = mop["l2"]

    data.bicycle_rider = bicycle_rider
    data.bicycle = bicycle
    if data.metadata.model_upper_body:
        data.rider = rider
    data.system = system
    data.eoms = eoms
    data.constants = constants
    data.input_vars = input_vars


def set_simulator(data: DataStorage) -> None:
    simulator = Simulator(data.system)
    simulator.constants = data.constants
    simulator.inputs = {ri: lambda t, x: 0.0 for ri in data.input_vars}
    simulator.initial_conditions = {xi: 0.0 for xi in data.x}
    simulator.initial_conditions[data.bicycle.q[4]] = 0.314
    simulator.initialize(False)
    data.simulator = simulator
