from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto, unique

import brim as bm
import numpy as np
import numpy.typing as npt
import sympy as sm
from opty.direct_collocation import Problem
from sympy.physics.mechanics._system import System


@unique
class SteerWith(Enum):
    """Enumeration of options for controlling the bicycle steering."""
    PEDAL_STEER_TORQUE = auto()
    HUMAN_TORQUE = auto()


@dataclass(frozen=True)
class Metadata:
    front_frame_suspension: bool
    upper_body_bicycle_rider: bool
    steer_with: SteerWith
    parameter_data_dir: str
    bicycle_parametrization: str
    duration: float
    longitudinal_displacement: float
    lateral_displacement: float
    straight_lengths: float
    num_nodes: int
    weight: float

    @property
    def interval_value(self):
        return self.duration / (self.num_nodes - 1)


@dataclass
class DataStorage:
    """Data storage object."""
    metadata: Metadata
    bicycle_rider: bm.BicycleRider | None = None
    bicycle: bm.WhippleBicycle | None = None
    rider: bm.Rider | None = None
    system: System | None = None
    eoms: sm.ImmutableMatrix | None = None
    input_vars: sm.ImmutableMatrix | None = None
    constants: dict[sm.Basic, float] | None = None
    problem: Problem | None = None
    initial_guess: npt.NDArray[np.float_] | None = None
    solution: npt.NDArray[np.float_] | None = None

    def __getstate__(self):
        # Problem cannot be pickled.
        return dict(
            metadata=self.metadata,
            bicycle_rider=self.bicycle_rider,
            bicycle=self.bicycle,
            rider=self.rider,
            system=self.system,
            eoms=self.eoms,
            input_vars=self.input_vars,
            constants=self.constants,
            initial_guess=self.initial_guess,
            solution=self.solution,
        )

    def __setstate__(self, state):
        for key, value in state.items():
            setattr(self, key, value)

    @property
    def x(self) -> sm.ImmutableMatrix:
        """State variables."""
        return self.system.q.col_join(self.system.u)

    @property
    def r(self) -> sm.ImmutableMatrix:
        """Input variables."""
        return self.input_vars

    @property
    def time_array(self) -> npt.NDArray[np.float_]:
        """Time array."""
        return np.linspace(0, self.metadata.duration, self.metadata.num_nodes)

    @property
    def solution_state(self) -> npt.NDArray[np.float_]:
        """State trajectory from the solution."""
        n, N = self.x.shape[0], self.metadata.num_nodes
        return self.solution[:n * N].reshape((n, N))

    @property
    def solution_input(self) -> npt.NDArray[np.float_]:
        """Input trajectory from the solution."""
        n, q, N = self.x.shape[0], self.r.shape[0], self.metadata.num_nodes
        return self.solution[n * N:(q + n) * N].reshape((q, N))

    @property
    def target(self) -> sm.Expr:
        """Target path."""
        cos_trans = lambda x: (1 - sm.cos(sm.pi * x)) / 2
        s = self.metadata.straight_lengths
        d_lat = self.metadata.lateral_displacement
        d_long = self.metadata.longitudinal_displacement
        x, y = self.bicycle.q[0], self.bicycle.q[1]
        return y - sm.Piecewise(
            (0, x < s),
            (d_lat * cos_trans((x - s) / (d_long - 2 * s)), x <= d_long - s),
            (d_lat, True))
