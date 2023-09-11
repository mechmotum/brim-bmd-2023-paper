from typing import Any

from brim.core import Attachment
from brim.rider.base_connections import LeftShoulderBase, RightShoulderBase
from sympy import Matrix
from sympy.physics.mechanics import PinJoint, ReferenceFrame, dynamicsymbols, Point
from sympy.physics.mechanics._system import System


class FlexRotShoulderMixin:
    """Shoulder joint with flexion and rotation."""

    @property
    def descriptions(self) -> dict[Any, str]:
        """Descriptions of the objects."""
        return {
            **super().descriptions,
            self.q[0]: "Flexion angle of the shoulder.",
            self.q[1]: "Endorotation angle of the shoulder.",
            self.u[0]: "Flexion angular velocity of the shoulder.",
            self.u[1]: "Rotation angular velocity of the shoulder.",
        }

    def _define_objects(self) -> None:
        """Define the objects."""
        super()._define_objects()
        self.q = Matrix(dynamicsymbols(self._add_prefix("q_flexion, q_rotation")))
        self.u = Matrix(dynamicsymbols(self._add_prefix("u_flexion, u_rotation")))
        self._system = System.from_newtonian(self.torso.body)
        self._intermediate = Attachment(ReferenceFrame(self._add_prefix("int_frame")),
                                        Point(self._add_prefix("int_point")))


class FlexRotLeftShoulder(FlexRotShoulderMixin, LeftShoulderBase):
    """Left shoulder joint with flexion and rotation."""

    def _define_kinematics(self) -> None:
        """Define the kinematics."""
        super()._define_kinematics()
        self.system.add_joints(
            PinJoint(
                self._add_prefix("flexion_joint"), self.torso.body,
                self._intermediate.to_valid_joint_arg(), self.q[0],
                self.u[0], self.torso.left_shoulder_point, self._intermediate.point,
                self.torso.left_shoulder_frame, self._intermediate.frame,
                joint_axis=self.torso.y),
            PinJoint(
                self._add_prefix("rotation_joint"),
                self._intermediate.to_valid_joint_arg(), self.arm.shoulder, self.q[1],
                self.u[1], self._intermediate.point, self.arm.shoulder_interpoint,
                self._intermediate.frame, self.arm.shoulder_interframe,
                joint_axis=self._intermediate.frame.z)
        )


class FlexRotRightShoulder(FlexRotShoulderMixin, RightShoulderBase):
    """Right shoulder joint with flexion and rotation."""

    def _define_kinematics(self) -> None:
        """Define the kinematics."""
        super()._define_kinematics()
        self.system.add_joints(
            PinJoint(
                self._add_prefix("flexion_joint"), self.torso.body,
                self._intermediate.to_valid_joint_arg(), self.q[0],
                self.u[0], self.torso.right_shoulder_point, self._intermediate.point,
                self.torso.right_shoulder_frame, self._intermediate.frame,
                joint_axis=self.torso.y),
            PinJoint(
                self._add_prefix("rotation_joint"),
                self._intermediate.to_valid_joint_arg(), self.arm.shoulder, self.q[1],
                self.u[1], self._intermediate.point, self.arm.shoulder_interpoint,
                self._intermediate.frame, self.arm.shoulder_interframe,
                joint_axis=-self._intermediate.frame.z)
        )


class FlexAddShoulderMixin:
    """Shoulder joint with flexion and adduction."""

    @property
    def descriptions(self) -> dict[Any, str]:
        """Descriptions of the objects."""
        return {
            **super().descriptions,
            self.q[0]: "Flexion angle of the shoulder.",
            self.q[1]: "Adduction angle of the shoulder.",
            self.u[0]: "Flexion angular velocity of the shoulder.",
            self.u[1]: "Adduction angular velocity of the shoulder.",
        }

    def _define_objects(self) -> None:
        """Define the objects."""
        super()._define_objects()
        self.q = Matrix(dynamicsymbols(self._add_prefix("q_flexion, q_adduction")))
        self.u = Matrix(dynamicsymbols(self._add_prefix("u_flexion, u_adduction")))
        self._system = System.from_newtonian(self.torso.body)
        self._intermediate = Attachment(ReferenceFrame(self._add_prefix("int_frame")),
                                        Point(self._add_prefix("int_point")))


class FlexAddLeftShoulder(FlexAddShoulderMixin, LeftShoulderBase):
    """Left shoulder joint with flexion and adduction."""

    def _define_kinematics(self) -> None:
        """Define the kinematics."""
        super()._define_kinematics()
        self.system.add_joints(
            PinJoint(
                self._add_prefix("flexion_joint"), self.torso.body,
                self._intermediate.to_valid_joint_arg(), self.q[0],
                self.u[0], self.torso.left_shoulder_point, self._intermediate.point,
                self.torso.left_shoulder_frame, self._intermediate.frame,
                joint_axis=self.torso.y),
            PinJoint(
                self._add_prefix("adduction_joint"),
                self._intermediate.to_valid_joint_arg(), self.arm.shoulder, self.q[1],
                self.u[1], self._intermediate.point, self.arm.shoulder_interpoint,
                self._intermediate.frame, self.arm.shoulder_interframe,
                joint_axis=-self._intermediate.frame.x)
        )


class FlexAddRightShoulder(FlexAddShoulderMixin, RightShoulderBase):
    """Right shoulder joint with flexion and adduction."""

    def _define_kinematics(self) -> None:
        """Define the kinematics."""
        super()._define_kinematics()
        self.system.add_joints(
            PinJoint(
                self._add_prefix("flexion_joint"), self.torso.body,
                self._intermediate.to_valid_joint_arg(), self.q[0],
                self.u[0], self.torso.right_shoulder_point, self._intermediate.point,
                self.torso.right_shoulder_frame, self._intermediate.frame,
                joint_axis=self.torso.y),
            PinJoint(
                self._add_prefix("adduction_joint"),
                self._intermediate.to_valid_joint_arg(), self.arm.shoulder, self.q[1],
                self.u[1], self._intermediate.point, self.arm.shoulder_interpoint,
                self._intermediate.frame, self.arm.shoulder_interframe,
                joint_axis=self._intermediate.frame.x)
        )
