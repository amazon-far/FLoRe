"""Shared gripper position defaults for the YAM linear_4310 hands.

Kept dependency-free so the ROS node (``hand_pipeline_node``) and the
stand-alone CAN shake-out (``gripper_test``) can both import these
without pulling rclpy / CAN deps into each other.

Sim-frame semantics: positive motor rad = open. ``motor_direction=-1``
is applied inside the driver, so these values convert to negative raw
motor rad at the hardware.
"""

GRIPPER_OPEN_POS = 3.5
GRIPPER_CLOSED_POS = 0.0
