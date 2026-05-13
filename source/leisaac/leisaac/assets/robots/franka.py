from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

# Stiffer PD gains (HIGH_PD variant) give better IK tracking for teleoperation.
# init_state pos/rot left at defaults (0,0,0 / identity) so the env cfg can
# position the robot relative to the scene.
FRANKA_PANDA_LEISAAC_CFG = FRANKA_PANDA_HIGH_PD_CFG.copy()

FRANKA_ARM_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]
FRANKA_FINGER_JOINT_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]
FRANKA_ALL_JOINT_NAMES = FRANKA_ARM_JOINT_NAMES + FRANKA_FINGER_JOINT_NAMES
