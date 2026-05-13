from dataclasses import MISSING
from typing import Any

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import (
    ActionStateRecorderManagerCfg as RecordTerm,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg, TiledCameraCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, ArticulationRootPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.datasets.episode_data import EpisodeData
from leisaac.enhance.datasets.lerobot_dataset_handler import LeRobotDatasetCfg
from leisaac.assets.robots.franka import (
    FRANKA_PANDA_LEISAAC_CFG,
    FRANKA_ALL_JOINT_NAMES,
    FRANKA_ARM_JOINT_NAMES,
    FRANKA_FINGER_JOINT_NAMES,
)

import isaaclab.envs.mdp as mdp

from . import mdp as task_mdp


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

@configclass
class FrankaNutThreadSceneCfg(InteractiveSceneCfg):
    """Scene: ground plane + SeattleLabTable + Franka + bolt (fixed) + nut (free)."""

    # Ground plane — global prim, shared across all envs
    ground_plane: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=GroundPlaneCfg(),
    )

    # SeattleLabTable — same placement used by IsaacLab's Franka manipulation tasks
    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.707, 0.0, 0.0, 0.707)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        ),
    )

    # Franka Panda robot — base at origin, facing +x toward the table/bolt
    robot: ArticulationCfg = FRANKA_PANDA_LEISAAC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Bolt M16 — fixed to the table surface via fix_root_link=True.
    # The factory USD has an articulation root, so ArticulationCfg is required.
    # Position: on the table surface (z≈0) directly in front of the robot.
    # Tune x/y if the bolt is out of the gripper's reach.
    bolt: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Bolt",
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), joint_pos={}, joint_vel={}),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Factory/factory_bolt_m16.usd",
            rigid_props=RigidBodyPropertiesCfg(disable_gravity=True),
            articulation_props=ArticulationRootPropertiesCfg(fix_root_link=True),
        ),
        actuators={},
    )

    # Nut M16 — free rigid body the user picks up and threads onto the bolt.
    # Starts on the table slightly beside the bolt so it's easy to grasp.
    # Tune position to match the initial Franka joint configuration.
    # Nut M16 — free to move, picked up and threaded by the user.
    # Factory USD also has an articulation root, so ArticulationCfg is required.
    nut: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Nut",
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.5, 0.15, 0.005), joint_pos={}, joint_vel={}),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Factory/factory_nut_m16.usd",
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=ArticulationRootPropertiesCfg(fix_root_link=False),
        ),
        actuators={},
    )

    # End-effector frame tracker (root=panda_link0, target=panda_hand)
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="gripper",
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="jaw",
                # Offset forward along the finger approach axis so object detection
                # lines up with the fingertip gap. Tune if needed.
                offset=OffsetCfg(pos=(0.0, 0.0, 0.107)),
            ),
        ],
    )

    # Wrist camera mounted on panda_hand
    wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, -0.08), rot=(0.0, 0.7071, 0.7071, 0.0), convention="ros"
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,
    )

    # Front camera looking at the workspace
    front: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/front_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.2, 0.0, 0.8), rot=(0.0, -0.3827, 0.9239, 0.0), convention="ros"
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,
    )

    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


# ---------------------------------------------------------------------------
# MDP building blocks
# ---------------------------------------------------------------------------

@configclass
class ActionsCfg:
    """IK arm + relative finger gripper — 6D EE delta + 2D finger = 8D total action."""

    arm_action: mdp.ActionTermCfg = MISSING
    gripper_action: mdp.ActionTermCfg = MISSING


@configclass
class EventCfg:
    reset_all = EventTerm(func=task_mdp.reset_scene_to_default, mode="reset")


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=task_mdp.joint_pos)
        joint_vel = ObsTerm(func=task_mdp.joint_vel)
        joint_pos_rel = ObsTerm(func=task_mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=task_mdp.joint_vel_rel)
        actions = ObsTerm(func=task_mdp.last_action)
        wrist = ObsTerm(
            func=task_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("wrist"), "data_type": "rgb", "normalize": False},
        )
        front = ObsTerm(
            func=task_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("front"), "data_type": "rgb", "normalize": False},
        )
        ee_frame_state = ObsTerm(
            func=task_mdp.ee_frame_state,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame"), "robot_cfg": SceneEntityCfg("robot")},
        )
        joint_pos_target = ObsTerm(func=task_mdp.joint_pos_target, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    pass


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)


# ---------------------------------------------------------------------------
# Main env config
# ---------------------------------------------------------------------------

@configclass
class FrankaNutThreadEnvCfg(ManagerBasedRLEnvCfg):
    """Franka Panda + M16 bolt/nut threading environment for leisaac teleoperation."""

    scene: FrankaNutThreadSceneCfg = MISSING

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    recorders: RecordTerm = RecordTerm()

    dynamic_reset_gripper_effort_limit: bool = False  # Franka has no physical motor limits

    ee_body_name: str = "panda_hand"  # used by SO101Keyboard/Gamepad for EE-frame delta conversion

    robot_name: str = "franka_panda"
    default_feature_joint_names: list[str] = [f"{j}.pos" for j in FRANKA_ALL_JOINT_NAMES]
    task_description: str = "Thread the M16 nut onto the bolt"

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene = FrankaNutThreadSceneCfg(num_envs=1, env_spacing=2.5)

        self.decimation = 1
        self.episode_length_s = 60.0
        self.viewer.eye = (1.5, -0.8, 1.2)
        self.viewer.lookat = (0.5, 0.0, 0.0)

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.render.enable_translucency = True

        self.scene.ee_frame.visualizer_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)

    def use_teleop_device(self, teleop_device: str) -> None:
        """Configure actions for the given teleop device.

        Supports ``keyboard`` and ``gamepad``.  Both use IK-relative EE control
        (6D) plus relative finger position (2D) → 8D action, matching the
        ``preprocess_device_action`` keyboard/gamepad branch.
        """
        self.task_type = teleop_device

        if teleop_device in ("keyboard", "gamepad"):
            self.actions.arm_action = mdp.DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=["panda_joint.*"],
                body_name="panda_hand",
                controller=mdp.DifferentialIKControllerCfg(
                    command_type="pose", ik_method="dls", use_relative_mode=True
                ),
                body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
            )
            self.actions.gripper_action = mdp.RelativeJointPositionActionCfg(
                asset_name="robot",
                joint_names=["panda_finger_joint.*"],
                scale=1.0,
            )
            # Gravity disabled so the arm holds pose when the user isn't moving it
            self.scene.robot.spawn.rigid_props.disable_gravity = True
        else:
            raise ValueError(
                f"Teleop device '{teleop_device}' is not supported for Franka NutThread. "
                "Use 'keyboard' or 'gamepad'."
            )

    def preprocess_device_action(self, action: dict[str, Any], teleop_device) -> torch.Tensor:
        """Pass keyboard/gamepad actions through unchanged (already correct shape)."""
        from leisaac.devices.action_process import preprocess_device_action
        return preprocess_device_action(action, teleop_device)

    def build_lerobot_frame(self, episode_data: EpisodeData, dataset_cfg: LeRobotDatasetCfg) -> dict:
        """Build a LeRobot-compatible frame.

        Franka is sim-only, so joint positions are already in radians — no
        motor-unit conversion needed.
        """
        obs_data = episode_data._data["obs"]
        action = episode_data._data["actions"][-1].cpu().numpy()

        frame = {
            "action": action,
            "observation.state": obs_data["joint_pos"][-1].cpu().numpy(),
            "task": self.task_description,
        }
        for frame_key in dataset_cfg.features.keys():
            if not frame_key.startswith("observation.images"):
                continue
            camera_key = frame_key.split(".")[-1]
            frame[frame_key] = obs_data[camera_key][-1].cpu().numpy()

        return frame
