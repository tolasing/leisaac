from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sim.schemas.schemas_cfg import ArticulationRootPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from leisaac.assets.robots.lerobot import SO101_FOLLOWER_CFG

from ..template import (
    SingleArmObservationsCfg,
    SingleArmTaskEnvCfg,
    SingleArmTaskSceneCfg,
    SingleArmTerminationsCfg,
)
from . import mdp


# SO101 repositioned for the industrial table (surface z≈0, bolt at x=0.5 in front).
# Tune x/y/z if the arm can't reach the bolt.
_SO101_INDUSTRIAL_CFG = SO101_FOLLOWER_CFG.copy()
_SO101_INDUSTRIAL_CFG.init_state = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    rot=(0.0, 0.0, 0.0, 1.0),
    joint_pos={
        "shoulder_pan": 0.0,
        "shoulder_lift": 0.0,
        "elbow_flex": 0.0,
        "wrist_flex": 0.0,
        "wrist_roll": 0.0,
        "gripper": 0.0,
    },
)


@configclass
class SO101NutThreadSceneCfg(SingleArmTaskSceneCfg):
    """Industrial scene: SeattleLabTable (as main scene) + ground plane + SO101 + bolt + nut."""

    # SeattleLabTable is the "scene" backdrop — same role the kitchen USD plays in pick_orange
    scene: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.707, 0.0, 0.0, 0.707)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        ),
    )

    # Ground plane — global prim path so it's shared (not cloned per env)
    ground_plane: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=GroundPlaneCfg(),
    )

    # Override default SO101 position for the industrial table surface (z≈0)
    robot: ArticulationCfg = _SO101_INDUSTRIAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Bolt M16 — fixed to table (factory USD has articulation root → ArticulationCfg required)
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

    # Nut M16 — free rigid body, picked up and threaded onto the bolt
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


@configclass
class SO101NutThreadEnvCfg(SingleArmTaskEnvCfg):
    """SO101 arm on a SeattleLabTable with M16 bolt/nut threading task."""

    # All three must be class attributes so super().__post_init__() sees them populated
    scene: SO101NutThreadSceneCfg = SO101NutThreadSceneCfg(env_spacing=2.5)
    observations: SingleArmObservationsCfg = SingleArmObservationsCfg()
    terminations: SingleArmTerminationsCfg = SingleArmTerminationsCfg()

    task_description: str = "Thread the M16 nut onto the bolt"

    def __post_init__(self) -> None:
        super().__post_init__()

        self.episode_length_s = 60.0
        self.viewer.eye = (1.5, -0.8, 1.2)
        self.viewer.lookat = (0.5, 0.0, 0.0)
