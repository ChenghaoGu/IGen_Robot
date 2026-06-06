"""Project root paths for IGen_Code (portable, no hard-coded machine paths)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CONTROL_YOUR_ROBOT_ROOT = PROJECT_ROOT / "control_your_robot"
ROLO_ENV_ROOT = CONTROL_YOUR_ROBOT_ROOT / "rolo_env"

FRANKA_CUROBO_ENV_YAML = ROLO_ENV_ROOT / "task_config" / "franka_follow_curobo_env.yaml"
FRANKA_CUROBO_CONFIG_YAML = ROLO_ENV_ROOT / "config" / "franka.yml"
DEBUG_MESH_OBJ = CONTROL_YOUR_ROBOT_ROOT / "debug_mesh.obj"
