import os
import yaml
from typing import Dict, Any


def load_yaml_config(filename: str) -> Dict[str, Any]:
    """loads yaml-files from config-folder."""

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.dirname(current_script_dir)
    config_path = os.path.join(package_dir, "config", filename)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config-file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise Exception(f"Error parsing {filename}: {e}")


def get_agent_config(agent_name: str) -> Dict[str, Any]:
    return load_yaml_config("agents.yaml")[agent_name]


def get_task_config(task_name: str) -> Dict[str, Any]:
    return load_yaml_config("tasks.yaml")[task_name]


def get_model_config() -> Dict[str, Any]:
    return load_yaml_config("configs.yaml").get("models", {})
