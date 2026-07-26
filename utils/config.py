import yaml
import os
from typing import Any, Dict

class ConfigNamespace:
    """A namespace object to allow dot-notation access to config dictionary keys."""
    def __init__(self, d: Dict[str, Any]):
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigNamespace(value))
            elif isinstance(value, list):
                setattr(self, key, [
                    ConfigNamespace(item) if isinstance(item, dict) else item
                    for item in value
                ])
            else:
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert namespace back to dictionary."""
        d = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigNamespace):
                d[key] = value.to_dict()
            elif isinstance(value, list):
                d[key] = [
                    item.to_dict() if isinstance(item, ConfigNamespace) else item
                    for item in value
                ]
            else:
                d[key] = value
        return d

    def __repr__(self) -> str:
        return f"ConfigNamespace({self.__dict__})"


def load_config(filepath: str) -> ConfigNamespace:
    """Load configuration from a YAML file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        config_dict = yaml.safe_load(f)
        
    if config_dict is None:
        config_dict = {}
        
    return ConfigNamespace(config_dict)
