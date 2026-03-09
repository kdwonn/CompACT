"""Utility functions for Hydra configuration management."""

import os
import yaml
from omegaconf import DictConfig, OmegaConf
from typing import Optional


# Register a custom resolver that can evaluate lambda expressions
OmegaConf.register_new_resolver("divide", lambda x, y: x // y)


def load_experiment_config(exp_dir: str) -> Optional[DictConfig]:
    """
    Load the original experiment configuration from the .hydra directory.

    Args:
        exp_dir: Path to experiment directory

    Returns:
        DictConfig if found, None otherwise
    """
    hydra_config_path = os.path.join(exp_dir, ".hydra", "config.yaml")

    if os.path.exists(hydra_config_path):
        try:
            return OmegaConf.load(hydra_config_path)
        except Exception as e:
            print(f"Error loading config from {hydra_config_path}: {e}")
            return None
    else:
        # Try to find any YAML config in the directory as fallback
        yaml_files = [f for f in os.listdir(exp_dir) if f.endswith(".yaml")]
        if yaml_files:
            try:
                yaml_path = os.path.join(exp_dir, yaml_files[0])
                print(f"No .hydra/config.yaml found, trying {yaml_path}")
                return OmegaConf.load(yaml_path)
            except Exception as e:
                print(f"Error loading fallback config: {e}")
                return None
        else:
            print(f"Warning: No config file found at {hydra_config_path}")
            return None


def merge_configs(
    base_config: DictConfig,
    override_config: DictConfig,
    keys_to_merge: Optional[list] = None,
) -> DictConfig:
    """
    Merge selected keys from override_config into base_config.

    Args:
        base_config: Base configuration to update
        override_config: Configuration with values to override
        keys_to_merge: List of top-level keys to merge. If None, merge all keys.

    Returns:
        Updated config
    """
    result = OmegaConf.create(OmegaConf.to_container(base_config))

    if keys_to_merge is None:
        # Merge all keys
        return OmegaConf.merge(result, override_config)

    # Merge only specified keys
    for key in keys_to_merge:
        if key in override_config:
            if key in result:
                # If the key exists in both, merge the nested configs
                result[key] = OmegaConf.merge(result[key], override_config[key])
            else:
                # If the key only exists in override_config, just copy it
                result[key] = override_config[key]

    return result


def print_config_summary(config: DictConfig, prefix: str = "") -> None:
    """
    Print a summary of the configuration for debugging.

    Args:
        config: Configuration to print
        prefix: Prefix for nested keys
    """
    for key, value in config.items():
        if isinstance(value, DictConfig):
            print(f"{prefix}{key}:")
            print_config_summary(value, prefix + "  ")
        else:
            print(f"{prefix}{key}: {value}")


def load_tokenizer_config(tokenizer_path: str) -> Optional[DictConfig]:
    """
    Load tokenizer configuration from the tokenizer training path.

    Args:
        tokenizer_path: Path to tokenizer training directory

    Returns:
        DictConfig with tokenizer configuration if found, None otherwise
    """
    if not tokenizer_path:
        return None

    # Try loading from .hydra/config.yaml first
    hydra_config_path = os.path.join(tokenizer_path, ".hydra", "config.yaml")

    if os.path.exists(hydra_config_path):
        try:
            tokenizer_config = OmegaConf.load(hydra_config_path)
            OmegaConf.resolve(tokenizer_config)
            print(f"Loaded tokenizer config from {hydra_config_path}")
            return tokenizer_config
        except Exception as e:
            print(f"Error loading tokenizer config from {hydra_config_path}: {e}")

    # Try loading from config.yaml directly
    config_path = os.path.join(tokenizer_path, "config.yaml")
    if os.path.exists(config_path):
        try:
            tokenizer_config = OmegaConf.load(config_path)
            OmegaConf.resolve(tokenizer_config)
            print(f"Loaded tokenizer config from {config_path}")
            return tokenizer_config
        except Exception as e:
            print(f"Error loading tokenizer config from {config_path}: {e}")

    print(f"Warning: No tokenizer config found in {tokenizer_path}")
    return None


def save_config(config: DictConfig, path: str) -> None:
    """
    Save a configuration to a YAML file.

    Args:
        config: Configuration to save
        path: Path to save the configuration
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(
            OmegaConf.to_container(config, resolve=True), f, default_flow_style=False
        )
