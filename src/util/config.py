"""Defines functions related to configuration files."""

from importlib import import_module

import os
import sys
import yaml


def load_model_module(path):
    """Loads a model configuration file.

    Args:
      path: The relative path to the configuration file.

    Returns:
      A Python module.
    """
    dirname, filename = os.path.split(path)
    module_name, _ = os.path.splitext(filename)
    sys.path.insert(0, os.path.abspath(dirname))
    module = import_module(module_name)

    if not hasattr(module, "model"):
        raise ImportError("No model defined in {}".format(path))

    return module


def load_config(config_paths, config=None):
    """Loads configuration files.

    Args:
      config_paths: A list of configuration files.
      config: A (possibly non empty) config dictionary to fill.

    Returns:
      The configuration dictionary.
    """
    if config is None:
        config = {}

    for config_path in config_paths:
        with open(config_path, mode="rb") as config_file:
            subconfig = yaml.load(config_file.read())

            # Add or update section in main configuration.
            for section in subconfig:
                if section in config:
                    if isinstance(config[section], dict):
                        config[section].update(subconfig[section])
                    else:
                        config[section] = subconfig[section]
                else:
                    config[section] = subconfig[section]

    return config


def load_configuration(path):
    return load_config([path])