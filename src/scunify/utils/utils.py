from typing import Any

import scanpy as sc
import yaml


def save_yaml(data: Any, file_path: str) -> None:
    """
    Save data to a YAML file.

    :param data: The data to be saved.
    :param file_path: The path to the file where data should be saved.
    """
    with open(file_path, "w") as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False)


def load_yaml(file_path: str) -> Any:
    """
    Load data from a YAML file.

    :param file_path: The path to the YAML file to be loaded.
    :return: The data loaded from the YAML file.
    """
    with open(file_path) as file:
        data = yaml.safe_load(file)
    return data


def read_h5ad(file_path: str):
    return sc.read_h5ad(file_path)
