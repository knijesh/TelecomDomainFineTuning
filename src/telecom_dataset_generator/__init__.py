# src/telecom_dataset_generator/__init__.py
"""Telecom Dataset Generator"""

__version__ = "1.0.0"

from .generator import generate_dataset, save_datasets

__all__ = ["generate_dataset", "save_datasets"]