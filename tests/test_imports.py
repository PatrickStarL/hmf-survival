"""Minimal smoke test: verify the core training pipeline modules import cleanly.

This does not require GPU, model weights, or any TCGA data -- it only checks
that the model/training code itself is free of import-time errors (missing
files, broken cross-references, syntax errors introduced by refactors, etc).
Run with: python -m pytest tests/test_imports.py -v
"""

import importlib

import pytest

CORE_MODULES = [
    "src.mil_models.model_factory_hierarchical_fusion",
    "src.mil_models.model_multimodal_hierarchical_fusion",
    "src.mil_models.model_multimodal_encoders",
    "src.mil_models.model_multimodal_cls_fusion",
    "src.training.trainer_hierarchical_fusion",
    "src.training.main_survival_hierarchical_fusion",
    "src.wsi_datasets",
    "src.utils.losses",
    "src.utils.utils",
]


@pytest.mark.parametrize("module_name", CORE_MODULES)
def test_module_imports(module_name):
    importlib.import_module(module_name)


def test_model_factory_exposes_entry_point():
    factory = importlib.import_module("src.mil_models.model_factory_hierarchical_fusion")
    assert hasattr(factory, "create_multimodal_survival_model")


def test_trainer_exposes_train_function():
    trainer = importlib.import_module("src.training.trainer_hierarchical_fusion")
    assert hasattr(trainer, "train")
