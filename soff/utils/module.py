"""Utilities for the python's native module system"""

import inspect
import importlib
import re
import os
from types import ModuleType
from typing import Dict, List, Optional, Type


def load_all_submodules(
        parent_module: Optional[ModuleType] = None) -> List[ModuleType]:
    """
    This function is intended to be called from `__init__.py`.
    It automatically loads all .py files in the same directory
    and return them as a list of modules.

    If parent_module is provided, it loads all .py module under that submodule
    """
    parent_module = parent_module or inspect.getmodule(inspect.stack()[1][0])
    assert parent_module is not None and parent_module.__file__ is not None
    return [
        importlib.import_module(f"{parent_module.__name__}.{module[:-3]}")
        for module in os.listdir(os.path.dirname(parent_module.__file__))
        if module != '__init__.py' and module[-3:] == '.py'
    ]


def load_all_direct_classes(module: ModuleType) -> Dict[str, Type]:
    """Load all classes directly defined modules (but not imported)"""
    return {
        name: cls for name, cls in module.__dict__.items()
        if isinstance(cls, type) and module.__name__ in cls.__module__
    }


def camel_to_snake(name):
    """Convert camel case to snake case"""
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
