"""Smoke tests asserting that every ``peruse`` module is importable.

Modules that are not reached by ``peruse/__init__.py`` are easy to break
silently when an upstream dependency renames part of its API. Importing them
explicitly here turns such a break into a test failure instead of a surprise at
run time.
"""

import importlib

import pytest

MODULE_NAMES = [
    'peruse',
    'peruse.doc_simil',
    'peruse.single_wf_snip_analysis',
    'peruse.slang_lite_py2http_ws',
    'peruse.slang_lite_ws',
    'peruse.util',
]


@pytest.mark.parametrize('module_name', MODULE_NAMES)
def test_module_imports(module_name):
    """Every listed module imports without raising."""
    assert importlib.import_module(module_name) is not None


def test_slang_lite_py2http_ws_builds_app():
    """The py2http module builds its module-level HTTP ``app`` on import."""
    module = importlib.import_module('peruse.slang_lite_py2http_ws')
    assert getattr(module, 'app', None) is not None
