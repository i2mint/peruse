"""Pytest configuration for the ``peruse`` test suite.

This file exists for its *location*, not its contents. Under pytest's default
``prepend`` import mode the repository root is added to ``sys.path`` only when a
``conftest.py`` sits there. Without it, ``tests/`` alone is on the path and
``import peruse`` resolves to whatever ``peruse`` is *installed* -- which, for an
editable install, is one fixed directory that need not be the checkout under
test. The suite would then report on the wrong source tree: green on a broken
checkout, red on a fixed one. Keeping this file here pins the tests to the tree
they live in.
"""
