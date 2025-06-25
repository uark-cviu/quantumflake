# quantumflake/__init__.py

"""
QuantumFlake: A framework for 2D material detection and classification.

This package provides a streamlined pipeline for identifying and classifying
graphene flakes and other 2D materials from microscopy images.
"""

__version__ = "0.1.0"

from .pipeline import FlakePipeline
from .utils.vis import draw_overlay