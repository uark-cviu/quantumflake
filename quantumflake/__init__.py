"""
QuantumFlake package metadata and lazy exports.
"""

__version__ = "0.2.4"
__all__ = ["FlakePipeline", "draw_overlay"]


def __getattr__(name):
    if name == "FlakePipeline":
        from .pipeline import FlakePipeline

        return FlakePipeline
    if name == "draw_overlay":
        from .utils.vis import draw_overlay

        return draw_overlay
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
