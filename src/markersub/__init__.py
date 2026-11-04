"""markersub: segment-wise lower-limb kinematics and kinetics for marker-based and markerless
motion capture, and marker substitution analysis.

Quick start::

    from markersub import Settings, analyse_dataset
    table = analyse_dataset("example_data", "results", Settings())
"""
from .pipeline import analyse_dataset, analyse_subject, read_subject
from .settings import Settings
from .substitution import Variant, standard_variants

__version__ = "0.1.0"
__all__ = ["Settings", "analyse_dataset", "analyse_subject", "read_subject", "Variant",
           "standard_variants", "__version__"]
