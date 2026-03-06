"""
pyable-hip: Hip-specific medical image analysis built on pyable.

Provides specialised classes and workflows for hip joint analysis including
bilateral label-map handling, per-structure registration, pre/post comparison,
morphometric measurements (cartilage thickness, alpha angle, coverage), and
atlas-based segmentation refinement.
"""

__version__ = "0.1.0"

# ── Core data structures ────────────────────────────────────────────────
from pyable_hip.constants import (
    Tissue,
    LabelRegistry,
    RoiConfig,
    DEFAULT_ROI_CONFIGS,
)

# ── Labelmap handling ────────────────────────────────────────────────────
from pyable_hip.hip_labelmap import HipLabelMap

# ── Case container ───────────────────────────────────────────────────────
from pyable_hip.hip_case import HipCase

# ── Registration ─────────────────────────────────────────────────────────
from pyable_hip.hip_registration import (
    register_whole_hip,
    register_per_structure,
    combine_registered_masks,
    register_hip_cases,
    resample_labelmap,
)

# ── Comparison / change-maps ─────────────────────────────────────────────
from pyable_hip.hip_comparison import (
    compare_structure,
    compare_hip_labelmaps,
    compute_per_structure_distance_maps,
    compute_per_structure_change_maps,
    export_comparison_report,
)

# ── Morphometrics ────────────────────────────────────────────────────────
from pyable_hip.hip_morphometrics import (
    tissue_volumes,
    cartilage_thickness,
    cartilage_thickness_map,
    cartilage_coverage,
    tissue_morphometrics,
    all_tissue_morphometrics,
    alpha_angle,
    alpha_angle_from_indices,
    hip_morphometric_report,
)

# ── Segmentation refinement ─────────────────────────────────────────────
from pyable_hip.hip_refinement import (
    remove_small_components,
    clip_roi_to_reference_z,
    geodesic_refinement,
    intensity_based_refinement,
    refine_cartilage,
    smooth_roi,
    refine_hip_segmentation,
)

__all__ = [
    # constants
    "Tissue",
    "LabelRegistry",
    "RoiConfig",
    "DEFAULT_ROI_CONFIGS",
    # labelmap
    "HipLabelMap",
    # case
    "HipCase",
    # registration
    "register_whole_hip",
    "register_per_structure",
    "combine_registered_masks",
    "register_hip_cases",
    "resample_labelmap",
    # comparison
    "compare_structure",
    "compare_hip_labelmaps",
    "compute_per_structure_distance_maps",
    "compute_per_structure_change_maps",
    "export_comparison_report",
    # morphometrics
    "tissue_volumes",
    "cartilage_thickness",
    "cartilage_thickness_map",
    "cartilage_coverage",
    "tissue_morphometrics",
    "all_tissue_morphometrics",
    "alpha_angle",
    "alpha_angle_from_indices",
    "hip_morphometric_report",
    # refinement
    "remove_small_components",
    "clip_roi_to_reference_z",
    "geodesic_refinement",
    "intensity_based_refinement",
    "refine_cartilage",
    "smooth_roi",
    "refine_hip_segmentation",
]
