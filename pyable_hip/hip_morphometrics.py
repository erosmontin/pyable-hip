"""
Hip-specific morphometric measurements.

Cartilage thickness, coverage mapping, per-tissue volume/surface area,
and alpha angle computation from landmark points.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk

from pyable import Roiable, Imaginable
from pyable.metrics import (
    _ensure_same_geometry,
    roi_principal_extents,
    roi_max_feret,
    roi_max_thickness_inscribed,
    roi_mean_thickness,
    compute_compactness_score,
    compute_connectivity_score,
)

from .constants import LabelRegistry


# ============================================================================
# PER-TISSUE VOLUME / SURFACE AREA
# ============================================================================


def tissue_volumes(
    labelmap: sitk.Image,
    registry: Optional[LabelRegistry] = None,
    labels: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute volume (mm³) for each tissue.

    Parameters
    ----------
    labelmap : sitk.Image
        Multi-label image.
    registry : LabelRegistry, optional
    labels : list[str], optional
        Tissues to measure.  If None, all registered.

    Returns
    -------
    dict[str, float]
        ``{tissue_name: volume_mm3}``
    """
    if registry is None:
        registry = LabelRegistry.template_preset()
    if labels is None:
        labels = registry.names()

    spacing = labelmap.GetSpacing()
    voxel_vol = float(np.prod(spacing))
    arr = sitk.GetArrayFromImage(labelmap)

    volumes = {}
    for name in labels:
        try:
            lv = registry.label(name)
        except KeyError:
            continue
        n = int(np.sum(arr == lv))
        volumes[name] = n * voxel_vol

    return volumes


# ============================================================================
# CARTILAGE THICKNESS
# ============================================================================


def cartilage_thickness(
    cartilage_mask: sitk.Image,
) -> Dict[str, float]:
    """Mean / median / max thickness of a cartilage ROI.

    Uses the EDT-based local thickness measure from pyable.metrics.

    Parameters
    ----------
    cartilage_mask : sitk.Image
        Binary mask for the cartilage.

    Returns
    -------
    dict with ``mean``, ``median``, ``std``, ``max``, ``p25``, ``p75`` (mm).
    """
    arr = sitk.GetArrayFromImage(cartilage_mask) > 0
    spacing = cartilage_mask.GetSpacing()[::-1]  # ZYX
    return roi_mean_thickness(arr, spacing=spacing)


def cartilage_thickness_map(
    cartilage_mask: sitk.Image,
) -> sitk.Image:
    """Voxel-wise thickness map (2 * EDT) inside the cartilage.

    Parameters
    ----------
    cartilage_mask : sitk.Image
        Binary mask.

    Returns
    -------
    sitk.Image
        Float32 thickness image (mm), 0 outside the cartilage.
    """
    from scipy.ndimage import distance_transform_edt

    arr = sitk.GetArrayFromImage(cartilage_mask) > 0
    spacing = cartilage_mask.GetSpacing()[::-1]  # ZYX
    edt = distance_transform_edt(arr, sampling=spacing)
    thickness = (2.0 * edt * arr).astype(np.float32)

    out = sitk.GetImageFromArray(thickness)
    out.CopyInformation(cartilage_mask)
    return out


# ============================================================================
# CARTILAGE COVERAGE
# ============================================================================


def cartilage_coverage(
    cartilage_mask: sitk.Image,
    bone_mask: sitk.Image,
    max_distance_mm: float = 3.0,
) -> Dict[str, float]:
    """Fraction of bone surface covered by adjacent cartilage.

    Parameters
    ----------
    cartilage_mask : sitk.Image
        Binary cartilage mask.
    bone_mask : sitk.Image
        Binary bone mask (same grid).
    max_distance_mm : float
        Cartilage voxels within this distance of the bone surface
        are considered as "covering" the bone.

    Returns
    -------
    dict with ``coverage_fraction``, ``covered_surface_voxels``,
    ``total_surface_voxels``.
    """
    from scipy.ndimage import distance_transform_edt

    bone_mask = _ensure_same_geometry(cartilage_mask, bone_mask)

    bone_arr = sitk.GetArrayFromImage(bone_mask) > 0
    cart_arr = sitk.GetArrayFromImage(cartilage_mask) > 0
    spacing = cartilage_mask.GetSpacing()[::-1]  # ZYX

    # Bone surface: erode then XOR
    from scipy.ndimage import binary_erosion
    struct = np.ones((3, 3, 3), dtype=bool)
    bone_surface = bone_arr & ~binary_erosion(bone_arr, structure=struct)

    total_surface = int(bone_surface.sum())
    if total_surface == 0:
        return {
            "coverage_fraction": 0.0,
            "covered_surface_voxels": 0,
            "total_surface_voxels": 0,
        }

    # Distance from each bone surface voxel to nearest cartilage
    if cart_arr.sum() == 0:
        return {
            "coverage_fraction": 0.0,
            "covered_surface_voxels": 0,
            "total_surface_voxels": total_surface,
        }

    cart_edt = distance_transform_edt(~cart_arr, sampling=spacing)
    covered = bone_surface & (cart_edt <= max_distance_mm)
    n_covered = int(covered.sum())

    return {
        "coverage_fraction": n_covered / total_surface,
        "covered_surface_voxels": n_covered,
        "total_surface_voxels": total_surface,
    }


# ============================================================================
# PER-TISSUE MORPHOMETRICS
# ============================================================================


def tissue_morphometrics(
    binary_mask: sitk.Image,
) -> dict:
    """Compute shape metrics for a single tissue.

    Parameters
    ----------
    binary_mask : sitk.Image
        Binary mask.

    Returns
    -------
    dict with ``volume_mm3``, ``principal_extents``, ``max_feret_mm``,
    ``max_thickness_mm``, ``compactness``, ``n_components``,
    ``thickness_stats``.
    """
    arr = sitk.GetArrayFromImage(binary_mask) > 0
    spacing = binary_mask.GetSpacing()
    spacing_zyx = spacing[::-1]
    voxel_vol = float(np.prod(spacing))

    volume = float(arr.sum()) * voxel_vol

    extents = roi_principal_extents(arr, spacing=spacing_zyx)
    feret = roi_max_feret(arr, spacing=spacing_zyx)
    max_thick = roi_max_thickness_inscribed(arr, spacing=spacing_zyx)
    compact = compute_compactness_score(binary_mask)
    n_comp = compute_connectivity_score(binary_mask)
    thickness = roi_mean_thickness(arr, spacing=spacing_zyx)

    return {
        "volume_mm3": volume,
        "principal_extents": {
            "length": extents["length"],
            "width": extents["width"],
            "thickness": extents["thickness"],
        },
        "max_feret_mm": feret,
        "max_thickness_mm": max_thick,
        "compactness": compact,
        "n_components": n_comp,
        "thickness_stats": thickness,
    }


def all_tissue_morphometrics(
    labelmap: sitk.Image,
    registry: Optional[LabelRegistry] = None,
    labels: Optional[List[str]] = None,
) -> Dict[str, dict]:
    """Compute morphometrics for every tissue in a hip label map.

    Parameters
    ----------
    labelmap : sitk.Image
    registry : LabelRegistry, optional
    labels : list[str], optional

    Returns
    -------
    dict[str, dict]
    """
    if registry is None:
        registry = LabelRegistry.template_preset()
    if labels is None:
        labels = registry.names()

    results = {}
    for name in labels:
        try:
            lv = registry.label(name)
        except KeyError:
            continue
        mask = sitk.Cast(sitk.Equal(labelmap, lv), sitk.sitkUInt8)
        if sitk.GetArrayViewFromImage(mask).sum() == 0:
            continue
        results[name] = tissue_morphometrics(mask)

    return results


# ============================================================================
# ALPHA ANGLE
# ============================================================================


def alpha_angle(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
) -> float:
    """Compute the alpha angle from three landmark points.

    The alpha angle is the angle at *p2* in the triangle (p1, p2, p3).

    Parameters
    ----------
    p1, p2, p3 : array-like, shape (3,)
        Physical coordinates of the three landmarks.

    Returns
    -------
    float
        Angle in degrees.
    """
    p1, p2, p3 = np.asarray(p1), np.asarray(p2), np.asarray(p3)
    v1 = p1 - p2
    v2 = p3 - p2

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def alpha_angle_from_indices(
    idx1: np.ndarray,
    idx2: np.ndarray,
    idx3: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> float:
    """Alpha angle from voxel indices (converted to physical coordinates).

    Parameters
    ----------
    idx1, idx2, idx3 : array-like, shape (3,)
        Voxel indices (x, y, z).
    spacing : tuple
        Voxel spacing.
    origin : tuple
        Image origin.

    Returns
    -------
    float
        Angle in degrees.
    """
    spacing = np.asarray(spacing)
    origin = np.asarray(origin)
    p1 = np.asarray(idx1) * spacing + origin
    p2 = np.asarray(idx2) * spacing + origin
    p3 = np.asarray(idx3) * spacing + origin
    return alpha_angle(p1, p2, p3)


# ============================================================================
# HIP-SPECIFIC AGGREGATE ANALYSIS
# ============================================================================


def hip_morphometric_report(
    labelmap: sitk.Image,
    registry: Optional[LabelRegistry] = None,
) -> dict:
    """Full morphometric report for a hip label map.

    Computes per-tissue morphometrics, cartilage thickness, and
    cartilage-to-bone coverage for all bone-cartilage pairs.

    Parameters
    ----------
    labelmap : sitk.Image
    registry : LabelRegistry, optional

    Returns
    -------
    dict with ``'volumes'``, ``'morphometrics'``, ``'cartilage_thickness'``,
    ``'cartilage_coverage'``.
    """
    if registry is None:
        registry = LabelRegistry.template_preset()

    # Volumes
    volumes = tissue_volumes(labelmap, registry)

    # Morphometrics
    morpho = all_tissue_morphometrics(labelmap, registry)

    # Cartilage thickness
    cart_thick = {}
    for name in registry.cartilages():
        try:
            lv = registry.label(name)
            mask = sitk.Cast(sitk.Equal(labelmap, lv), sitk.sitkUInt8)
            if sitk.GetArrayViewFromImage(mask).sum() == 0:
                continue
            cart_thick[name] = cartilage_thickness(mask)
        except Exception:
            pass

    # Cartilage coverage (requires parent bone)
    coverage = {}
    for tissue in registry:
        if tissue.category == "cartilage" and tissue.parent:
            try:
                cart_lv = registry.label(tissue.name)
                bone_lv = registry.label(tissue.parent)
                cart_mask = sitk.Cast(sitk.Equal(labelmap, cart_lv), sitk.sitkUInt8)
                bone_mask = sitk.Cast(sitk.Equal(labelmap, bone_lv), sitk.sitkUInt8)
                if (sitk.GetArrayViewFromImage(cart_mask).sum() == 0
                        or sitk.GetArrayViewFromImage(bone_mask).sum() == 0):
                    continue
                coverage[tissue.name] = cartilage_coverage(cart_mask, bone_mask)
            except Exception:
                pass

    return {
        "volumes": volumes,
        "morphometrics": morpho,
        "cartilage_thickness": cart_thick,
        "cartilage_coverage": coverage,
    }
