"""
Hip-specific comparison utilities for pre/post treatment analysis.

Combines per-structure overlap metrics, surface distance maps, and
boolean change maps into a single structured report.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk

from pyable import Roiable, Imaginable
from pyable.metrics import (
    _ensure_same_geometry,
    compare_segmentations,
    compute_change_maps,
    surface_distance_map_image,
    export_surface_with_error,
    export_surface_vtp,
    surface_mask,
)

from .constants import LabelRegistry
from .hip_labelmap import HipLabelMap


# ============================================================================
# PER-STRUCTURE COMPARISON
# ============================================================================


def compare_structure(
    pre_mask: sitk.Image,
    post_mask: sitk.Image,
    name: str = "",
) -> dict:
    """Compare two binary masks for one structure.

    Returns overlap, surface distance, and change-map statistics.

    Parameters
    ----------
    pre_mask, post_mask : sitk.Image
        Binary masks (same grid expected; post should be registered).
    name : str
        Tissue name for labelling results.

    Returns
    -------
    dict  with keys:
        ``overlap`` (Dice, Jaccard, …), ``surface`` (MSD, HD95, …),
        ``change`` (n_removed, n_added, n_changed), ``name``.
    """
    post_mask = _ensure_same_geometry(pre_mask, post_mask)

    # Overlap + surface metrics from pyable.metrics
    metrics = compare_segmentations(pre_mask, post_mask)

    # Change maps
    maps = compute_change_maps(pre_mask, post_mask)
    change_stats = {
        "n_removed": int(np.sum(sitk.GetArrayFromImage(maps["removed"]) > 0)),
        "n_added": int(np.sum(sitk.GetArrayFromImage(maps["added"]) > 0)),
        "n_changed": int(np.sum(sitk.GetArrayFromImage(maps["changed"]) > 0)),
    }

    # Voxel volume for mm³ conversion
    spacing = pre_mask.GetSpacing()
    voxel_vol = float(np.prod(spacing))
    change_stats["removed_mm3"] = change_stats["n_removed"] * voxel_vol
    change_stats["added_mm3"] = change_stats["n_added"] * voxel_vol

    return {"name": name, "overlap": metrics, "change": change_stats}


# ============================================================================
# FULL HIP COMPARISON
# ============================================================================


def compare_hip_labelmaps(
    pre: sitk.Image,
    post: sitk.Image,
    registry: Optional[LabelRegistry] = None,
    labels: Optional[List[str]] = None,
) -> Dict[str, dict]:
    """Compare two hip label maps on a per-tissue basis.

    Parameters
    ----------
    pre, post : sitk.Image
        Multi-label images (same grid; post already registered).
    registry : LabelRegistry, optional
        Tissue mapping.  Defaults to template preset.
    labels : list[str], optional
        Tissue names to compare.  If None, all registered tissues.

    Returns
    -------
    dict[str, dict]
        Per-tissue comparison results.
    """
    if registry is None:
        registry = LabelRegistry.template_preset()

    post = _ensure_same_geometry(pre, post)

    if labels is None:
        labels = registry.names()

    results = {}
    for name in labels:
        try:
            label_val = registry.label(name)
        except KeyError:
            continue
        pre_bin = sitk.Cast(sitk.Equal(pre, label_val), sitk.sitkUInt8)
        post_bin = sitk.Cast(sitk.Equal(post, label_val), sitk.sitkUInt8)

        # Skip if both empty
        if (sitk.GetArrayViewFromImage(pre_bin).sum() == 0
                and sitk.GetArrayViewFromImage(post_bin).sum() == 0):
            continue

        results[name] = compare_structure(pre_bin, post_bin, name=name)

    return results


# ============================================================================
# DISTANCE MAP GENERATION (per-structure)
# ============================================================================


def compute_per_structure_distance_maps(
    pre: sitk.Image,
    post_registered: sitk.Image,
    registry: Optional[LabelRegistry] = None,
    labels: Optional[List[str]] = None,
) -> Dict[str, dict]:
    """Compute surface distance map images for each structure.

    Parameters
    ----------
    pre, post_registered : sitk.Image
        Multi-label images (same grid).
    registry : LabelRegistry, optional
    labels : list[str], optional
        Tissue names.  If None, all registered.

    Returns
    -------
    dict[str, dict]
        ``{name: {'pre_to_post': sitk.Image, 'post_to_pre': sitk.Image}}``
    """
    if registry is None:
        registry = LabelRegistry.template_preset()

    post_registered = _ensure_same_geometry(pre, post_registered)

    if labels is None:
        labels = registry.names()

    results = {}
    for name in labels:
        try:
            label_val = registry.label(name)
        except KeyError:
            continue

        pre_bin = sitk.Cast(sitk.Equal(pre, label_val), sitk.sitkUInt8)
        post_bin = sitk.Cast(sitk.Equal(post_registered, label_val), sitk.sitkUInt8)

        if (sitk.GetArrayViewFromImage(pre_bin).sum() == 0
                or sitk.GetArrayViewFromImage(post_bin).sum() == 0):
            continue

        pre_to_post = surface_distance_map_image(pre_bin, post_bin)
        post_to_pre = surface_distance_map_image(post_bin, pre_bin)

        results[name] = {
            "pre_to_post": pre_to_post,
            "post_to_pre": post_to_pre,
        }

    return results


# ============================================================================
# CHANGE MAP GENERATION (per-structure)
# ============================================================================


def compute_per_structure_change_maps(
    pre: sitk.Image,
    post_registered: sitk.Image,
    registry: Optional[LabelRegistry] = None,
    labels: Optional[List[str]] = None,
) -> Dict[str, Dict[str, sitk.Image]]:
    """Boolean change maps per tissue.

    Parameters
    ----------
    pre, post_registered : sitk.Image
    registry : LabelRegistry, optional
    labels : list[str], optional

    Returns
    -------
    dict[str, dict[str, sitk.Image]]
        ``{name: {'removed': Image, 'added': Image, 'changed': Image}}``
    """
    if registry is None:
        registry = LabelRegistry.template_preset()

    post_registered = _ensure_same_geometry(pre, post_registered)

    if labels is None:
        labels = registry.names()

    results = {}
    for name in labels:
        try:
            label_val = registry.label(name)
        except KeyError:
            continue

        pre_bin = sitk.Cast(sitk.Equal(pre, label_val), sitk.sitkUInt8)
        post_bin = sitk.Cast(sitk.Equal(post_registered, label_val), sitk.sitkUInt8)
        results[name] = compute_change_maps(pre_bin, post_bin)

    return results


# ============================================================================
# EXPORT HELPERS
# ============================================================================


def export_comparison_report(
    pre: sitk.Image,
    post_registered: sitk.Image,
    out_dir: str,
    registry: Optional[LabelRegistry] = None,
    labels: Optional[List[str]] = None,
    export_meshes: bool = True,
    export_vtp: bool = False,
) -> dict:
    """Run full comparison and write results + meshes to *out_dir*.

    Parameters
    ----------
    pre, post_registered : sitk.Image
        Multi-label images.
    out_dir : str
        Output directory.
    registry : LabelRegistry, optional
    labels : list[str], optional
    export_meshes : bool
        Write surface+error PLY meshes via pyvista.
    export_vtp : bool
        Write VTP meshes with distance scalars via VTK.

    Returns
    -------
    dict  with ``'metrics'``, ``'change_maps'``, ``'paths'``.
    """
    os.makedirs(out_dir, exist_ok=True)
    if registry is None:
        registry = LabelRegistry.template_preset()
    post_registered = _ensure_same_geometry(pre, post_registered)

    if labels is None:
        labels = registry.names()

    # Per-structure metrics
    metrics = compare_hip_labelmaps(pre, post_registered, registry, labels)

    # Change maps
    change_maps = compute_per_structure_change_maps(
        pre, post_registered, registry, labels,
    )

    # Write change maps as NIfTI
    paths = {}
    for name, maps in change_maps.items():
        for key, img in maps.items():
            p = os.path.join(out_dir, f"{name}_{key}.nii.gz")
            sitk.WriteImage(sitk.Cast(img, sitk.sitkUInt8), p)
            paths[f"{name}_{key}"] = p

    # Distance maps
    dist_maps = compute_per_structure_distance_maps(
        pre, post_registered, registry, labels,
    )
    for name, dm in dist_maps.items():
        for direction, img in dm.items():
            p = os.path.join(out_dir, f"{name}_{direction}.nii.gz")
            sitk.WriteImage(img, p)
            paths[f"{name}_{direction}"] = p

    # Surface meshes
    if export_meshes:
        for name in labels:
            try:
                label_val = registry.label(name)
            except KeyError:
                continue
            pre_bin = sitk.Cast(sitk.Equal(pre, label_val), sitk.sitkUInt8)
            post_bin = sitk.Cast(sitk.Equal(post_registered, label_val), sitk.sitkUInt8)
            if (sitk.GetArrayViewFromImage(pre_bin).sum() == 0
                    or sitk.GetArrayViewFromImage(post_bin).sum() == 0):
                continue
            mesh_dir = os.path.join(out_dir, f"meshes_{name}")
            try:
                export_surface_with_error(
                    pre_bin, post_bin,
                    spacing=pre.GetSpacing(),
                    out_dir=mesh_dir,
                )
            except ImportError:
                pass  # pyvista not available

    if export_vtp:
        for name in labels:
            try:
                label_val = registry.label(name)
            except KeyError:
                continue
            pre_bin = sitk.Cast(sitk.Equal(pre, label_val), sitk.sitkUInt8)
            post_bin = sitk.Cast(sitk.Equal(post_registered, label_val), sitk.sitkUInt8)
            if (sitk.GetArrayViewFromImage(pre_bin).sum() == 0
                    or sitk.GetArrayViewFromImage(post_bin).sum() == 0):
                continue
            # Distance to post on pre surface
            dist_to_post = sitk.Abs(sitk.SignedMaurerDistanceMap(
                post_bin, insideIsPositive=False,
                squaredDistance=False, useImageSpacing=True,
            ))
            vtp_path = os.path.join(out_dir, f"{name}_pre_surface.vtp")
            try:
                export_surface_vtp(pre_bin, vtp_path, distance_image=dist_to_post)
                paths[f"{name}_vtp"] = vtp_path
            except ImportError:
                pass  # vtk not available

    # Write metrics as JSON
    try:
        import json
        json_path = os.path.join(out_dir, "comparison_metrics.json")
        # Make JSON-serialisable
        out_metrics = {}
        for name, m in metrics.items():
            out_metrics[name] = {
                "overlap": {k: float(v) if v is not None else None
                            for k, v in m.get("overlap", {}).items()},
                "change": m.get("change", {}),
            }
        with open(json_path, "w") as f:
            json.dump(out_metrics, f, indent=2)
        paths["metrics_json"] = json_path
    except Exception:
        pass

    return {"metrics": metrics, "change_maps": change_maps, "paths": paths}
