"""
Hip-specific registration utilities.

Provides per-structure registration (separate femur L/R + acetabulum)
and whole-hip registration using signed-distance-map metrics.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk

from pyable import Imaginable, LabelMapable, Roiable
from pyable.metrics import (
    _ensure_same_geometry,
    signed_distance_map,
    rigid_register_masks,
    split_connected_components,
    split_left_right,
)

from .constants import LabelRegistry
from .hip_labelmap import HipLabelMap


# ============================================================================
# LOW-LEVEL REGISTRATION HELPERS
# ============================================================================


def _union_mask(
    label_img: sitk.Image,
    label_values: List[int],
) -> sitk.Image:
    """Build a union binary mask from multiple label values."""
    m = sitk.Equal(label_img, label_values[0])
    for v in label_values[1:]:
        m = m | sitk.Equal(label_img, v)
    return sitk.Cast(m, sitk.sitkUInt8)


def _register_mask_pair(
    fixed_mask: sitk.Image,
    moving_mask: sitk.Image,
    method: str = "rigid",
    iterations: int = 200,
) -> Tuple[sitk.Image, sitk.Transform]:
    """Register a moving binary mask onto a fixed one.

    Thin wrapper around ``pyable.metrics.rigid_register_masks``.
    """
    return rigid_register_masks(
        fixed_mask, moving_mask, method=method, iterations=iterations,
    )


def resample_labelmap(
    moving_labelmap: sitk.Image,
    reference: sitk.Image,
    transform: sitk.Transform,
) -> sitk.Image:
    """Resample a label map with nearest-neighbor interpolation."""
    return sitk.Resample(
        sitk.Cast(moving_labelmap, sitk.sitkUInt16),
        reference,
        transform,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt16,
    )


# ============================================================================
# WHOLE-HIP REGISTRATION
# ============================================================================


def register_whole_hip(
    fixed_labelmap: sitk.Image,
    moving_labelmap: sitk.Image,
    registry: Optional[LabelRegistry] = None,
    method: str = "rigid",
    iterations: int = 200,
) -> Tuple[sitk.Image, sitk.Transform]:
    """Register the whole moving label map onto the fixed one.

    A union mask of all registered bone labels is used for registration.

    Parameters
    ----------
    fixed_labelmap, moving_labelmap : sitk.Image
        Multi-label images.
    registry : LabelRegistry, optional
        Determines which labels to include in the union mask.
    method : str
        ``'rigid'`` or ``'affine'``.
    iterations : int
        Max optimiser iterations.

    Returns
    -------
    (registered_labelmap, transform) : tuple
    """
    if registry is None:
        registry = LabelRegistry.template_preset()

    bone_labels = [registry.label(n) for n in registry.bones()]
    if not bone_labels:
        # fallback: use all non-zero
        arr = sitk.GetArrayFromImage(fixed_labelmap)
        bone_labels = sorted(int(v) for v in np.unique(arr) if v != 0)

    fixed_mask = _union_mask(fixed_labelmap, bone_labels)
    moving_mask = _union_mask(moving_labelmap, bone_labels)

    _, transform = _register_mask_pair(
        fixed_mask, moving_mask, method=method, iterations=iterations,
    )

    resampled = resample_labelmap(moving_labelmap, fixed_labelmap, transform)
    return resampled, transform


# ============================================================================
# PER-STRUCTURE REGISTRATION
# ============================================================================


def register_per_structure(
    fixed_labelmap: sitk.Image,
    moving_labelmap: sitk.Image,
    structures: Optional[Dict[str, int]] = None,
    registry: Optional[LabelRegistry] = None,
    method: str = "rigid",
    iterations: int = 200,
) -> Dict[str, dict]:
    """Register each structure independently and combine the results.

    For bilateral data (e.g. left + right femur), each connected component
    is registered separately.

    Parameters
    ----------
    fixed_labelmap, moving_labelmap : sitk.Image
        Multi-label images.
    structures : dict[str, int], optional
        ``{name: label_value}`` of structures to register.
        If *None*, bones from *registry* are used.
    registry : LabelRegistry, optional
        Tissue label mapping.
    method : str
        ``'rigid'`` or ``'affine'``.
    iterations : int
        Max optimiser iterations.

    Returns
    -------
    dict[str, dict]
        Per-structure results::

            {
                'femur_L': {
                    'transform': sitk.Transform,
                    'mask': sitk.Image,          # registered binary mask
                },
                ...
            }

    Notes
    -----
    If a structure has two connected components (bilateral), they are
    split via ``split_left_right`` and registered independently.
    """
    if registry is None:
        registry = LabelRegistry.template_preset()

    if structures is None:
        structures = {n: registry.label(n) for n in registry.bones()}

    results: Dict[str, dict] = {}

    for name, label_val in structures.items():
        fixed_bin = sitk.Cast(sitk.Equal(fixed_labelmap, label_val), sitk.sitkUInt8)
        moving_bin = sitk.Cast(sitk.Equal(moving_labelmap, label_val), sitk.sitkUInt8)

        # Check how many components the fixed mask has
        fixed_components = split_connected_components(fixed_bin, sort_by="size")
        moving_components = split_connected_components(moving_bin, sort_by="size")

        if len(fixed_components) >= 2 and len(moving_components) >= 2:
            # Bilateral — split and register each side
            try:
                fixed_L, fixed_R = split_left_right(fixed_bin, axis=0)
                moving_L, moving_R = split_left_right(moving_bin, axis=0)
            except RuntimeError:
                # Fallback: register as one piece
                reg_mask, tfm = _register_mask_pair(
                    fixed_bin, moving_bin, method, iterations,
                )
                results[name] = {"transform": tfm, "mask": reg_mask}
                continue

            reg_L, tfm_L = _register_mask_pair(
                fixed_L, moving_L, method, iterations,
            )
            reg_R, tfm_R = _register_mask_pair(
                fixed_R, moving_R, method, iterations,
            )
            results[f"{name}_L"] = {"transform": tfm_L, "mask": reg_L}
            results[f"{name}_R"] = {"transform": tfm_R, "mask": reg_R}
        else:
            # Single component — register directly
            reg_mask, tfm = _register_mask_pair(
                fixed_bin, moving_bin, method, iterations,
            )
            results[name] = {"transform": tfm, "mask": reg_mask}

    return results


def combine_registered_masks(
    reference: sitk.Image,
    structure_results: Dict[str, dict],
    registry: Optional[LabelRegistry] = None,
) -> sitk.Image:
    """Combine per-structure registered masks into a single label map.

    Parameters
    ----------
    reference : sitk.Image
        Reference geometry image.
    structure_results : dict
        Output from ``register_per_structure``.
    registry : LabelRegistry, optional
        Used to map structure names back to label values.

    Returns
    -------
    sitk.Image
        Combined multi-label image.
    """
    if registry is None:
        registry = LabelRegistry.template_preset()

    combined = sitk.Image(reference.GetSize(), sitk.sitkUInt16)
    combined.CopyInformation(reference)

    for struct_name, data in structure_results.items():
        mask = data["mask"]

        # Strip side suffix for label lookup
        base_name = struct_name.rstrip("_LR").rstrip("_")
        if base_name.endswith("_L") or base_name.endswith("_R"):
            base_name = base_name[:-2]
        if base_name not in registry:
            # try with full name
            base_name = struct_name

        try:
            label_val = registry.label(base_name)
        except KeyError:
            # Use a hash-based fallback
            label_val = abs(hash(struct_name)) % 255 + 1

        mask_u16 = sitk.Cast(mask > 0, sitk.sitkUInt16)
        val_img = sitk.Image(reference.GetSize(), sitk.sitkUInt16)
        val_img.CopyInformation(reference)
        val_img = val_img + label_val

        inv_mask = sitk.Cast(sitk.Not(mask > 0), sitk.sitkUInt16)
        combined = combined * inv_mask + val_img * mask_u16

    return combined


# ============================================================================
# HIGH-LEVEL API (operates on HipCase)
# ============================================================================


def register_hip_cases(
    fixed_case: "HipCase",
    moving_case: "HipCase",
    per_structure: bool = True,
    method: str = "rigid",
    iterations: int = 200,
) -> Dict:
    """Register two HipCase objects.

    Parameters
    ----------
    fixed_case, moving_case : HipCase
        Cases with label maps.
    per_structure : bool
        If True, register each bone independently.
    method : str
        ``'rigid'`` or ``'affine'``.
    iterations : int
        Max optimiser iterations.

    Returns
    -------
    dict
        Keys: ``'registered_labelmap'`` (HipLabelMap),
        ``'transforms'`` (per-structure dict or single transform),
        ``'combined_image'`` (sitk.Image).
    """
    if not fixed_case.has_labelmap or not moving_case.has_labelmap:
        raise RuntimeError("Both cases must have label maps for registration.")

    fixed_img = fixed_case.labelmap.getImage()
    moving_img = moving_case.labelmap.getImage()
    registry = fixed_case.registry

    if per_structure:
        struct_results = register_per_structure(
            fixed_img, moving_img,
            registry=registry, method=method, iterations=iterations,
        )
        combined = combine_registered_masks(fixed_img, struct_results, registry)
        result_lm = HipLabelMap(image=combined, registry=registry.copy())
        return {
            "registered_labelmap": result_lm,
            "transforms": struct_results,
            "combined_image": combined,
        }
    else:
        registered, transform = register_whole_hip(
            fixed_img, moving_img,
            registry=registry, method=method, iterations=iterations,
        )
        result_lm = HipLabelMap(image=registered, registry=registry.copy())
        return {
            "registered_labelmap": result_lm,
            "transforms": {"whole": transform},
            "combined_image": registered,
        }
