"""
Hip-specific atlas-based segmentation refinement.

Provides geodesic active contour, region-growing, and probability-based
refinement for hip tissues, with bone-constrained cartilage refinement
and Z-extension for FOV mismatches.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    distance_transform_edt,
    gaussian_gradient_magnitude,
    label as nd_label,
)

from pyable import Roiable, Imaginable, LabelMapable

from .constants import LabelRegistry, RoiConfig, DEFAULT_ROI_CONFIGS
from .hip_labelmap import HipLabelMap


# ============================================================================
# SMALL COMPONENT FILTERING
# ============================================================================


def remove_small_components(
    roi: sitk.Image,
    min_size_voxels: int = 10,
) -> sitk.Image:
    """Remove connected components smaller than *min_size_voxels*.

    Parameters
    ----------
    roi : sitk.Image
        Binary ROI.
    min_size_voxels : int
        Minimum component size to keep.

    Returns
    -------
    sitk.Image
        Filtered binary ROI (UInt8).
    """
    binary = sitk.BinaryThreshold(roi, 1, 255, 1, 0)
    labeled = sitk.ConnectedComponent(binary)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(labeled)

    for lab in stats.GetLabels():
        if stats.GetNumberOfPixels(lab) < min_size_voxels:
            labeled = sitk.BinaryThreshold(
                labeled, lab, lab, 0, 1,
            ) * labeled  # zero out that label
            # Simpler approach: threshold out
    # Rebuild: keep only labels with enough voxels
    keep = sitk.Image(roi.GetSize(), sitk.sitkUInt8)
    keep.CopyInformation(roi)
    for lab in stats.GetLabels():
        if stats.GetNumberOfPixels(lab) >= min_size_voxels:
            comp = sitk.Equal(labeled, lab)
            keep = keep | sitk.Cast(comp, sitk.sitkUInt8)

    return keep


# ============================================================================
# Z-EXTENSION (FOV mismatch compensation)
# ============================================================================


def clip_roi_to_reference_z(
    roi: sitk.Image,
    reference: sitk.Image,
    margin_slices: int = 2,
) -> sitk.Image:
    """Clip a ROI to the Z-range of a reference image's non-zero region.

    Useful when the atlas FOV extends beyond the patient image.

    Parameters
    ----------
    roi : sitk.Image
        Binary ROI to clip.
    reference : sitk.Image
        Reference image (e.g. patient segmentation).
    margin_slices : int
        Extra slices of margin to keep beyond the reference extent.

    Returns
    -------
    sitk.Image
        Clipped binary ROI.
    """
    roi = sitk.Cast(roi > 0, sitk.sitkUInt8)
    ref_arr = sitk.GetArrayFromImage(reference)  # ZYX

    # Find non-zero Z range in reference
    nonzero_z = np.where(ref_arr.sum(axis=(1, 2)) > 0)[0]
    if nonzero_z.size == 0:
        return roi  # reference is empty, return as-is

    z_min = max(0, nonzero_z.min() - margin_slices)
    z_max = min(ref_arr.shape[0] - 1, nonzero_z.max() + margin_slices)

    roi_arr = sitk.GetArrayFromImage(roi).copy()
    roi_arr[:z_min] = 0
    roi_arr[z_max + 1:] = 0

    clipped = sitk.GetImageFromArray(roi_arr)
    clipped.CopyInformation(roi)
    return sitk.Cast(clipped, sitk.sitkUInt8)


# ============================================================================
# GEODESIC ACTIVE CONTOUR REFINEMENT
# ============================================================================


def geodesic_refinement(
    initial_roi: sitk.Image,
    intensity_image: sitk.Image,
    propagation: float = -0.5,
    curvature: float = 1.0,
    advection: float = 1.0,
    iterations: int = 100,
    rms_error: float = 0.01,
    sigma_mm: float = 1.0,
) -> sitk.Image:
    """Geodesic active contour refinement of an initial ROI.

    Parameters
    ----------
    initial_roi : sitk.Image
        Initial binary segmentation (seed).
    intensity_image : sitk.Image
        Grayscale image for edge detection.
    propagation : float
        Propagation scaling (negative = shrink, positive = grow).
    curvature : float
        Curvature weight (higher = smoother).
    advection : float
        Advection weight (edge attraction).
    iterations : int
        Max iterations.
    rms_error : float
        Convergence threshold.
    sigma_mm : float
        Gaussian sigma for edge detection.

    Returns
    -------
    sitk.Image
        Refined binary ROI (UInt8).
    """
    # Signed distance map as level-set initialisation
    init_f = sitk.Cast(initial_roi > 0, sitk.sitkFloat32)
    distance = sitk.SignedMaurerDistanceMap(
        sitk.Cast(initial_roi > 0, sitk.sitkUInt8),
        insideIsPositive=False,
        squaredDistance=False,
        useImageSpacing=True,
    )

    # Edge potential (speed image)
    img_f = sitk.Cast(intensity_image, sitk.sitkFloat32)
    grad = sitk.GradientMagnitudeRecursiveGaussian(img_f, sigma=sigma_mm)
    speed = sitk.BoundedReciprocal(grad)

    # Geodesic active contour
    gac = sitk.GeodesicActiveContourLevelSetImageFilter()
    gac.SetPropagationScaling(propagation)
    gac.SetCurvatureScaling(curvature)
    gac.SetAdvectionScaling(advection)
    gac.SetMaximumRMSError(rms_error)
    gac.SetNumberOfIterations(iterations)

    ls_out = gac.Execute(
        sitk.Cast(distance, sitk.sitkFloat32),
        sitk.Cast(speed, sitk.sitkFloat32),
    )

    # Threshold level-set output (negative = inside)
    refined = sitk.BinaryThreshold(ls_out, -1e10, 0, 1, 0)
    return sitk.Cast(refined, sitk.sitkUInt8)


# ============================================================================
# INTENSITY-BASED EXPANSION / SHRINKAGE
# ============================================================================


def intensity_based_refinement(
    initial_roi: sitk.Image,
    intensity_image: sitk.Image,
    method: str = "confidence_connected",
    multiplier: float = 2.5,
    n_iterations: int = 3,
    radius: int = 2,
) -> sitk.Image:
    """Region-growing refinement using intensity information.

    Parameters
    ----------
    initial_roi : sitk.Image
        Initial binary segmentation.
    intensity_image : sitk.Image
        Grayscale image.
    method : str
        ``'confidence_connected'`` or ``'neighbourhood'``.
    multiplier : float
        Confidence multiplier.
    n_iterations : int
        Number of growing iterations.
    radius : int
        Neighbourhood radius.

    Returns
    -------
    sitk.Image
        Refined binary ROI (UInt8).
    """
    roi_arr = sitk.GetArrayFromImage(initial_roi) > 0
    seeds = np.argwhere(roi_arr)
    if seeds.size == 0:
        return sitk.Cast(initial_roi, sitk.sitkUInt8)

    # Sample up to 500 seed points
    if seeds.shape[0] > 500:
        idx = np.random.choice(seeds.shape[0], 500, replace=False)
        seeds = seeds[idx]

    img_f = sitk.Cast(intensity_image, sitk.sitkFloat32)

    if method == "confidence_connected":
        cc = sitk.ConfidenceConnectedImageFilter()
        cc.SetMultiplier(multiplier)
        cc.SetNumberOfIterations(n_iterations)
        cc.SetInitialNeighborhoodRadius(radius)
        # Seeds are (z, y, x) but sitk wants (x, y, z)
        for s in seeds:
            cc.AddSeed((int(s[2]), int(s[1]), int(s[0])))
        result = cc.Execute(img_f)
    else:
        # Fallback: neighbourhood connected
        nc = sitk.NeighborhoodConnectedImageFilter()
        nc.SetRadius([radius] * 3)
        img_arr = sitk.GetArrayViewFromImage(img_f)
        roi_vals = img_arr[roi_arr]
        nc.SetLower(float(roi_vals.mean() - multiplier * roi_vals.std()))
        nc.SetUpper(float(roi_vals.mean() + multiplier * roi_vals.std()))
        for s in seeds:
            nc.AddSeed((int(s[2]), int(s[1]), int(s[0])))
        result = nc.Execute(img_f)

    return sitk.Cast(result, sitk.sitkUInt8)


# ============================================================================
# BONE-CONSTRAINED CARTILAGE REFINEMENT
# ============================================================================


def refine_cartilage(
    cartilage_roi: sitk.Image,
    bone_roi: sitk.Image,
    intensity_image: sitk.Image,
    max_distance_mm: float = 5.0,
    propagation: float = -0.3,
    curvature: float = 1.5,
    min_component_voxels: int = 20,
) -> sitk.Image:
    """Refine a cartilage ROI constrained to stay near its parent bone.

    Pipeline:
        1. Geodesic refinement for boundary smoothing
        2. Clip to within *max_distance_mm* of bone surface
        3. Remove small disconnected components

    Parameters
    ----------
    cartilage_roi : sitk.Image
        Initial cartilage binary mask.
    bone_roi : sitk.Image
        Parent bone binary mask (e.g. femur for femur_cartilage).
    intensity_image : sitk.Image
        Grayscale image for GAC.
    max_distance_mm : float
        Maximum allowed distance from bone surface.
    propagation : float
        GAC propagation (negative = conservative).
    curvature : float
        GAC curvature weight.
    min_component_voxels : int
        Remove components smaller than this.

    Returns
    -------
    sitk.Image
        Refined cartilage ROI (UInt8).
    """
    # Step 1: Geodesic refinement
    refined = geodesic_refinement(
        cartilage_roi, intensity_image,
        propagation=propagation,
        curvature=curvature,
    )

    # Step 2: Distance constraint to bone
    bone_dist = sitk.Abs(sitk.SignedMaurerDistanceMap(
        sitk.Cast(bone_roi > 0, sitk.sitkUInt8),
        insideIsPositive=False,
        squaredDistance=False,
        useImageSpacing=True,
    ))
    # Keep only cartilage within max_distance_mm of bone, but NOT inside bone
    bone_arr = sitk.GetArrayFromImage(bone_roi) > 0
    dist_arr = sitk.GetArrayFromImage(bone_dist)
    refined_arr = sitk.GetArrayFromImage(refined) > 0

    # Mask: within distance and outside bone
    valid = (dist_arr <= max_distance_mm) & (~bone_arr)
    refined_arr = refined_arr & valid

    out = sitk.GetImageFromArray(refined_arr.astype(np.uint8))
    out.CopyInformation(cartilage_roi)

    # Step 3: Remove small components
    out = remove_small_components(out, min_component_voxels)

    return out


# ============================================================================
# MORPHOLOGICAL SMOOTHING
# ============================================================================


def smooth_roi(
    roi: sitk.Image,
    radius: int = 1,
    mode: str = "closing",
) -> sitk.Image:
    """Binary morphological smoothing.

    Parameters
    ----------
    roi : sitk.Image
        Binary ROI.
    radius : int
        Kernel radius.
    mode : str
        ``'closing'``, ``'opening'``, or ``'both'``.

    Returns
    -------
    sitk.Image
        Smoothed binary ROI (UInt8).
    """
    img = sitk.Cast(roi > 0, sitk.sitkUInt8)
    if mode in ("closing", "both"):
        img = sitk.BinaryMorphologicalClosing(img, [radius] * img.GetDimension())
    if mode in ("opening", "both"):
        img = sitk.BinaryMorphologicalOpening(img, [radius] * img.GetDimension())
    return sitk.Cast(img, sitk.sitkUInt8)


# ============================================================================
# FULL HIP REFINEMENT PIPELINE
# ============================================================================


def refine_hip_segmentation(
    atlas_labelmap: sitk.Image,
    patient_image: sitk.Image,
    patient_labelmap: Optional[sitk.Image] = None,
    registry: Optional[LabelRegistry] = None,
    roi_configs: Optional[Dict[str, RoiConfig]] = None,
    min_component_voxels: int = 20,
) -> sitk.Image:
    """Full atlas-based segmentation refinement for a hip case.

    Pipeline for each tissue:
        1. Extract atlas ROI
        2. Z-clip to patient FOV (if needed)
        3. Geodesic refinement (bones) or bone-constrained refinement (cartilages)
        4. Clean small components
        5. Reassemble into a multi-label image

    Parameters
    ----------
    atlas_labelmap : sitk.Image
        Atlas segmentation warped into patient space.
    patient_image : sitk.Image
        Patient intensity image (for GAC edge detection).
    patient_labelmap : sitk.Image, optional
        Patient segmentation (for Z-clipping reference).
    registry : LabelRegistry, optional
    roi_configs : dict[str, RoiConfig], optional
        Per-tissue processing config.  Defaults to ``DEFAULT_ROI_CONFIGS``.
    min_component_voxels : int
        Remove components smaller than this.

    Returns
    -------
    sitk.Image
        Refined multi-label segmentation.
    """
    if registry is None:
        registry = LabelRegistry.template_preset()
    if roi_configs is None:
        roi_configs = DEFAULT_ROI_CONFIGS

    # Output array
    ref_size = atlas_labelmap.GetSize()
    result_arr = np.zeros(
        sitk.GetArrayFromImage(atlas_labelmap).shape, dtype=np.uint16,
    )

    # First pass: refine bones (they are needed as constraints for cartilages)
    bone_masks = {}
    for name in registry.bones():
        cfg = roi_configs.get(name)
        if cfg is None:
            continue

        atlas_roi = sitk.Cast(
            sitk.Equal(atlas_labelmap, cfg.atlas_label), sitk.sitkUInt8,
        )
        if sitk.GetArrayViewFromImage(atlas_roi).sum() == 0:
            continue

        # Z-clipping
        if cfg.needs_z_extension and patient_labelmap is not None:
            atlas_roi = clip_roi_to_reference_z(atlas_roi, patient_labelmap)

        # Geodesic refinement
        if cfg.needs_boundary_refinement:
            atlas_roi = geodesic_refinement(
                atlas_roi, patient_image, propagation=-0.5,
            )

        # Clean
        atlas_roi = remove_small_components(atlas_roi, min_component_voxels)
        atlas_roi = smooth_roi(atlas_roi, radius=1, mode="both")

        bone_masks[name] = atlas_roi
        label_val = cfg.atlas_label
        roi_arr = sitk.GetArrayFromImage(atlas_roi) > 0
        result_arr[roi_arr] = label_val

    # Second pass: refine cartilages (constrained by parent bones)
    for name in registry.cartilages():
        cfg = roi_configs.get(name)
        if cfg is None:
            continue

        atlas_roi = sitk.Cast(
            sitk.Equal(atlas_labelmap, cfg.atlas_label), sitk.sitkUInt8,
        )
        if sitk.GetArrayViewFromImage(atlas_roi).sum() == 0:
            continue

        # Bone constraint
        parent_bone = cfg.constraint_roi
        if parent_bone and parent_bone in bone_masks:
            atlas_roi = refine_cartilage(
                atlas_roi,
                bone_masks[parent_bone],
                patient_image,
            )
        elif cfg.needs_boundary_refinement:
            atlas_roi = geodesic_refinement(
                atlas_roi, patient_image, propagation=-0.3,
            )

        atlas_roi = remove_small_components(atlas_roi, min_component_voxels)

        label_val = cfg.atlas_label
        roi_arr = sitk.GetArrayFromImage(atlas_roi) > 0
        result_arr[roi_arr] = label_val

    # Build output
    result_img = sitk.GetImageFromArray(result_arr)
    result_img.CopyInformation(atlas_labelmap)
    return result_img
