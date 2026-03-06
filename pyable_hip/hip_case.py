"""
HipCase — Container for a single hip patient case.

Holds one or more images (e.g. WO, FO, Dixon) and a label map, with
metadata about laterality.  Provides high-level access to sides, images,
and ROIs.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import SimpleITK as sitk

from pyable import Imaginable

from .constants import LabelRegistry
from .hip_labelmap import HipLabelMap


class HipCase:
    """Container for a patient hip case.

    A case may be **bilateral** (both hips in one image) or **unilateral**
    (one side only).  The class stores the reference image(s) and the
    associated label map, and exposes convenience methods for side
    extraction, ROI access, and metadata.

    Parameters
    ----------
    image : str or Imaginable
        Primary image (e.g. water-only Dixon).
    labelmap : str or HipLabelMap, optional
        Label-map file or object.
    extra_images : dict[str, str | Imaginable], optional
        Additional contrast images keyed by name (e.g. ``{'fo': 'fo.nii'}``).
    patient_id : str, optional
        Unique patient identifier.
    side : str or None
        ``'L'``, ``'R'``, or ``None`` (bilateral).
    registry : LabelRegistry, optional
        Tissue label registry.  Defaults to template preset.
    orientation : str
        Desired DICOM orientation (default ``'LPS'``).
    resolution : list[float], optional
        If provided, resample images to this spacing on load.

    Examples
    --------
    >>> case = HipCase('wo.nii', 'seg.nii', patient_id='C-1')
    >>> left_case = case.getSide('L')
    >>> femur = case.getRoi('femur')
    """

    def __init__(
        self,
        image,
        labelmap=None,
        extra_images: Optional[Dict[str, object]] = None,
        patient_id: Optional[str] = None,
        side: Optional[str] = None,
        registry: Optional[LabelRegistry] = None,
        orientation: str = "LPS",
        resolution: Optional[List[float]] = None,
    ):
        self.patient_id = patient_id
        self.side = side.upper() if side else None
        self.orientation = orientation
        self.registry = registry or LabelRegistry.template_preset()

        # Load primary image
        self.image = self._load_image(image, resolution)

        # Load extra images
        self.extra_images: Dict[str, Imaginable] = {}
        if extra_images:
            for name, img in extra_images.items():
                self.extra_images[name] = self._load_image(img, resolution)

        # Load label map
        self.labelmap: Optional[HipLabelMap] = None
        if labelmap is not None:
            self.labelmap = self._load_labelmap(labelmap, resolution)

    # ========================================================================
    # INTERNAL LOADERS
    # ========================================================================

    def _load_image(
        self, src, resolution: Optional[List[float]] = None,
    ) -> Imaginable:
        if isinstance(src, str):
            img = Imaginable(src)
        elif isinstance(src, Imaginable):
            img = src.getDuplicate()
        else:
            img = Imaginable(image=src)

        if resolution is not None:
            img.changeImageSpacing(resolution)
        img.dicomOrient(self.orientation)
        return img

    def _load_labelmap(
        self, src, resolution: Optional[List[float]] = None,
    ) -> HipLabelMap:
        if isinstance(src, HipLabelMap):
            lm = HipLabelMap(image=src.getImage(), registry=self.registry.copy())
        elif isinstance(src, str):
            lm = HipLabelMap(src, registry=self.registry.copy())
        else:
            lm = HipLabelMap(image=src, registry=self.registry.copy())

        if resolution is not None:
            lm.changeImageSpacing(resolution)
        lm.dicomOrient(self.orientation)
        return lm

    # ========================================================================
    # PROPERTIES
    # ========================================================================

    @property
    def is_bilateral(self) -> bool:
        return self.side is None

    @property
    def has_labelmap(self) -> bool:
        return self.labelmap is not None

    # ========================================================================
    # SIDE ACCESS
    # ========================================================================

    def getSide(self, side: str, resolution: Optional[List[float]] = None) -> "HipCase":
        """Extract one side from a bilateral case.

        Parameters
        ----------
        side : str
            ``'L'`` or ``'R'``.
        resolution : list[float], optional
            Resample to this spacing.

        Returns
        -------
        HipCase
            Unilateral case for the requested side.
        """
        side = side.upper()
        if not self.is_bilateral:
            if self.side == side:
                return self  # already the right side
            raise ValueError(
                f"Case is unilateral (side={self.side}), cannot extract '{side}'"
            )

        def _crop_side(img: Imaginable, s: str) -> Imaginable:
            dup = img.getDuplicate()
            if resolution:
                dup.changeImageSpacing(resolution)
            dup.dicomOrient(self.orientation)
            sz = dup.getImageSize()
            half = sz[0] // 2
            if s == "L":
                dup.cropImage([half, 0, 0], [0, 0, 0])
            else:
                dup.cropImage([0, 0, 0], [half, 0, 0])
            return dup

        new_image = _crop_side(self.image, side)

        new_extra = {}
        for name, img in self.extra_images.items():
            new_extra[name] = _crop_side(img, side)

        new_lm = None
        if self.labelmap is not None:
            new_lm = self.labelmap.prepareSide(side, resolution=resolution)

        return HipCase(
            image=new_image,
            labelmap=new_lm,
            extra_images=new_extra if new_extra else None,
            patient_id=self.patient_id,
            side=side,
            registry=self.registry.copy(),
            orientation=self.orientation,
        )

    def getBothSides(
        self, resolution: Optional[List[float]] = None,
    ) -> Tuple["HipCase", "HipCase"]:
        """Split a bilateral case into (left, right)."""
        return self.getSide("L", resolution), self.getSide("R", resolution)

    # ========================================================================
    # ROI ACCESS
    # ========================================================================

    def getRoi(self, tissue_name: str) -> "Roiable":
        """Get a binary ROI for a named tissue.

        Parameters
        ----------
        tissue_name : str
            Tissue name from the registry (e.g. ``'femur'``).

        Returns
        -------
        Roiable
        """
        if self.labelmap is None:
            raise RuntimeError("No label map loaded for this case.")
        return self.labelmap.extractTissue(tissue_name)

    def getAllRois(self) -> Dict[str, "Roiable"]:
        """Get all registered tissue ROIs present in the label map."""
        if self.labelmap is None:
            raise RuntimeError("No label map loaded for this case.")
        return self.labelmap.extractAll()

    # ========================================================================
    # IMAGE ACCESS
    # ========================================================================

    def getImage(self, name: Optional[str] = None) -> Imaginable:
        """Return the primary image, or a named extra image.

        Parameters
        ----------
        name : str, optional
            If given, return the extra image with that name.
        """
        if name is None:
            return self.image
        if name not in self.extra_images:
            raise KeyError(
                f"Extra image '{name}' not found. "
                f"Available: {list(self.extra_images.keys())}"
            )
        return self.extra_images[name]

    # ========================================================================
    # WRITE / EXPORT
    # ========================================================================

    def write(self, out_dir: str, fmt: str = "nii") -> Dict[str, str]:
        """Write all images and label map to *out_dir*.

        Returns
        -------
        dict[str, str]
            Mapping of keys → output paths.
        """
        os.makedirs(out_dir, exist_ok=True)
        paths = {}

        img_path = os.path.join(out_dir, f"image.{fmt}")
        self.image.writeImageAs(img_path)
        paths["image"] = img_path

        for name, img in self.extra_images.items():
            p = os.path.join(out_dir, f"{name}.{fmt}")
            img.writeImageAs(p)
            paths[name] = p

        if self.labelmap is not None:
            lm_path = os.path.join(out_dir, f"labelmap.{fmt}")
            self.labelmap.writeImageAs(lm_path)
            paths["labelmap"] = lm_path

        return paths

    # ========================================================================
    # REPR
    # ========================================================================

    def __repr__(self) -> str:
        parts = [f"HipCase(patient_id={self.patient_id!r}"]
        parts.append(f"side={self.side!r}")
        parts.append(f"bilateral={self.is_bilateral}")
        parts.append(f"has_labelmap={self.has_labelmap}")
        if self.extra_images:
            parts.append(f"extra_images={list(self.extra_images.keys())}")
        return ", ".join(parts) + ")"
