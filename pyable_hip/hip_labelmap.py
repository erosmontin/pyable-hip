"""
HipLabelMap — LabelMapable subclass with hip-specific tissue operations.

Provides named extraction of bones, cartilages, and muscles, bilateral
splitting, and tissue-aware convenience methods.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk

from pyable import LabelMapable, Roiable, Imaginable

from .constants import LabelRegistry, Tissue


class HipLabelMap(LabelMapable):
    """A LabelMapable that knows about hip tissue types.

    Parameters
    ----------
    filename : str, optional
        Path to a label-map image file.
    image : sitk.Image, optional
        SimpleITK label image.
    registry : LabelRegistry, optional
        Tissue → label mapping.  Defaults to ``LabelRegistry.template_preset()``.
    verbose : bool
        Verbose flag forwarded to the base class.

    Examples
    --------
    >>> seg = HipLabelMap('segmentation.nii.gz')
    >>> femur = seg.extractTissue('femur')
    >>> left, right = seg.splitBilateral()
    """

    def __init__(
        self,
        filename: Optional[str] = None,
        image: Optional[sitk.Image] = None,
        registry: Optional[LabelRegistry] = None,
        verbose: bool = False,
    ):
        super().__init__(filename=filename, image=image, verbose=verbose)
        self.registry = registry or LabelRegistry.template_preset()

    # ========================================================================
    # TISSUE EXTRACTION
    # ========================================================================

    def extractTissue(self, name: str) -> Roiable:
        """Extract a single tissue as a binary Roiable.

        Parameters
        ----------
        name : str
            Tissue name registered in the label registry.

        Returns
        -------
        Roiable
            Binary mask for the requested tissue.

        Example
        -------
        >>> femur = seg.extractTissue('femur')
        """
        if name not in self.registry:
            raise KeyError(
                f"Tissue '{name}' not in registry. "
                f"Available: {self.registry.names()}"
            )
        label_val = self.registry.label(name)
        return self.extractLabel(label_val)

    def extractBones(self) -> Dict[str, Roiable]:
        """Extract all bone tissues as a dict of Roiables."""
        return {n: self.extractTissue(n) for n in self.registry.bones()}

    def extractCartilages(self) -> Dict[str, Roiable]:
        """Extract all cartilage tissues as a dict of Roiables."""
        return {n: self.extractTissue(n) for n in self.registry.cartilages()}

    def extractMuscles(self) -> Dict[str, Roiable]:
        """Extract all muscle tissues as a dict of Roiables."""
        return {n: self.extractTissue(n) for n in self.registry.muscles()}

    def extractByCategory(self, category: str) -> Dict[str, Roiable]:
        """Extract all tissues of a given category."""
        return {n: self.extractTissue(n)
                for n in self.registry.by_category(category)}

    def extractAll(self) -> Dict[str, Roiable]:
        """Extract every registered tissue as a dict of Roiables."""
        result = {}
        for tissue in self.registry:
            try:
                result[tissue.name] = self.extractTissue(tissue.name)
            except Exception:
                pass  # label not present in this image
        return result

    # ========================================================================
    # BILATERAL SPLITTING
    # ========================================================================

    def splitBilateral(
        self,
        orientation: str = "LPS",
        axis: int = 0,
    ) -> Tuple["HipLabelMap", "HipLabelMap"]:
        """Split a bilateral hip label map into left and right halves.

        Assumes the image is in the given *orientation* (default LPS) and
        halves along the specified *axis* (default 0 = X = left-right in LPS).

        Parameters
        ----------
        orientation : str
            DICOM orientation string to enforce before splitting.
        axis : int
            Axis index to halve (0 = X in LPS → left/right).

        Returns
        -------
        (left, right) : tuple[HipLabelMap, HipLabelMap]

        Example
        -------
        >>> left, right = bilateral_seg.splitBilateral()
        """
        dup = self.getDuplicate()
        dup.dicomOrient(orientation)

        size = list(dup.getImageSize())
        half = size[axis] // 2

        # Left: keep first half along axis (crop from the right)
        crop_upper_left = [0] * len(size)
        crop_upper_left[axis] = size[axis] - half
        left = dup.getDuplicate()
        crop_lower_left = [0] * len(size)
        left.cropImage(crop_lower_left, crop_upper_left)

        # Right: keep second half along axis (crop from the left)
        crop_lower_right = [0] * len(size)
        crop_lower_right[axis] = half
        right = dup.getDuplicate()
        crop_upper_right = [0] * len(size)
        right.cropImage(crop_lower_right, crop_upper_right)

        left_hlm = HipLabelMap(image=left.getImage(), registry=self.registry.copy())
        right_hlm = HipLabelMap(image=right.getImage(), registry=self.registry.copy())
        return left_hlm, right_hlm

    # ========================================================================
    # SIDE PREPARATION (matches func.py prepareLeft / prepareRight)
    # ========================================================================

    def prepareSide(
        self,
        side: str,
        resolution: Optional[List[float]] = None,
        orientation: str = "LPS",
    ) -> "HipLabelMap":
        """Prepare a single-side label map from a bilateral image.

        Orients to *orientation*, optionally resamples to *resolution*, and
        crops the appropriate half.

        Parameters
        ----------
        side : str
            ``'L'`` or ``'R'``.
        resolution : list[float], optional
            Target isotropic spacing (e.g. ``[2, 2, 2]``).
        orientation : str
            DICOM orientation string.

        Returns
        -------
        HipLabelMap
            Cropped single-side label map.
        """
        dup = self.getDuplicate()
        if resolution is not None:
            dup.changeImageSpacing(resolution)
        dup.dicomOrient(orientation)

        size = dup.getImageSize()
        half = size[0] // 2

        if side.upper() == "L":
            dup.cropImage([half, 0, 0], [0, 0, 0])
        elif side.upper() == "R":
            dup.cropImage([0, 0, 0], [half, 0, 0])
        else:
            raise ValueError(f"side must be 'L' or 'R', got '{side}'")

        return HipLabelMap(image=dup.getImage(), registry=self.registry.copy())

    # ========================================================================
    # TISSUE-AWARE WRITE / EXPORT
    # ========================================================================

    def writeAllTissues(self, out_dir: str, fmt: str = "nii") -> Dict[str, str]:
        """Write each registered tissue as a separate binary file.

        Parameters
        ----------
        out_dir : str
            Output directory (created if needed).
        fmt : str
            File extension (``'nii'``, ``'nii.gz'``, ``'mha'``).

        Returns
        -------
        dict[str, str]
            ``{tissue_name: output_path}``
        """
        import os
        os.makedirs(out_dir, exist_ok=True)
        paths = {}
        for tissue in self.registry:
            try:
                roi = self.extractTissue(tissue.name)
                if roi.getNumberOfNonZeroVoxels() == 0:
                    continue
                out_path = os.path.join(out_dir, f"{tissue.name}.{fmt}")
                roi.writeImageAs(out_path)
                paths[tissue.name] = out_path
            except Exception:
                pass
        return paths

    # ========================================================================
    # DESCRIBE
    # ========================================================================

    def describe(self) -> dict:
        """Summarise the hip label map including per-tissue voxel counts."""
        info = super().describe()
        info["registry"] = repr(self.registry)
        arr = self.getImageAsNumpy()
        counts = {}
        for tissue in self.registry:
            n = int(np.sum(arr == tissue.label))
            if n > 0:
                counts[tissue.name] = n
        info["tissue_voxels"] = counts
        self._print_describe(info)
        return info
