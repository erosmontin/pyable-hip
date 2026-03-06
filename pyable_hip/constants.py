"""
Label registry and default tissue configurations for hip analysis.

The LabelRegistry maps human-readable tissue names to integer label values.
Ship two built-in presets (``TEMPLATE`` and ``PATIENT``) and let users
register arbitrary additional tissues.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


# ============================================================================
# Tissue descriptor
# ============================================================================


@dataclass
class Tissue:
    """Descriptor for a single tissue type.

    Parameters
    ----------
    name : str
        Human-readable name (e.g. ``'femur'``).
    label : int
        Integer value in the label map.
    category : str
        One of ``'bone'``, ``'cartilage'``, ``'muscle'``, ``'other'``.
    side : str or None
        ``'left'``, ``'right'``, or ``None`` (bilateral / unspecified).
    parent : str or None
        Name of a parent tissue (e.g. cartilage → bone it attaches to).
    """

    name: str
    label: int
    category: str = "other"
    side: Optional[str] = None
    parent: Optional[str] = None


# ============================================================================
# Label registry
# ============================================================================


class LabelRegistry:
    """Configurable mapping from tissue names to label values.

    Examples
    --------
    >>> reg = LabelRegistry.template_preset()
    >>> reg['femur']
    Tissue(name='femur', label=2, category='bone', ...)
    >>> reg.label('femur')
    2
    >>> reg.bones()
    ['femur', 'acetabulum']
    """

    def __init__(self, tissues: Optional[Sequence[Tissue]] = None):
        self._tissues: Dict[str, Tissue] = {}
        if tissues:
            for t in tissues:
                self.register(t)

    # -- mutators ------------------------------------------------------------

    def register(self, tissue: Tissue) -> None:
        """Add or replace a tissue entry."""
        self._tissues[tissue.name] = tissue

    def unregister(self, name: str) -> None:
        self._tissues.pop(name, None)

    # -- queries -------------------------------------------------------------

    def __getitem__(self, name: str) -> Tissue:
        return self._tissues[name]

    def __contains__(self, name: str) -> bool:
        return name in self._tissues

    def __iter__(self):
        return iter(self._tissues.values())

    def __len__(self) -> int:
        return len(self._tissues)

    def names(self) -> List[str]:
        return list(self._tissues.keys())

    def label(self, name: str) -> int:
        """Return the integer label for *name*."""
        return self._tissues[name].label

    def name_for_label(self, label_value: int) -> Optional[str]:
        """Reverse lookup: label value → tissue name (first match)."""
        for t in self._tissues.values():
            if t.label == label_value:
                return t.name
        return None

    def by_category(self, category: str) -> List[str]:
        """Return tissue names matching *category*."""
        return [t.name for t in self._tissues.values()
                if t.category == category]

    def bones(self) -> List[str]:
        return self.by_category("bone")

    def cartilages(self) -> List[str]:
        return self.by_category("cartilage")

    def muscles(self) -> List[str]:
        return self.by_category("muscle")

    def label_to_name_dict(self) -> Dict[int, str]:
        """Return ``{label_value: name}`` mapping."""
        return {t.label: t.name for t in self._tissues.values()}

    def name_to_label_dict(self) -> Dict[str, int]:
        """Return ``{name: label_value}`` mapping."""
        return {t.name: t.label for t in self._tissues.values()}

    def copy(self) -> "LabelRegistry":
        return LabelRegistry(list(deepcopy(t) for t in self._tissues.values()))

    # -- presets -------------------------------------------------------------

    @classmethod
    def template_preset(cls) -> "LabelRegistry":
        """Default atlas / template label scheme.

        =============================  =====  ==========
        Tissue                         Label  Category
        =============================  =====  ==========
        acetabular_cartilage              1   cartilage
        femur                             2   bone
        femur_cartilage                   3   cartilage
        acetabulum                        4   bone
        =============================  =====  ==========
        """
        return cls([
            Tissue("acetabular_cartilage", 1, "cartilage", parent="acetabulum"),
            Tissue("femur", 2, "bone"),
            Tissue("femur_cartilage", 3, "cartilage", parent="femur"),
            Tissue("acetabulum", 4, "bone"),
        ])

    @classmethod
    def patient_preset(cls) -> "LabelRegistry":
        """Common patient-segmentation label scheme.

        =============================  =====  ==========
        Tissue                         Label  Category
        =============================  =====  ==========
        acetabulum                        1   bone
        femur                             2   bone
        =============================  =====  ==========
        """
        return cls([
            Tissue("acetabulum", 1, "bone"),
            Tissue("femur", 2, "bone"),
        ])

    def __repr__(self) -> str:
        items = ", ".join(f"{t.name}={t.label}" for t in self._tissues.values())
        return f"LabelRegistry({items})"


# ============================================================================
# ROI processing config (per-tissue refinement settings)
# ============================================================================

@dataclass
class RoiConfig:
    """Per-tissue processing parameters for atlas refinement.

    Parameters
    ----------
    atlas_label : int
        Label value in the atlas/template segmentation.
    patient_label : int or None
        Label value in the patient ground-truth (None if unavailable).
    needs_z_extension : bool
        Whether the ROI may need FOV extension along Z.
    needs_boundary_refinement : bool
        Whether geodesic / region-growing refinement should run.
    has_ground_truth : bool
        Whether a patient ground-truth exists for this tissue.
    constraint_roi : str or None
        Name of a bone ROI that constrains this tissue (e.g. cartilage → bone).
    """

    atlas_label: int
    patient_label: Optional[int] = None
    needs_z_extension: bool = False
    needs_boundary_refinement: bool = True
    has_ground_truth: bool = False
    constraint_roi: Optional[str] = None


# Default ROI processing configs (matches refine_atlas_complete.py)
DEFAULT_ROI_CONFIGS: Dict[str, RoiConfig] = {
    "femur": RoiConfig(
        atlas_label=2, patient_label=2,
        needs_z_extension=True, has_ground_truth=True,
    ),
    "acetabulum": RoiConfig(
        atlas_label=4, patient_label=1,
        needs_z_extension=True, has_ground_truth=True,
    ),
    "femur_cartilage": RoiConfig(
        atlas_label=3, patient_label=None,
        needs_z_extension=False, has_ground_truth=False,
        constraint_roi="femur",
    ),
    "acetabular_cartilage": RoiConfig(
        atlas_label=1, patient_label=None,
        needs_z_extension=False, has_ground_truth=False,
        constraint_roi="acetabulum",
    ),
}
