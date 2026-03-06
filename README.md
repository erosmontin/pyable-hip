# pyable-hip

Hip-specific medical image analysis built on [pyable](https://github.com/erosmontin/pyable).

## Features

- **HipLabelMap** – bilateral label-map handling with tissue extraction
- **HipCase** – patient container linking image, labelmap, and metadata
- **Registration** – per-structure and whole-hip registration workflows
- **Comparison** – pre/post overlap metrics, distance maps, change maps
- **Morphometrics** – cartilage thickness, coverage, alpha angle, volumes
- **Refinement** – atlas-based segmentation refinement with bone-constrained cartilage

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from pyable_hip import HipCase, HipLabelMap, LabelRegistry

registry = LabelRegistry.template_preset()
case = HipCase(image_path="image.nii", labelmap_path="labels.nii", registry=registry)
left, right = case.getBothSides()
```
