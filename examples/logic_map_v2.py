"""
Boolean change maps (removed / added / changed) per ROI.

Rewrite of logic_map.py using HipLabelMap classes.
"""

from pyable_hip import HipLabelMap, LabelRegistry


def main():
    pre_path = "/home/eros/Downloads/Hip_Patient_P12_1.nii.gz"
    post_registered_path = "post_registered_to_pre.nii.gz"

    registry = LabelRegistry.patient_preset()

    pre = HipLabelMap(pre_path, registry=registry)
    post = HipLabelMap(post_registered_path, registry=registry)

    # Per-tissue change maps using Roiable.getChangeMaps
    for name in registry.names():
        pre_roi = pre.extractTissue(name)
        post_roi = post.extractTissue(name)

        changes = pre_roi.getChangeMaps(post_roi)

        for key, roi in changes.items():
            roi.writeImageAs(f"{name}_{key}.nii.gz")

        print(f"{name} -> wrote {name}_removed/added/changed.nii.gz")


if __name__ == "__main__":
    main()
