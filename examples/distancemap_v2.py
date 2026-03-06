"""
Surface distance metrics (Dice, HD95, mean distance, etc.) per ROI.

Rewrite of distancemap.py using HipLabelMap classes.
"""

from pyable_hip import HipLabelMap, LabelRegistry


def main():
    pre_path = "/home/eros/Downloads/Hip_Patient_P12_1.nii.gz"
    post_registered_path = "post_registered_to_pre.nii.gz"

    registry = LabelRegistry.patient_preset()

    pre = HipLabelMap(pre_path, registry=registry)
    post = HipLabelMap(post_registered_path, registry=registry)

    # Per-tissue comparison using Roiable.compareTo
    for name in registry.names():
        pre_roi = pre.extractTissue(name)
        post_roi = post.extractTissue(name)

        metrics = pre_roi.compareTo(post_roi)

        print(f"\n{name}")
        for key, val in metrics.items():
            print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
