"""
Surface distance maps between pre and post-treatment ROIs.

Rewrite of distance_map.py using HipLabelMap classes.
"""

from pyable_hip import HipLabelMap, LabelRegistry


def main():
    pre_path = "/home/eros/Downloads/Hip_Patient_P12_1.nii.gz"
    post_registered_path = "post_registered_to_pre.nii.gz"

    registry = LabelRegistry.patient_preset()

    pre = HipLabelMap(pre_path, registry=registry)
    post = HipLabelMap(post_registered_path, registry=registry)

    # For each tissue, extract ROIs and compute surface distance maps
    for name in registry.names():
        pre_roi = pre.extractTissue(name)
        post_roi = post.extractTissue(name)

        # Distance from PRE surface to POST mask (and vice versa)
        dist_a_to_b = pre_roi.getSurfaceDistanceMap(post_roi)
        dist_b_to_a = post_roi.getSurfaceDistanceMap(pre_roi)

        dist_a_to_b.writeImageAs(f"surf_dist_{name}_AtoB.nii.gz")
        dist_b_to_a.writeImageAs(f"surf_dist_{name}_BtoA.nii.gz")
        print(f"{name}: wrote surf_dist_{name}_AtoB/BtoA.nii.gz")


if __name__ == "__main__":
    main()
