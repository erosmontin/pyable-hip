"""
Whole-hip registration (union mask) + distance maps.

Rewrite of maps.py using HipLabelMap classes.
"""

from pyable_hip import HipLabelMap, LabelRegistry


def main():
    pre_path = "/home/eros/Downloads/Hip_Patient_P12_1.nii.gz"
    post_path = "/home/eros/Downloads/Hip_Patient_P12_2.nii.gz"

    registry = LabelRegistry.patient_preset()

    pre = HipLabelMap(pre_path, registry=registry)
    post = HipLabelMap(post_path, registry=registry)

    # 1) Register post -> pre using all labels (union mask)
    post_registered, transform = post.registerTo(pre, method='rigid', iterations=200)
    post_registered.writeImageAs("post_registered_to_pre.nii.gz")

    # Wrap registered result as HipLabelMap to use extractTissue
    post_reg = HipLabelMap(image=post_registered.getImage(), registry=registry)

    # 2) Distance maps per structure (PRE surface -> POST)
    for name in registry.names():
        pre_roi = pre.extractTissue(name)
        post_roi = post_reg.extractTissue(name)

        dist_map = pre_roi.getSurfaceDistanceMap(post_roi)
        dist_map.writeImageAs(f"dist_pre_to_post_{name}.nii.gz")
        print(f"  dist_pre_to_post_{name}.nii.gz")

    print("Done.")


if __name__ == "__main__":
    main()
