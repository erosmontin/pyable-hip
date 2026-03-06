"""
VTP surface meshes colored by distance-to-post.

Rewrite of vtp_distancemap.py using HipLabelMap classes.
"""

from pyable_hip import HipLabelMap, LabelRegistry


def main():
    pre_label_path = "/home/eros/Downloads/Hip_Patient_P12_1.nii.gz"
    post_registered_path = "post_registered_3parts_to_pre.nii.gz"

    registry = LabelRegistry.patient_preset()

    pre = HipLabelMap(pre_label_path, registry=registry)
    post_reg = HipLabelMap(post_registered_path, registry=registry)

    # For each tissue, export VTP surface colored by distance to post
    for name in registry.names():
        pre_roi = pre.extractTissue(name)
        post_roi = post_reg.extractTissue(name)

        out_vtp = f"{name}_pre_surface_dist_to_post.vtp"
        pre_roi.exportSurfaceVTP(out_vtp, distance_to=post_roi)

        print(f"Wrote: {out_vtp}")

    print("\nOpen in ParaView -> Color by 'dist_mm'.")


if __name__ == "__main__":
    main()
