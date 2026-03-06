"""
Per-structure rigid registration (femur L/R + acetabulum) + change maps.

Rewrite of three.py using HipLabelMap classes.
"""

from pyable import Roiable
from pyable_hip import HipLabelMap, LabelRegistry


def main():
    pre_path = "/home/eros/Downloads/Hip_Patient_P12_1.nii.gz"
    post_path = "/home/eros/Downloads/Hip_Patient_P12_2.nii.gz"

    registry = LabelRegistry.patient_preset()

    pre = HipLabelMap(pre_path, registry=registry)
    post = HipLabelMap(post_path, registry=registry)

    # --- Per-structure registration ---
    # Acetabulum: single structure
    pre_acet = pre.extractTissue('acetabulum')
    post_acet = post.extractTissue('acetabulum')
    post_acet_reg, T_acet = post_acet.registerTo(pre_acet, method='rigid')

    # Femur: split L/R and register each side independently
    pre_fem = pre.extractTissue('femur')
    post_fem = post.extractTissue('femur')

    pre_fem_L, pre_fem_R = pre_fem.splitLeftRight()
    post_fem_L, post_fem_R = post_fem.splitLeftRight()

    post_fem_L_reg, T_fem_L = post_fem_L.registerTo(pre_fem_L, method='rigid')
    post_fem_R_reg, T_fem_R = post_fem_R.registerTo(pre_fem_R, method='rigid')

    # Save per-structure registered masks
    for name, roi in [('acet', post_acet_reg), ('femur_L', post_fem_L_reg),
                       ('femur_R', post_fem_R_reg)]:
        roi.writeImageAs(f"post_{name}_reg_mask.nii.gz")

    # --- Change maps per structure ---
    for label, pre_roi, post_roi in [
        ('acetabulum', pre_acet, post_acet_reg),
        ('femur_left', pre_fem_L, post_fem_L_reg),
        ('femur_right', pre_fem_R, post_fem_R_reg),
    ]:
        changes = pre_roi.getChangeMaps(post_roi)
        for key, roi in changes.items():
            roi.writeImageAs(f"{label}_{key}.nii.gz")

    print("Wrote:")
    print("  post_*_reg_mask.nii.gz")
    print("  acetabulum_[removed|added|changed].nii.gz")
    print("  femur_left_[removed|added|changed].nii.gz")
    print("  femur_right_[removed|added|changed].nii.gz")


if __name__ == "__main__":
    main()
