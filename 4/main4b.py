import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from main import (SETS, OUT_DIR, SAVED, apply_H, build_mosaic, chain_homographies, load, log, out_path,
                  pair_homographies, save_fig, save_img)
from features import (auto_homography, draw_matches, draw_matches_ransac, match_pair,
                      plot_corners, show_descriptors)

N_KEEP = 500     
N_MATCH = 2000  
RATIO = 0.8    
PART = "B_features"
RANSAC_THRESH = 4.0  
RANSAC_ITERS = 5000
MIN_INLIERS = 15   
 
def sane_homography(H, shape):
    h, w = shape[:2]
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float)
    with np.errstate(all="ignore"):
        wd = np.c_[corners, np.ones(4)] @ H.T
        if not np.all(np.isfinite(wd)) or np.any(wd[:, 2] <= 1e-6):
            return False
        c = wd[:, :2] / wd[:, 2:3]
    span = c.max(0) - c.min(0)
    return bool(np.all(np.isfinite(span)) and 0.2 * w < span[0] < 5 * w and 0.2 * h < span[1] < 5 * h)

def run_B_features():
    for set_name, stems, _ in SETS:
        ims = [load(s) for s in stems]
        for k in range(len(ims) - 1):
            r = match_pair(ims[k], ims[k + 1], N_KEEP, RATIO)
            tag = f"{set_name}_{k}to{k + 1}"
            log(f"[{tag}] {stems[k]} -> {stems[k + 1]}: {len(r['pts1'])}/{len(r['pts2'])} descriptors, "
                f"{len(r['idx1'])} matches (ratio < {RATIO})")

            name = f"{tag}_matches.png"
            draw_matches(ims[k], ims[k + 1], r["pts1"], r["pts2"], r["idx1"], r["idx2"],
                         max_lines=80, save_as=out_path(PART, name), show=False,
                         title=f"{stems[k]} -> {stems[k + 1]}, ratio < {RATIO}")
            SAVED.append(out_path(PART, name))

            if k == 0:   
                c = r["corners1"]
                for key, label in (("all", "Harris corners"), ("strongest", f"Top {N_KEEP}, no ANMS"),
                                   ("anms", f"ANMS, {N_KEEP} corners")):
                    name = f"{set_name}_corners_{key}.png"
                    plot_corners(ims[0], c[key], label, out_path(PART, name), show=False, s=2 if key == "all" else 12)
                    SAVED.append(out_path(PART, name))
                name = f"{set_name}_descriptors.png"
                show_descriptors(c["gray"], r["pts1"], r["desc1"], n=8,
                                 save_as=out_path(PART, name), show=False)
                SAVED.append(out_path(PART, name))



def run_B4():
    part = "B4_mosaics"
    for set_name, stems, ref in SETS:
        ims = [load(s) for s in stems]
 
        auto_H, failed = [], []
        for k in range(len(ims) - 1):
            H, r = auto_homography(ims[k], ims[k + 1], N_MATCH, RATIO, RANSAC_THRESH, RANSAC_ITERS)
            auto_H.append(H)
            n, ni = len(r["m1"]), int(r["inliers"].sum())
            log(f"[{set_name}] {stems[k]} -> {stems[k + 1]}: {n} matches, {ni} RANSAC inliers "
                f"({100 * ni / max(n, 1):.0f}%)\nH_auto =\n{np.array2string(H, precision=5)}")
            if ni < MIN_INLIERS or not sane_homography(H, ims[k].shape):
                failed.append(k)
                log(f"  FAILED: {ni} inliers (need {MIN_INLIERS}) / homography unreasonable for "
                    f"{stems[k]} -> {stems[k + 1]}. Look at the matches figure for this pair.")
            name = f"{set_name}_ransac_{k}to{k + 1}.png"
            draw_matches_ransac(ims[k], ims[k + 1], r["m1"], r["m2"], r["inliers"],
                                save_as=out_path(part, name), show=False,
                                title=f"{stems[k]} -> {stems[k + 1]}")
            SAVED.append(out_path(part, name))
 
        if failed:
            log(f"[{set_name}] skipping the automatic mosaic: pair(s) {failed} did not give a reliable "
                f"homography. Try a larger N_MATCH, a higher RATIO / RANSAC_THRESH, or reshoot with more overlap.")
            continue
 
        manual_H, _ = pair_homographies(set_name, stems, ims)        
        for k, (Ha, Hm) in enumerate(zip(auto_H, manual_H)):         
            h, w = ims[k].shape[:2]
            corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float)
            gap = np.linalg.norm(apply_H(Ha, corners) - apply_H(Hm, corners), axis=1).mean()
            log(f"[{set_name}] pair {k}: auto vs manual homography differ by {gap:.1f}px at the image corners")
 
        try:
            auto_m = build_mosaic(ims, chain_homographies(auto_H, ref), blend="two_band")[0]
            manual_m = build_mosaic(ims, chain_homographies(manual_H, ref), blend="two_band")[0]
        except (ValueError, MemoryError) as e:
            log(f"[{set_name}] could not build mosaic: {e}")
            continue
        save_img(part, f"{set_name}_automatic.jpg", auto_m)
        save_img(part, f"{set_name}_manual.jpg", manual_m)
 
        fig, axs = plt.subplots(2, 1, figsize=(14, 12))
        axs[0].imshow(manual_m); axs[0].set_title(f"{set_name}: manual correspondences (Part A)")
        axs[1].imshow(auto_m); axs[1].set_title(f"{set_name}: automatic (Harris + ANMS + MOPS + RANSAC)")
        for a in axs:
            a.axis("off")
        plt.tight_layout()
        save_fig(part, f"{set_name}_manual_vs_automatic.png")
        log(f"[{set_name}] manual {manual_m.shape[1]}x{manual_m.shape[0]}, automatic {auto_m.shape[1]}x{auto_m.shape[0]}")

if __name__ == "__main__":
    what = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    os.makedirs(OUT_DIR, exist_ok=True)
    if what in ("all", "features"):
        run_B_features()
    if what in ("all", "ransac"):
        run_B4()
    print(f"\nSaved {len(SAVED)} files:")
    for p in SAVED:
        print("  ", p)