import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from harris import get_harris_corners, dist2
from main import apply_H, computeH


def to_gray(im):
    if im.ndim == 2:
        return im
    return (im @ np.array([0.299, 0.587, 0.114], dtype=np.float32)).astype(np.float32)


def anms(coords_yx, strengths, n_keep=500, c_robust=0.9, chunk=1000):
    pts = np.asarray(coords_yx, dtype=np.float64)
    strengths = np.asarray(strengths, dtype=np.float64)
    N = len(pts)
    r2 = np.full(N, np.inf)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        d2 = dist2(pts[s:e], pts)                                
        stronger = strengths[s:e, None] < c_robust * strengths[None, :]
        r2[s:e] = np.where(stronger, d2, np.inf).min(axis=1)

    order = np.lexsort((-strengths, -r2))[:n_keep]
    return order, r2


def detect_corners_xy(im, n_keep=500, c_robust=0.9):
    gray = to_gray(im)
    h, coords = get_harris_corners(gray)       
    ys, xs = coords
    strengths = h[ys, xs]
    yx = np.stack([ys, xs], axis=1)
    idx, _ = anms(yx, strengths, n_keep, c_robust)
    top = np.argsort(-strengths)[:n_keep]
    return {"all": yx[:, ::-1], "strongest": yx[top][:, ::-1], "anms": yx[idx][:, ::-1],
            "h": h, "gray": gray}


def plot_corners(im, pts_xy, title="", save_as=None, show=True, s=12):
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(im)
    ax.scatter(pts_xy[:, 0], pts_xy[:, 1], s=s, c="red", edgecolors="yellow", linewidths=0.5)
    ax.set_title(f"{title} ({len(pts_xy)} points)")
    ax.axis("off")
    if save_as:
        plt.savefig(save_as, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
        
def extract_descriptors(gray, pts_xy, window=40, spacing=5, sigma=2.0, eps=1e-8):
    blurred = gaussian_filter(gray.astype(np.float32), sigma)
    half = window // 2
    H, W = gray.shape
    descs, kept = [], []
    for x, y in np.asarray(pts_xy):
        x, y = int(round(x)), int(round(y))
        if x - half < 0 or y - half < 0 or x + half > W or y + half > H:
            continue
        patch = blurred[y - half:y + half:spacing, x - half:x + half:spacing]  
        d = patch.reshape(-1)
        d = (d - d.mean()) / (d.std() + eps)                                
        descs.append(d)
        kept.append((x, y))
    return np.array(descs, dtype=np.float32), np.array(kept)


def show_descriptors(gray, pts_xy, descs, n=8, seed=0, window=40, save_as=None, show=True):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(descs), size=min(n, len(descs)), replace=False)
    half = window // 2
    fig, axs = plt.subplots(2, len(idx), figsize=(2.2 * len(idx), 4.8))
    axs = np.asarray(axs).reshape(2, -1)
    for c, i in enumerate(idx):
        x, y = np.asarray(pts_xy[i]).astype(int)
        axs[0, c].imshow(gray[y - half:y + half, x - half:x + half], cmap="gray")
        axs[0, c].set_title(f"#{i} ({x},{y})", fontsize=8)
        axs[1, c].imshow(descs[i].reshape(8, 8), cmap="gray", interpolation="nearest")
        for r in (0, 1):
            axs[r, c].axis("off")
    axs[0, 0].text(-0.15, 0.5, "40x40 window", rotation=90, transform=axs[0, 0].transAxes,
                   va="center", ha="right", fontsize=9)
    axs[1, 0].text(-0.15, 0.5, "8x8 descriptor", rotation=90, transform=axs[1, 0].transAxes,
                   va="center", ha="right", fontsize=9)
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

def match_features(desc1, desc2, ratio=0.7, mutual=False, unique=True):
    d1 = np.asarray(desc1, dtype=np.float64)
    d2 = np.asarray(desc2, dtype=np.float64)
    if len(d2) < 2:
        return np.array([], int), np.array([], int), np.array([])
    D = np.maximum(dist2(d1, d2), 0)                  
    two = np.argpartition(D, 1, axis=1)[:, :2]    
    rows = np.arange(len(d1))
    first_is_a = D[rows, two[:, 0]] <= D[rows, two[:, 1]]
    nn1 = np.where(first_is_a, two[:, 0], two[:, 1])
    nn2 = np.where(first_is_a, two[:, 1], two[:, 0])
    e1, e2 = np.sqrt(D[rows, nn1]), np.sqrt(D[rows, nn2])
    ratios = e1 / (e2 + 1e-12)
    keep = ratios < ratio
    if mutual:
        back = np.argmin(D, axis=0)                     
        keep &= back[nn1] == rows
    if unique and keep.any():
        cand = np.flatnonzero(keep)
        best_first = cand[np.argsort(ratios[cand])]          
        _, first = np.unique(nn1[best_first], return_index=True)
        keep = np.zeros_like(keep)
        keep[best_first[first]] = True
    return rows[keep], nn1[keep], ratios[keep]


def draw_matches(im1, im2, pts1_xy, pts2_xy, idx1, idx2, max_lines=None, save_as=None,
                 show=True, title=""):
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2) + im1.shape[2:], dtype=im1.dtype)
    canvas[:h1, :w1] = im1
    canvas[:h2, w1:] = im2
    p1 = np.asarray(pts1_xy)[idx1]
    p2 = np.asarray(pts2_xy)[idx2] + np.array([w1, 0])
    if max_lines is not None and len(p1) > max_lines:
        sel = np.random.default_rng(0).choice(len(p1), max_lines, replace=False)
        p1, p2 = p1[sel], p2[sel]
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.imshow(canvas)
    colors = plt.cm.hsv(np.linspace(0, 1, max(len(p1), 1), endpoint=False))
    for (x1, y1), (x2, y2), c in zip(p1, p2, colors):
        ax.plot([x1, x2], [y1, y2], "-", color=c, lw=1)
        ax.plot([x1, x2], [y1, y2], "o", color=c, ms=4)
    ax.set_title(f"{title} ({len(idx1)} matches)".strip())
    ax.axis("off")
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

def match_pair(im1, im2, n_keep=500, ratio=0.7, mutual=False):
    c1, c2 = detect_corners_xy(im1, n_keep), detect_corners_xy(im2, n_keep)
    d1, p1 = extract_descriptors(c1["gray"], c1["anms"])
    d2, p2 = extract_descriptors(c2["gray"], c2["anms"])
    i1, i2, ratios = match_features(d1, d2, ratio, mutual)
    return {"m1": p1[i1], "m2": p2[i2], "pts1": p1, "pts2": p2, "desc1": d1, "desc2": d2,
            "idx1": i1, "idx2": i2, "ratios": ratios, "corners1": c1, "corners2": c2}

def _inlier_mask(H, p1, p2, thresh):
    with np.errstate(all="ignore"):
        err = np.linalg.norm(apply_H(H, p1) - p2, axis=1)
    return np.isfinite(err) & (err < thresh)


def _bad_sample(q, min_tri=0.05):

    for i, j, k in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
        a, b, c = q[i], q[j], q[k]
        cross = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        longest2 = max(((a - b) ** 2).sum(), ((a - c) ** 2).sum(), ((b - c) ** 2).sum())
        if longest2 < 1e-9 or cross / longest2 < min_tri:
            return True
    return False


def _degenerate_H(H):
    det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
    return (not np.all(np.isfinite(H))) or not (0.02 < abs(det) < 50)


def ransac_homography(p1, p2, n_iters=5000, thresh=3.0, seed=0):
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    N = len(p1)
    if N < 4:
        raise ValueError(f"need at least 4 matches for a homography, only {N}.")
    rng = np.random.default_rng(seed)
    best = np.zeros(N, dtype=bool)
    for _ in range(n_iters):
        idx = rng.choice(N, 4, replace=False)
        if _bad_sample(p1[idx]) or _bad_sample(p2[idx]):
            continue
        H = computeH(p1[idx], p2[idx])
        if _degenerate_H(H):
            continue
        inl = _inlier_mask(H, p1, p2, thresh)
        if inl.sum() > best.sum():
            best = inl
    if best.sum() < 4:
        raise RuntimeError("no consistent homography. check matches / ratio.")
    for _ in range(2):
        H = computeH(p1[best], p2[best])
        refit = _inlier_mask(H, p1, p2, thresh)
        if refit.sum() < 4 or _degenerate_H(H):
            break
        best = refit
    return computeH(p1[best], p2[best]), best


def auto_homography(im1, im2, n_keep=500, ratio=0.7, thresh=3.0, n_iters=5000, seed=0):
    r = match_pair(im1, im2, n_keep, ratio)
    H, inl = ransac_homography(r["m1"], r["m2"], n_iters, thresh, seed)
    r["H"], r["inliers"] = H, inl
    return H, r


def draw_matches_ransac(im1, im2, m1, m2, inliers, save_as=None, show=True, title=""):
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2) + im1.shape[2:], dtype=im1.dtype)
    canvas[:h1, :w1] = im1
    canvas[:h2, w1:] = im2
    m1, m2, inliers = np.asarray(m1), np.asarray(m2) + np.array([w1, 0]), np.asarray(inliers)
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.imshow(canvas)
    for (x1, y1), (x2, y2) in zip(m1[~inliers], m2[~inliers]):
        ax.plot([x1, x2], [y1, y2], "-", color="red", lw=0.8, alpha=0.7)
    for (x1, y1), (x2, y2) in zip(m1[inliers], m2[inliers]):
        ax.plot([x1, x2], [y1, y2], "-", color="lime", lw=1)
        ax.plot([x1, x2], [y1, y2], "o", color="lime", ms=3)
    ax.set_title(f"{title} - {inliers.sum()} inliers (green), {(~inliers).sum()} outliers (red)".strip(" -"))
    ax.axis("off")
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)