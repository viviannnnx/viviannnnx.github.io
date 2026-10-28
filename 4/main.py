import json
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from scipy.ndimage import distance_transform_edt, gaussian_filter

#pt1
def load_image(path, max_width=1200):
    im = plt.imread(path)
    im = im.astype(np.float32) / 255.0 if im.dtype == np.uint8 else im.astype(np.float32)
    if im.ndim == 2:                       
        im = np.stack([im] * 3, axis=-1)
    im = im[..., :3]                

    h, w = im.shape[:2]
    if w > max_width:
        s = max_width / w
        new_w, new_h = max_width, int(h * s)
        im = gaussian_filter(im, sigma=(0.5 / s, 0.5 / s, 0))   # anti-alias before shrinking
        im, _, _ = warpImageBilinear(im, np.diag([s, s, 1.0]), bbox=(0, 0, new_w - 1, new_h - 1))
    return np.clip(im, 0, 1)


def pick_points(im, n, title="Click points"):
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(im)
    ax.set_title(title)
    pts = np.array(plt.ginput(n, timeout=0))
    plt.close(fig)
    return pts


def save_points(path, pts):
    with open(path, "w") as f:
        json.dump(np.asarray(pts).tolist(), f)


def load_points(path):
    with open(path) as f:
        return np.array(json.load(f), dtype=float)


def show_correspondences(im1, im2, p1, p2, save_as=None, show=True):
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    for ax, im, p in zip(axs, (im1, im2), (p1, p2)):
        ax.imshow(im)
        ax.scatter(p[:, 0], p[:, 1], c="r", s=30)
        for i, (x, y) in enumerate(p):
            ax.text(x + 6, y - 6, str(i), color="yellow", fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()

#pt2
def build_system(im1_pts, im2_pts):
    p = np.asarray(im1_pts, float)
    q = np.asarray(im2_pts, float)
    n = len(p)
    x, y = p[:, 0], p[:, 1]
    xp, yp = q[:, 0], q[:, 1]
    A = np.zeros((2 * n, 8))
    b = np.zeros(2 * n)
    A[0::2, 0], A[0::2, 1], A[0::2, 2] = x, y, 1
    A[0::2, 6], A[0::2, 7] = -x * xp, -y * xp
    A[1::2, 3], A[1::2, 4], A[1::2, 5] = x, y, 1
    A[1::2, 6], A[1::2, 7] = -x * yp, -y * yp
    b[0::2], b[1::2] = xp, yp
    return A, b


def computeH(im1_pts, im2_pts):
    A, b = build_system(im1_pts, im2_pts)
    h, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.append(h, 1.0).reshape(3, 3)


def apply_H(H, pts):
    pts = np.asarray(pts, float)
    ph = np.c_[pts, np.ones(len(pts))] @ H.T
    return ph[:, :2] / ph[:, 2:3]

#pt3
def compute_bbox(shape, H):
    h, w = shape[:2]
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float)
    wdiv = np.c_[corners, np.ones(4)] @ H.T
    c = wdiv[:, :2] / wdiv[:, 2:3]
    xmin, ymin = np.floor(c.min(0)).astype(int)
    xmax, ymax = np.ceil(c.max(0)).astype(int)
    return int(xmin), int(ymin), int(xmax), int(ymax)


def _warp(im, H, bbox, method):
    if im.ndim == 2:
        im = im[..., None]
    h, w, ch = im.shape
    if bbox is None:
        bbox = compute_bbox(im.shape, H)
    xmin, ymin, xmax, ymax = bbox
    out_w, out_h = xmax - xmin + 1, ymax - ymin + 1

    xc, yc = np.meshgrid(np.arange(xmin, xmax + 1), np.arange(ymin, ymax + 1))
    pts = np.stack([xc.ravel(), yc.ravel(), np.ones(xc.size)])
    src = np.linalg.inv(H) @ pts
    wd = src[2]
    ok = np.abs(wd) > 1e-12
    wd = np.where(ok, wd, 1.0)
    xs, ys = src[0] / wd, src[1] / wd
    valid = ok & (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)

    out = np.zeros((xc.size, ch), dtype=np.float32)
    xv, yv = xs[valid], ys[valid]
    if method == "nearest":
        xi = np.rint(xv).astype(int)
        yi = np.rint(yv).astype(int)
        out[valid] = im[yi, xi]
    elif method == "bilinear":
        x0 = np.clip(np.floor(xv).astype(int), 0, w - 2)
        y0 = np.clip(np.floor(yv).astype(int), 0, h - 2)
        dx = (xv - x0)[:, None]
        dy = (yv - y0)[:, None]
        out[valid] = ((1 - dx) * (1 - dy) * im[y0, x0] + dx * (1 - dy) * im[y0, x0 + 1]
                      + (1 - dx) * dy * im[y0 + 1, x0] + dx * dy * im[y0 + 1, x0 + 1])
    else:
        raise ValueError(method)

    out = out.reshape(out_h, out_w, ch)
    mask = valid.reshape(out_h, out_w)
    if ch == 1:
        out = out[..., 0]
    return out, mask, (xmin, ymin)


def warpImageNearestNeighbor(im, H, bbox=None):
    return _warp(im, H, bbox, "nearest")


def warpImageBilinear(im, H, bbox=None):
    return _warp(im, H, bbox, "bilinear")


def rectify(im, corners, out_w, out_h, method="bilinear"):
    target = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], float)
    H = computeH(corners, target)
    fn = warpImageBilinear if method == "bilinear" else warpImageNearestNeighbor
    warped, mask, _ = fn(im, H, bbox=(0, 0, out_w - 1, out_h - 1))
    return warped, H

#pt4
def _dist_weight(mask):
    return distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1].astype(np.float32)


def blend_feather(ims, masks):
    """Weighted average with distance-transform weights."""
    ws = [_dist_weight(m) for m in masks]
    num = sum(w[..., None] * im for w, im in zip(ws, ims))
    den = sum(ws)[..., None]
    return np.where(den > 0, num / np.maximum(den, 1e-8), 0).astype(np.float32)


def blend_two_band(ims, masks, sigma=8):
    ws = [_dist_weight(m) for m in masks]
    lows, highs = [], []
    for im, m in zip(ims, masks):
        mf = m.astype(np.float32)
        num = gaussian_filter(im * mf[..., None], sigma=(sigma, sigma, 0))
        den = gaussian_filter(mf, sigma)[..., None]
        low = num / np.maximum(den, 1e-6)         
        lows.append(low * mf[..., None])
        highs.append((im - low) * mf[..., None])
    low_blend = blend_feather(lows, masks)
    idx = np.argmax(np.stack(ws), axis=0)
    high = np.take_along_axis(np.stack(highs), idx[None, ..., None], axis=0)[0]
    return np.clip(low_blend + high, 0, 1).astype(np.float32)


def build_mosaic(ims, Hs, method="bilinear", blend="two_band", sigma=8):
    """ims: list of images. Hs: list of 3x3 matrices mapping each image INTO the
    reference frame (use np.eye(3) for the reference image itself).
    For chains, multiply:  H_left_to_ref = H_mid_to_ref @ H_left_to_mid."""
    boxes = [compute_bbox(im.shape, H) for im, H in zip(ims, Hs)]
    bbox = (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))
    warp = warpImageBilinear if method == "bilinear" else warpImageNearestNeighbor
    warped, masks = [], []
    for im, H in zip(ims, Hs):
        w, m, _ = warp(im, H, bbox=bbox)
        warped.append(w)
        masks.append(m)
    if blend == "feather":
        return blend_feather(warped, masks), warped, masks
    return blend_two_band(warped, masks, sigma), warped, masks


BASE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE, "images") 
OUT_DIR = "results"
PTS_DIR = "points"
MAX_WIDTH = 1200     
N_POINTS = 12        
SETS = [
    ("mosaic1", ["image1.2", "image1.3", "image1.4"], 1),
    ("mosaic2", ["image2.1", "image2.2", "image2.3"], 1),
    ("mosaic3", ["image3.1", "image3.2", "image3.3"], 1),
]


RECTIFY = [
    {"name": "rectify1", "image": "image1.4", "size": (150, 500)},
    {"name": "rectify2", "image": "extrarectify2", "size": (300, 500)},
    {"name": "rectify3", "image": "extrarectify", "size": (500, 200)},
]

EXTS = ["", ".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
SAVED = []
os.makedirs(PTS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def find_image(stem):
    for ext in EXTS:
        p = os.path.join(IMG_DIR, stem + ext)
        if os.path.isfile(p):
            return p
def find_image(stem):
    for ext in EXTS:
        p = os.path.join(IMG_DIR, stem + ext)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"Couldn't find '{stem}' in {os.path.abspath(IMG_DIR)}\n"
        f"Folder contents: {os.listdir(IMG_DIR) if os.path.isdir(IMG_DIR) else 'FOLDER DOES NOT EXIST'}")

def load(stem):
    return load_image(find_image(stem), MAX_WIDTH)

def out_path(part, name):
    folder = os.path.join(OUT_DIR, part)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, name)


def log(msg):
    print(msg)
    with open(os.path.join(OUT_DIR, "log.txt"), "a") as f:
        f.write(msg + "\n")


def save_img(part, name, im):
    path = out_path(part, name)
    kw = {"pil_kwargs": {"quality": 95}} if name.lower().endswith((".jpg", ".jpeg")) else {}
    plt.imsave(path, (np.clip(im, 0, 1) * 255 + 0.5).astype(np.uint8), **kw)
    SAVED.append(path)


def save_fig(part, name):
    path = out_path(part, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    SAVED.append(path)


def save_text(part, name, text):
    path = out_path(part, name)
    with open(path, "w") as f:
        f.write(text)
    SAVED.append(path)


def montage(images, labels, part, name):
    """Images in a row, each with a title (A.1 source photos)."""
    fig, axs = plt.subplots(1, len(images), figsize=(5.5 * len(images), 4.8))
    for a, im, lab in zip(np.atleast_1d(axs), images, labels):
        a.imshow(im)
        a.set_title(lab)
        a.axis("off")
    save_fig(part, name)


# ---------------------------- point picking ---------------------------------
def get_corners(name, im):
    path = os.path.join(PTS_DIR, f"{name}_corners.json")
    if os.path.exists(path):
        log(f"[{name}] using cached corners {path}")
        return load_points(path)
    print(f"\n[{name}] Click the 4 corners in order: top-left, top-right, bottom-right, bottom-left")
    pts = pick_points(im, 4, f"{name}: click TL, TR, BR, BL")
    if len(pts) != 4:
        raise RuntimeError("Need exactly 4 clicks.")
    save_points(path, pts)
    return pts


def pick_correspondences(imA, imB, n, title):
    hA, wA = imA.shape[:2]
    hB, wB = imB.shape[:2]
    canvas = np.zeros((max(hA, hB), wA + wB, 3), np.float32)
    canvas[:hA, :wA] = imA
    canvas[:hB, wA:] = imB
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.imshow(canvas)
    ax.axis("off")
    clicks = np.array(plt.ginput(2 * n, timeout=0, show_clicks=True))
    plt.close(fig)
    m = len(clicks) // 2
    A = clicks[0:2 * m:2]
    B = clicks[1:2 * m:2] - np.array([wA, 0])
    if m < 6:
        print(f"WARNING: only {m} pairs clicked; 6+ recommended.")
    if (A[:, 0] >= wA).any() or (B[:, 0] < 0).any():
        raise RuntimeError("Clicks not alternated")
    return A, B


def get_pairs(name, imA, imB, tag, desc):
    path = os.path.join(PTS_DIR, f"{name}_{tag}.json")
    if os.path.exists(path):
        log(f"[{name}] using cached correspondences {path}")
        d = json.load(open(path))
        return np.array(d["src"]), np.array(d["ref"])
    print(f"\n[{name}] correspondences {desc}")
    src, dst = pick_correspondences(imA, imB, N_POINTS, f"{name}: {desc}")
    json.dump({"src": src.tolist(), "ref": dst.tolist()}, open(path, "w"))
    return src, dst


def pair_homographies(set_name, stems, ims):
    Hs, pairs = [], []
    for k in range(len(ims) - 1):
        tag = f"{k}to{k + 1}"
        src, dst = get_pairs(set_name, ims[k], ims[k + 1], tag, f"{stems[k]} -> {stems[k + 1]}")
        Hs.append(computeH(src, dst))
        pairs.append((src, dst, tag))
    return Hs, pairs


def chain_homographies(pair_H, reference):
    to_ref = [None] * (len(pair_H) + 1)
    to_ref[reference] = np.eye(3)
    for k in range(reference - 1, -1, -1):
        to_ref[k] = to_ref[k + 1] @ pair_H[k]
    for k in range(reference + 1, len(to_ref)):
        to_ref[k] = to_ref[k - 1] @ np.linalg.inv(pair_H[k - 1])
    return [H / H[2, 2] for H in to_ref]


def reproj_error(H, src, dst):
    return np.linalg.norm(apply_H(H, src) - dst, axis=1)

def run_A1():
    for set_name, stems, ref in SETS:
        ims = [load(s) for s in stems]
        labels = [s + (" (ref)" if k == ref else "") for k, s in enumerate(stems)]
        montage(ims, labels, "A1_sources", f"{set_name}_sources.png")


def run_A2():
    part = "A2_homographies"
    for set_name, stems, _ in SETS:
        ims = [load(s) for s in stems]
        Hs, pairs = pair_homographies(set_name, stems, ims)
        for k, (H, (src, dst, tag)) in enumerate(zip(Hs, pairs)):
            err = reproj_error(H, src, dst)
            log(f"[{set_name}] {stems[k]} -> {stems[k + 1]}: {len(src)} points, reprojection error "
                f"mean {err.mean():.2f}px, max {err.max():.2f}px\nH =\n{np.array2string(H, precision=5)}")

            show_correspondences(ims[k], ims[k + 1], src, dst, show=False,
                                 save_as=out_path(part, f"{set_name}_corr_{tag}.png"))
            SAVED.append(out_path(part, f"{set_name}_corr_{tag}.png"))

            A, b = build_system(src, dst)
            lines = [f"System A h = b for {set_name} {stems[k]} -> {stems[k + 1]}",
                     "h = [h11 h12 h13 h21 h22 h23 h31 h32], h33 = 1", "A (2n x 8) | b (2n)"]
            lines += [" ".join(f"{v:12.2f}" for v in row) + f" | {v:10.2f}" for row, v in zip(A, b)]
            lines += ["", "H =", np.array2string(H, precision=6, suppress_small=True)]
            save_text(part, f"{set_name}_system_{tag}.txt", "\n".join(lines) + "\n")


def best_time(fn, reps=3):
    best, out = 1e9, None
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return out, best


def run_A3():
    part = "A3_rectification"
    for job in RECTIFY:
        name, (w, h) = job["name"], job["size"]
        im = load(job["image"])
        corners = get_corners(name, im)

        (nn, _), t_nn = best_time(lambda: rectify(im, corners, w, h, "nearest"))
        (bl, H), t_bl = best_time(lambda: rectify(im, corners, w, h, "bilinear"))
        log(f"[{name}] output {w}x{h}, nearest {t_nn * 1000:.1f} ms, bilinear {t_bl * 1000:.1f} ms "
            f"({t_bl / t_nn:.1f}x)\nH =\n{np.array2string(H, precision=5)}")

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.imshow(im)
        closed = np.vstack([corners, corners[:1]])
        ax.plot(closed[:, 0], closed[:, 1], "r-o", lw=2)
        for lab, (x, y) in zip(["TL", "TR", "BR", "BL"], corners):
            ax.text(x + 8, y - 8, lab, color="yellow", fontsize=12)
        ax.axis("off")
        save_fig(part, f"rect_{name}_input.png")

        save_img(part, f"rect_{name}_nearest.jpg", nn)
        save_img(part, f"rect_{name}_bilinear.jpg", bl)

        cx, cy = job.get("zoom", (w // 2, h // 2))
        x0 = int(np.clip(cx - 30, 0, max(w - 60, 0)))
        y0 = int(np.clip(cy - 30, 0, max(h - 60, 0)))
        fig, axs = plt.subplots(1, 2, figsize=(10, 5.2))
        for a, img, lab in zip(axs, (nn, bl), (f"Nearest neighbor ({t_nn * 1000:.0f} ms)",
                                               f"Bilinear ({t_bl * 1000:.0f} ms)")):
            a.imshow(img[y0:y0 + 60, x0:x0 + 60], interpolation="nearest")
            a.set_title(lab)
            a.axis("off")
        save_fig(part, f"rect_{name}_zoom.png")


def run_A4():
    part = "A4_mosaics"
    for set_name, stems, ref in SETS:
        ims = [load(s) for s in stems]
        Hs, _ = pair_homographies(set_name, stems, ims)
        to_ref = chain_homographies(Hs, ref)

        boxes = np.array([compute_bbox(im.shape, H) for im, H in zip(ims, to_ref)])
        cw, ch = boxes[:, 2].max() - boxes[:, 0].min() + 1, boxes[:, 3].max() - boxes[:, 1].min() + 1
        log(f"[{set_name}] canvas {cw}x{ch}")
        if cw > 12000 or ch > 12000:
            raise ValueError(f"{set_name}: canvas is far too large. Use fewer photos or check the clicks.")

        t = time.perf_counter()
        feather, warped, masks = build_mosaic(ims, to_ref, blend="feather")
        two_band, _, _ = build_mosaic(ims, to_ref, blend="two_band")
        log(f"[{set_name}] warp + both blends: {time.perf_counter() - t:.1f}s")

        naive = np.zeros_like(feather)
        for w_, m_ in zip(warped, masks):      
            naive[m_] = w_[m_]

        save_img(part, f"{set_name}_feather.jpg", feather)
        save_img(part, f"{set_name}_two_band.jpg", two_band)
        save_img(part, f"{set_name}_naive_overwrite.jpg", naive)
        for k, w_ in enumerate(warped):
            save_img(part, f"{set_name}_warped_{k}.jpg", w_)
        montage(ims, [s + (" (ref)" if k == ref else "") for k, s in enumerate(stems)],
                part, f"{set_name}_sources.png")



if __name__ == "__main__":
    # run_A1()
    # run_A2()
    run_A3()
    # run_A4()