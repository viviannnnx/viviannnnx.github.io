# CS194-26 (CS294-26): Project 1 starter Python code

# these are just some suggested libraries
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# name of the input file
imname = './1/CS180_fa2026_proj1_data/tobolsk.jpg'

# read in the image as grayscale (the glass plate scan is stacked grayscale)
im = cv.imread(imname, cv.IMREAD_GRAYSCALE)

# convert to float in [0,1] (might want to do this later on to save memory)
im = im.astype(np.float32) / 255.0
    
# compute the height of each part (just 1/3 of total)
height = int(np.floor(im.shape[0] / 3.0))

# separate color channels
b = im[:height]
g = im[height: 2*height]
r = im[2*height: 3*height]

def l2_score(im1, im2):
    return np.sqrt(np.sum((im1 - im2) ** 2))

def ncc_score(im1, im2):
    v1 = (im1 - im1.mean()).flatten()
    v2 = (im2 - im2.mean()).flatten()
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def crop_border(im, margin=0.1):
    h, w = im.shape[:2]
    dy, dx = int(h * margin), int(w * margin)
    return im[dy:h-dy, dx:w-dx]

def shift_im(im, dx, dy):
    return np.roll(np.roll(im, dy, axis=0), dx, axis=1)

def align(base, moving, window=15, metric='ncc'):
    best_score = 0
    best_offset = (0, 0)
    for dy in range(-window, window + 1):
        for dx in range(-window, window + 1):
            shifted = shift_im(moving, dx, dy)
            score = (ncc_score(crop_border(base), crop_border(shifted)) if metric == 'ncc'
                     else l2_score(crop_border(base), crop_border(shifted)))
            is_better = (score > best_score) if metric == 'ncc' else (score < best_score)
            if best_score == 0 or is_better:
                best_score = score
                best_offset = (dx, dy)
    return best_offset

def resize_half(im):
    return cv.resize(im, (im.shape[1] // 2, im.shape[0] // 2), interpolation=cv.INTER_AREA)

def pyramid_align(base, moving, metric, window=5, min_size=64):
    if min(base.shape[:2]) <= min_size:
        return align(base, moving, window=15, metric=metric)  

    small_base = resize_half(base)
    small_moving = resize_half(moving)
    dx_coarse, dy_coarse = pyramid_align(small_base, small_moving, metric=metric, window=window, min_size=min_size)

    dx_guess, dy_guess = dx_coarse * 2, dy_coarse * 2

    pre_shifted = shift_im(moving, dx_guess, dy_guess)
    dx_refine, dy_refine = align(base, pre_shifted, window=window, metric=metric)

    return dx_guess + dx_refine, dy_guess + dy_refine


# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)
gb_dx, gb_dy = pyramid_align(b,g,'ncc')
rb_dx, rb_dy = pyramid_align(b,r,'ncc')

ag = shift_im(g, gb_dx, gb_dy)
ar = shift_im(r, rb_dx, rb_dy)

# placeholders so the script runs before implementing align()
# ag = g
# ar = r
# create a color image
im_out = np.dstack([ar, ag, b])

# display the image using matplotlib (expects RGB)
plt.figure(figsize=(8, 8))
plt.imshow(im_out)
plt.title('Colorized')
plt.axis('off')
plt.show()

# prepare for OpenCV saving/display (expects BGR uint8)
out_uint8 = np.clip(im_out * 255.0, 0, 255).astype(np.uint8)
out_bgr = cv.cvtColor(out_uint8, cv.COLOR_RGB2BGR)

# save the image
fname = './1/images/multi/tobolsk_ncc.jpg'
cv.imwrite(fname, out_bgr)
print(f"G offset: ({gb_dx}, {gb_dy})")
print(f"R offset: ({rb_dx}, {rb_dy})")


