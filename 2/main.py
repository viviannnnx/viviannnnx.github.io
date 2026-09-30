import numpy as np
import scipy.signal as signal
import cv2
import matplotlib.pyplot as plt
import time

def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float64) / 255.0   

def load(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float64) / 255.0   

def pad_zero(image, kernel, mode="same"):
    kh, kw = kernel.shape
    if mode == "same":
        ph, pw = kh // 2, kw // 2
    elif mode == "full":
        ph, pw = kh - 1, kw - 1
    return np.pad(image, ((ph, ph), (pw, pw)), mode='constant', constant_values=0)

def conv2d_2loops(image, kernel, mode="same"):
    kernel = kernel[::-1, ::-1]
    kh, kw = kernel.shape
    padded = pad_zero(image, kernel, mode)
    H_out = padded.shape[0] - kh + 1
    W_out = padded.shape[1] - kw + 1
    out = np.zeros((H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            out[i, j] = np.sum(padded[i:i + kh, j:j + kw] * kernel)
    return out

def conv2d_4loops(image, kernel, mode="same"):
    kernel = kernel[::-1, ::-1]
    kh, kw = kernel.shape
    padded = pad_zero(image, kernel, mode)
    H_out = padded.shape[0] - kh + 1
    W_out = padded.shape[1] - kw + 1
    out = np.zeros((H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            acc = 0.0
            for u in range(kh):
                for v in range(kw):
                    acc += padded[i + u, j + v] * kernel[u, v]
            out[i, j] = acc
    return out

def save_image(img, title, filename, diverging=False):
    plt.figure()
    if diverging:
        plt.imshow(img, cmap='gray', vmin=-0.5, vmax=0.5)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.savefig(f"2/results/{filename}", bbox_inches='tight', dpi=150)
    plt.show()
    

# Dx = np.array([[1, 0, -1]]) 
# Dy = Dx.T     
# box = np.full((9, 9), 1/81)

# #PART 1.1
# selfie = load_gray("2/images/selfie.jpg")
# selfie_blurred = conv2d_2loops(selfie, box)
# selfie_dx = conv2d_2loops(selfie, Dx)
# selfie_dy = conv2d_2loops(selfie, Dy)

# save_image(selfie_blurred, "9x9 Box Blur", "selfie_box.png")
# save_image(selfie_dx, "Dx", "selfie_dx.png", diverging=True)
# save_image(selfie_dy, "Dy", "selfie_dy.png", diverging=True)

# #time test
# test_img = selfie  
# test_kernel = box     

# t0 = time.time()
# out_4loop = conv2d_4loops(test_img, test_kernel)
# t1 = time.time()
# print(f"4-loop (on {test_img.shape}): {t1 - t0:.4f} sec")

# t0 = time.time()
# out_2loop = conv2d_2loops(test_img, test_kernel)
# t1 = time.time()
# print(f"2-loop (on {test_img.shape}): {t1 - t0:.4f} sec")

# t0 = time.time()
# out_scipy = signal.convolve2d(test_img, test_kernel, mode='same', boundary='fill', fillvalue=0)
# t1 = time.time()
# print(f"scipy convolve2d (on {test_img.shape}): {t1 - t0:.4f} sec")
# print(np.max(np.abs(out_2loop - out_scipy)))
# # 4-loop (on (1280, 1099)): 24.7312 sec
# # 2-loop (on (1280, 1099)): 3.5027 sec
# # scipy convolve2d (on (1280, 1099)): 0.1219 sec


# #PART 1.2
# cameraman = load_gray("2/images/cameraman.png")
# cameraman_dx = signal.convolve2d(cameraman, Dx, mode='same', boundary='fill', fillvalue=0)
# cameraman_dy = signal.convolve2d(cameraman, Dy, mode='same', boundary='fill', fillvalue=0)

# save_image(cameraman_dx, "Cameraman dx", "cam_dx.png", diverging=True)
# save_image(cameraman_dy, "Cameraman dy", "cam_dy.png", diverging=True)

# grad_mag = np.sqrt(cameraman_dx**2 + cameraman_dy**2)
# save_image(grad_mag, "Gradient Magnitude", "cam_gradmag.png")

# for t in [0.1, 0.2, 0.3, 0.4]:
#     edges_t = (grad_mag > t).astype(float)
#     save_image(edges_t, f"threshold={t}", f"cam_edges_{t}.png")

# #from here concluded that best t between 0.2 and 0.3 
# for t in [0.22, 0.24, 0.26, 0.28]:
#     edges_t = (grad_mag > t).astype(float)
#     save_image(edges_t, f"threshold={t}", f"cam_edges_{t}.png")

# # # PART 1.3
def gaussian_kernel(ksize, sigma):
	g1d = cv2.getGaussianKernel(ksize, sigma)
	return g1d @ g1d.T
# gaussian = gaussian_kernel(9, 2)

# cam_blurred = signal.convolve2d(cameraman, gaussian, mode='same', boundary='fill', fillvalue=0)

# cam_blur_dx = signal.convolve2d(cam_blurred, Dx, mode='same', boundary='fill', fillvalue=0)
# cam_blur_dy = signal.convolve2d(cam_blurred, Dy, mode='same', boundary='fill', fillvalue=0)
# blur_grad_mag = np.sqrt(cam_blur_dx**2 + cam_blur_dy**2)

# save_image(cam_blurred, "Blurred Cameraman", "cam_blurred.png")
# save_image(cam_blur_dx, "Blurred dx", "cam_blur_dx.png", diverging=True)
# save_image(cam_blur_dy, "Blurred dy", "cam_blur_dy.png", diverging=True)
# save_image(blur_grad_mag, "Blurred Gradient Magnitude", "cam_blur_gradmag.png")

# for t in [0.1, 0.2, 0.3, 0.4]:
#     blur_edges = (blur_grad_mag > t).astype(float)
#     save_image(blur_edges, f"threshold={t}", f"cam_blur_edges_{t}.png")

# for t in [0.12, 0.14, 0.16, 0.18]:
#     blur_edges = (blur_grad_mag > t).astype(float)
#     save_image(blur_edges, f"threshold={t}", f"cam_blur_edges_{t}.png")

# DoG_dx = signal.convolve2d(gaussian, Dx, mode='full')
# DoG_dy = signal.convolve2d(gaussian, Dy, mode='full')
# save_image(DoG_dx, "DoG_dx", "dog_dx.png")
# save_image(DoG_dy, "DoG_dy", "dog_dy.png")


# cam_dog_dx = signal.convolve2d(cameraman, DoG_dx, mode='same', boundary='fill', fillvalue=0)
# cam_dog_dy = signal.convolve2d(cameraman, DoG_dy, mode='same', boundary='fill', fillvalue=0)
# gradient_dog = np.sqrt(cam_dog_dx**2 + cam_dog_dy**2)

# margin = 10
# a = blur_grad_mag[margin:-margin, margin:-margin]  
# b = gradient_dog[margin:-margin, margin:-margin]

# save_image(cam_dog_dx, "Cameraman via DoG_dx", "cam_dog_dx.png", diverging=True)
# save_image(cam_dog_dy, "Cameraman via DoG_dy", "cam_dog_dy.png", diverging=True)
# save_image(gradient_dog, "Gradient DoG", "gradient_dog.png")

# print(np.allclose(a, b, atol=1e-6))
# print("Max interior error:", np.max(np.abs(a - b)))
# # True
# # Max interior error: 7.467984564080155e-16

# # #Part 2
def convolve2d_color(img, kernel, mode='same', boundary='fill', fillvalue=0):
    channels = []
    for c in range(img.shape[2]):
        out = signal.convolve2d(img[:, :, c], kernel, mode=mode, boundary=boundary, fillvalue=fillvalue)
        channels.append(out)
    return np.stack(channels, axis=2)

# def unsharp_mask(img, ksize=9, sigma=2, alpha=1.0):
#     gaussian = gaussian_kernel(ksize, sigma)
#     blurred = convolve2d_color(img, gaussian)
#     high_freq = img - blurred
#     sharpened = img + alpha * high_freq
#     return np.clip(sharpened, 0, 1), blurred, high_freq

# # Part 2.1
# taj = load("2/images/taj.jpg")   

# taj_sharp, taj_blur, taj_high = unsharp_mask(taj, ksize=9, sigma=2, alpha=1.0)

# save_image(taj, "Taj Original", "taj_orig.png")
# save_image(taj_blur, "Taj Blurred", "taj_blur.png")
# save_image(taj_high, "Taj High Frequency", "taj_high.png", diverging=True)

# for alpha in [0.5, 1.0, 2.0]:
#     sharp, _, _ = unsharp_mask(taj, ksize=9, sigma=2, alpha=alpha)
#     save_image(sharp, f"alpha={alpha}", f"taj_sharp_a{alpha}.png")

# flowers = load("2/images/flowers.jpg")

# flowers_sharp, flowers_blur, flowers_high = unsharp_mask(flowers, ksize=9, sigma=2, alpha=1.0)

# save_image(flowers, "Flowers Original (Blur)", "blur_flowers.png")
# save_image(flowers_blur, "Flowers Blur (Blurrer)", "flowers_blur.png")
# save_image(flowers_high, "Flowers High Frequency", "high_flowers.png", diverging=True)

# for alpha in [0.5, 1.0, 2.0]:
#     sharp, _, _ = unsharp_mask(flowers, ksize=9, sigma=2, alpha=alpha)
#     save_image(sharp, f"alpha={alpha}", f"flowers_sharp_a{alpha}.png")

# sharp_img = load("2/images/cat.jpg")
# blurred_version = convolve2d_color(sharp_img, gaussian_kernel(9, 2), mode='same', boundary='fill', fillvalue=0)
# resharpened, _, _ = unsharp_mask(blurred_version, ksize=9, sigma=2, alpha=1.0)

# save_image(sharp_img, "Original Sharp", "resharp_orig.png")
# save_image(blurred_version, "Blurred", "resharp_blurred.png")
# save_image(resharpened, "Re-sharpened", "resharp_result.png")

# for alpha in [0.5, 1.0, 2.0]:
#     sharp, _, _ = unsharp_mask(blurred_version, ksize=9, sigma=2, alpha=alpha)
#     save_image(sharp, f"alpha={alpha}", f"resharp_sharp_a{alpha}.png")

# # Part 2.2
# # from hybrid starter
# import matplotlib.pyplot as plt
# from align_image_code import align_images

# # First load images

# # high sf
# im1 = plt.imread('2/images/DerekPicture.jpg') / 255.

# # low sf
# im2 = plt.imread('2/images/nutmeg.jpg') / 255.

# # Next align images (this code is provided, but may be improved)
# im1_aligned, im2_aligned = align_images(im1, im2)

# # You will provide the code below. Sigma1 and sigma2 are arbitrary 
# # cutoff values for the high and low frequencies
# def low_pass(img, sigma, ksize=None):
#     if ksize is None:
#         ksize = int(6 * sigma + 1) | 1   # ensure odd
#     kernel = gaussian_kernel(ksize, sigma)
#     return convolve2d_color(img, kernel) if img.ndim == 3 else signal.convolve2d(img, kernel, mode='same', boundary='fill', fillvalue=0)

# def high_pass(img, sigma, ksize=None):
#     return img - low_pass(img, sigma, ksize)

# def hybrid_image(im_high_src, im_low_src, sigma_high, sigma_low):
#     high = high_pass(im_high_src, sigma_high)
#     low = low_pass(im_low_src, sigma_low)
#     hybrid = low + high
#     return np.clip(hybrid, 0, 1), low, high

# def sigma_sweep(im_high_src, im_low_src, sigma_high_list, sigma_low_list, filename="sweep.png"):
#     n_rows, n_cols = len(sigma_high_list), len(sigma_low_list)
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
#     axes = np.array(axes).reshape(n_rows, n_cols)  # normalize shape even if 1 row/col
#     for i, sh in enumerate(sigma_high_list):
#         for j, sl in enumerate(sigma_low_list):
#             hybrid, _, _ = hybrid_image(im_high_src, im_low_src, sh, sl)
#             axes[i, j].imshow(hybrid)
#             axes[i, j].set_title(f"sh={sh}, sl={sl}", fontsize=9)
#             axes[i, j].axis('off')
#     plt.tight_layout()
#     plt.savefig(f"2/results/{filename}", dpi=150)
#     plt.show()

# def show_fft(img, title, filename):
#     if img.ndim == 3:
#         img_gray = cv2.cvtColor((np.clip(img,0,1)*255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)/255
#     else:
#         img_gray = img
#     fft = np.log(np.abs(np.fft.fftshift(np.fft.fft2(img_gray))) + 1e-8)
#     plt.figure()
#     plt.imshow(fft, cmap='gray')
#     plt.title(title)
#     plt.axis('off')
#     plt.savefig(f"2/results/{filename}", bbox_inches='tight', dpi=150)
#     plt.show()

# sigma1 = 8
# sigma2 = 5
# hybrid, low_img, high_img = hybrid_image(im2_aligned, im1_aligned, sigma1, sigma2)
# save_image(hybrid, "Hybrid Derek-Nutmeg", "hybrid_derek_nutmeg.png")

# # my own pictures
# jing = plt.imread('2/images/jing.jpg') / 255.
# raze = plt.imread('2/images/raze.jpg') / 255.
# jing_aligned, raze_aligned = align_images(jing, raze)

# sigma3 = 8
# sigma4 = 4

# hybrid_jing, low_jing, high_raze = hybrid_image(raze_aligned, jing_aligned, sigma3, sigma4)
# save_image(jing_aligned, "Jing", "jing_aligned.png")
# save_image(raze_aligned, "Raze", "raze_aligned.png")
# save_image(hybrid_jing, "Hybrid Jing-Raze", "hybrid_jing_raze.png")
# save_image(low_jing, "Low Jing", "low_jing.png")
# save_image(high_raze, "High Raze", "high_raze.png")

# smth = plt.imread('2/images/smth.jpg') / 255.
# jett = plt.imread('2/images/jett.jpg') / 255.
# smth_aligned, jett_aligned = align_images(smth, jett)

# sigma5 = 10
# sigma6 = 5
# hybrid_smth, low_jett, high_smth = hybrid_image(smth_aligned, jett_aligned, sigma5, sigma6)
# save_image(hybrid_smth, "Hybrid Smth-Jett", "hybrid_smth_jett.png")
# plt.imshow(hybrid_smth)

# sigma_sweep(im1_aligned, im2_aligned, [3, 5, 7], [6, 10, 14], "sweep_derek_nutmeg.png")
# sigma_sweep(raze_aligned, jing_aligned, [3, 5, 7], [6, 10, 14], "sweep_jing_raze.png")
# sigma_sweep(smth_aligned, jett_aligned, [3, 5, 7], [6, 10, 14], "sweep_smth_jett.png")

# show_fft(jing_aligned, "FFT: Jing (aligned)", "fft_jing.png")
# show_fft(raze_aligned, "FFT: Raze (aligned)", "fft_raze.png")
# show_fft(low_jing, "FFT: Low-pass", "fft_low.png")
# show_fft(high_raze, "FFT: High-pass", "fft_high.png")
# show_fft(hybrid_jing, "FFT: Hybrid", "fft_hybrid.png")


# # Part 2.3
def gaussian_stack(img, levels, sigma=2, ksize=None):
    if ksize is None:
        ksize = int(6 * sigma + 1) | 1
    kernel = gaussian_kernel(ksize, sigma)
    stack = [img]
    for i in range(1, levels):
        prev = stack[-1]
        blurred = convolve2d_color(prev, kernel) if img.ndim == 3 else \
                  signal.convolve2d(prev, kernel, mode='same', boundary='fill', fillvalue=0)
        stack.append(blurred)
    return stack

def laplacian_stack(g_stack):
    l_stack = []
    for i in range(len(g_stack) - 1):
        l_stack.append(g_stack[i] - g_stack[i + 1])
    l_stack.append(g_stack[-1])   
    return l_stack

# apple = load("2/images/apple.jpeg")
# orange = load("2/images/orange.jpeg")

# levels = 6
# sigma = 2

# g_stack_apple = gaussian_stack(apple, levels, sigma)
# l_stack_apple = laplacian_stack(g_stack_apple)

# g_stack_orange = gaussian_stack(orange, levels, sigma)
# l_stack_orange = laplacian_stack(g_stack_orange)

# def special_save_image(img, title, filename, diverging=False, gain=1.0):
#     plt.figure()
#     if diverging:
#         if img.ndim == 3:
#             plt.imshow(np.clip(img * gain + 0.5, 0, 1))
#         else:
#             plt.imshow(img * gain, cmap='gray', vmin=-0.5, vmax=0.5)
#     else:
#         plt.imshow(img, cmap='gray')
#     plt.title(title)
#     plt.axis('off')
#     plt.savefig(f"2/results/{filename}", bbox_inches='tight', dpi=150)
#     plt.show()

# for i in range(levels):
#     save_image(g_stack_apple[i], f"Gaussian level {i}", f"g_apple_{i}.png")
#     special_save_image(l_stack_apple[i], f"Laplacian level {i}", f"l_apple_{i}.png", diverging=True)
#     save_image(g_stack_orange[i], f"Gaussian level {i}", f"g_orange_{i}.png")
#     special_save_image(l_stack_orange[i], f"Laplacian level {i}", f"l_orange_{i}.png", diverging=True)


# # Part 2.4
def blend(im1, im2, mask, levels=5, sigma=2):
    g_stack1 = gaussian_stack(im1, levels, sigma)
    l_stack1 = laplacian_stack(g_stack1)

    g_stack2 = gaussian_stack(im2, levels, sigma)
    l_stack2 = laplacian_stack(g_stack2)

    g_stack_mask = gaussian_stack(mask, levels, sigma)

    blended_stack = []
    for i in range(levels):
        m = g_stack_mask[i]
        if im1.ndim == 3 and m.ndim == 2:
            m = m[:, :, None]   
        blended_level = m * l_stack1[i] + (1 - m) * l_stack2[i]
        blended_stack.append(blended_level)

    result = np.sum(blended_stack, axis=0)
    return np.clip(result, 0, 1), blended_stack, g_stack_mask, l_stack1, l_stack2

def match_shapes(*imgs, size=None):
    imgs = [im[:, :, :3] if im.ndim == 3 and im.shape[2] == 4 else im for im in imgs]  
    h0, w0 = imgs[0].shape[:2] if size is None else size
    target_ar = w0 / h0
    out = []
    for im in imgs:
        h, w = im.shape[:2]
        if w / h > target_ar:            
            new_w = int(round(h * target_ar))
            x0 = (w - new_w) // 2
            im = im[:, x0:x0 + new_w]
        elif w / h < target_ar:          
            new_h = int(round(w / target_ar))
            y0 = (h - new_h) // 2
            im = im[y0:y0 + new_h]
        im = cv2.resize(im, (w0, h0), interpolation=cv2.INTER_AREA)
        out.append(im)
    return out

def fig342(l1, l2, gm, result, filename, show=(0, 2, 4)):
    last = len(l1) - 1
    def prep(x, i): return np.clip(x + (0.5 if i < last else 0), 0, 1)
    def M(m, x): return m[:, :, None] * x if x.ndim == 3 else m * x
    fig, ax = plt.subplots(4, 3, figsize=(9, 12))
    for r, i in enumerate(show):
        a, b = M(gm[i], l1[i]), M(1 - gm[i], l2[i])
        for c, im in enumerate([a, b, a + b]):
            ax[r, c].imshow(prep(im, i)); ax[r, c].axis('off')
    A = sum(M(gm[i], l1[i]) for i in range(len(l1)))
    B = sum(M(1 - gm[i], l2[i]) for i in range(len(l1)))
    for c, im in enumerate([A, B, result]):
        ax[3, c].imshow(np.clip(im, 0, 1)); ax[3, c].axis('off')
    plt.tight_layout(); plt.savefig(f"2/results/{filename}", dpi=150); plt.show()
 
# h, w = apple.shape[:2]
# mask = np.zeros((h, w))
# mask[:, :w // 2] = 1   
# oraple, blended_stack, g_mask, l_apple, l_orange = blend(apple, orange, mask, levels=5, sigma=10)
# save_image(oraple, "Oraple", "oraple.png")
# fig342(l_apple, l_orange, g_mask, oraple, "fig342_oraple.png")

# rose = load("2/images/rose.jpg")
# carnation = load("2/images/carnation.jpg")
# rose, carnation= match_shapes(rose, carnation)

# h, w = rose.shape[:2]
# mask2 = np.zeros((h, w))
# mask2[:, :w // 2] = 1
# ronation, _, _, _, _ = blend(rose, carnation, mask2, levels=5, sigma=10)
# save_image(ronation, "Ronation", "ronation.png")

sage = load("2/images/sage.jpg")
jersey = load("2/images/jersey.jpg")
sage, jersey = match_shapes(sage, jersey)

mask3 = load_gray("2/images/sage_mask.png")
mask3 = (mask3 > 0.5).astype(float)   
print(mask3.shape, sage.shape[:2])   
prx_sage, _, gm3, l_j, l_s = blend(jersey, sage, mask3, levels=5, sigma=10)
save_image(prx_sage, "PRX Sage", "prx_sage.png")
fig342(l_j, l_s, gm3, prx_sage, "fig342_prx.png")