# %%
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import numpy as np

# k is for reducing dimension
rank_approximation = 200


# Singular value decomposition
def decomposition(image):
    # svd
    u, s, vh = np.linalg.svd(image)
    # form the sigma matrix (svd) from vector s
    st = np.zeros((u.shape[1], vh.shape[0]))
    st[:u.shape[1], :u.shape[1]] = np.diag(s)
    return u, st, vh


# compress the image with low rank approximation algorithm
def compress_image(u, st, vh, k):
    # final matrix
    B = u[:, :k] @ st[:k, :k] @ vh[:k, :]
    # output = matrix of final image
    return B


# calculate PSNR value
def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))


def plot_diagram(image, rank_range):
    u, st, vh = decomposition(image)
    psnr_array = np.zeros(rank_range)
    for i in range(rank_range):
        psnr_array[i] = psnr(image, compress_image(u, st, vh, i))
    plt.plot(psnr_array)
    plt.show()

#%%
# reading the image and converting to grayScale
image = PIL.Image.open('lion.jpg').convert('L')
image = np.asarray(image, dtype=float)
u, st, vh = decomposition(image)
final_image = compress_image(u, st, vh, rank_approximation)
final_image = np.array(final_image, dtype='uint8')
# print(psnr(image, final_image))

# converting array to image
img = Image.fromarray(final_image, 'L')
img.show()
img.save("lion200.jpg")

# plot_diagram(image, 1600)


# %%


def noisy(image, noise_type):
    row, col = image.shape
    noisy_img = np.zeros((row, col), dtype='uint8')
    noisy_img = noisy_img + image
    if noise_type == "gauss":
        noisy_img = noisy_img + np.random.normal(0, 20, (row, col))
    elif noise_type == "s&p":
        mat_w = np.random.rand(row, col)
        ans = np.zeros((row, col), dtype=bool)
        ans[mat_w < 0.045] = True
        noisy_img[ans] = 255

        mat_b = np.random.rand(row, col)
        ans = np.zeros((row, col), dtype=bool)
        ans[mat_b < 0.045] = True
        noisy_img[ans] = 0
    return noisy_img


img = PIL.Image.open('q2_pic.jpg').convert("L")
img = np.asarray(img)
gaussian_img = np.array(noisy(img, "gauss"), dtype='uint8')
print(psnr(np.array(img, dtype=float), np.array(gaussian_img, dtype=float)))
sp_img = np.array(noisy(img, "s&p"), dtype='uint8')
print(psnr(np.array(img, dtype=float), np.array(sp_img, dtype=float)))

#%%
u, s, vh = decomposition(np.array(gaussian_img, dtype=float))
u1, s1, vh1 = decomposition(np.array(sp_img, dtype=float))

#%%
gauss_psnr = np.zeros(500)
sp_psnr = np.zeros(500)
for i in range(500):
    gauss_psnr[i] = psnr(np.array(img, dtype=float), compress_image(u, s, vh, i))
    sp_psnr[i] = psnr(np.array(img, dtype=float), compress_image(u1, s1, vh1, i))
plt.plot(gauss_psnr, color='black')
plt.plot(sp_psnr, color='red')
plt.show()
#%%
noisy_image = np.array(sp_img, dtype='uint8')
noisy_image_g = np.array(gaussian_img, dtype='uint8')
noisy_image = Image.fromarray(noisy_image, 'L')
noisy_image_g = Image.fromarray(noisy_image_g, 'L')
noisy_image.show()
noisy_image_g.show()
noisy_image.save("s&p_noisedTOKA.jpg")
noisy_image.save("gaussian_noisedTOKA.jpg")
Image.fromarray(np.array(compress_image(u, s, vh, 13), dtype='uint8'), 'L').show()
Image.fromarray(np.array(compress_image(u1, s1, vh1, 13), dtype='uint8'), 'L').show()
Image.fromarray(np.array(compress_image(u, s, vh, 13), dtype='uint8'), 'L').save("TOKA_gauss_denoised_rank13.jpg")
Image.fromarray(np.array(compress_image(u1, s1, vh1, 13), dtype='uint8'), 'L').save("TOKA_sp_denoised_rank13.jpg")
Image.fromarray(np.array(compress_image(u, s, vh, 25), dtype='uint8'), 'L').show()
Image.fromarray(np.array(compress_image(u1, s1, vh1, 25), dtype='uint8'), 'L').show()
Image.fromarray(np.array(compress_image(u, s, vh, 25), dtype='uint8'), 'L').save("TOKA_gauss_denoised_rank25.jpg")
Image.fromarray(np.array(compress_image(u1, s1, vh1, 25), dtype='uint8'), 'L').save("TOKA_sp_denoised_rank25.jpg")
Image.fromarray(np.array(compress_image(u, s, vh, 150), dtype='uint8'), 'L').show()
Image.fromarray(np.array(compress_image(u1, s1, vh1, 150), dtype='uint8'), 'L').show()
Image.fromarray(np.array(compress_image(u, s, vh, 150), dtype='uint8'), 'L').save("TOKA_gauss_denoised_rank150.jpg")
Image.fromarray(np.array(compress_image(u1, s1, vh1, 150), dtype='uint8'), 'L').save("TOKA_sp_denoised_rank150.jpg")
