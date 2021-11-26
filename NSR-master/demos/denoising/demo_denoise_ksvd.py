from sklearn.feature_extraction.image \
    import extract_patches_2d, reconstruct_from_patches_2d
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
from ksvd import KSVD
from ksvd import _omp


def get_psnr(im, recon):
    return 10. * np.log(im.max() / np.sqrt(np.mean((im - recon) ** 2)))


def show_dictionary(A, name=None):
    n = int(np.sqrt(A.shape[0]))
    m = int(np.sqrt(A.shape[1]))
    A_show = A.reshape((n, n, m, m))
    fig, ax = plt.subplots(m, m, figsize=(4, 4))
    for row in range(m):
        for col in range(m):
            ax[row, col].imshow(A_show[:, :, col, row],
                                cmap='gray', interpolation='Nearest')
            ax[row, col].axis('off')
    if name is not None:
        plt.savefig(name, dpi=220)


def denoise_with_learned_dictionary(im, A, k0, eps, patch_size=8,
                                    lam=0.5, n_iter=1, im0=None):
    """ 学習済み辞書によるノイズ除去 """
    recon = im.copy()
    for h in range(n_iter):
        patches = extract_patches_2d(recon, (patch_size, patch_size))
        if h == 0:
            q = np.zeros((len(patches), A.shape[1]))
        for i, patch in enumerate(patches):
            if i % 1000 == 0:
                print(i)
            q[i] = _omp(patch.flatten(), A, k0, eps)
        recon_patches = \
            (np.dot(A, q.T).T).reshape((-1, patch_size, patch_size))
        recon = reconstruct_from_patches_2d(recon_patches, im.shape)
        recon = (im * lam + recon) / (lam + 1.)
        if im0 is not None:
            print(h, get_psnr(im0, recon))
    return recon

# make noise image
im = imread('barbara.png').astype(np.float)
Y = im + np.random.randn(im.shape[0], im.shape[1]) * 20.
Y.tofile('barbara_sig20')
Y = np.fromfile('barbara_sig20').reshape(im.shape)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(im, cmap='gray', interpolation='Nearest', vmin=0, vmax=255)
ax[1].imshow(Y, cmap='gray', interpolation='Nearest', vmin=0, vmax=255)
ax[0].axis('off')
ax[1].axis('off')
ax[0].set_title('Original')
ax[1].set_title('With noise \n{:.3f}'.format(get_psnr(im, Y)))
print("creating noise image done.")
plt.show(block=False)
plt.draw()


# make DCT dictionary as initial one
patch_size = 8
dict_size = 16
A_1D = np.zeros((patch_size, dict_size))
for k in np.arange(dict_size):
    for i in np.arange(patch_size):
        A_1D[i, k] = np.cos(i * k * np.pi / float(dict_size))
    if k != 0:
        A_1D[:, k] -= A_1D[:, k].mean()

A_DCT = np.kron(A_1D, A_1D)
print("creating DCT dictionary done.")
show_dictionary(A_DCT)
plt.show(block=False)
A_DCT.tofile('A_DCT.png')

# learning dictionary
patch_size = 8
patches = extract_patches_2d(Y, (patch_size, patch_size))
patches = patches.reshape((-1, patch_size ** 2))
M = len(patches)
A_KSVD = A_DCT.copy()
n_components = 256
sigma = 20
n_nonzero_coefs = 4
print('start learning dictionary')
model = KSVD(n_components=n_components, sigma=sigma,
             n_nonzero_coefs=n_nonzero_coefs, solver='omp',
             max_iter=1, verbose=1)

for _ in range(15):
    ndx = np.random.permutation(M)[:M // 10]
    est_code = model.fit_transform(patches[ndx].T, initial_dict=A_KSVD)
    A_KSVD = model.dictionary
    print('done')

# showing learned dictionary
show_dictionary(A_KSVD)
plt.show(block=False)
A_KSVD.tofile('A_KSVD.png')

# reconstructing image
eps = (patch_size ** 2) * (20. ** 2) * 1.15
recon_ksvd_dictionary = \
    denoise_with_learned_dictionary(Y, A_KSVD, 4, eps, im0=im)
recon_ksvd_dictionary.tofile('recon_ksvd_dictionary')

# showing reconstructed image
recon_ksvd_dictionary = np.fromfile('recon_ksvd_dictionary').reshape(im.shape)
plt.figure()
plt.imshow(recon_ksvd_dictionary, cmap='gray', interpolation='Nearest')
plt.axis('off')
plt.title('Reconstructed image\n{:.3f}'.format(get_psnr(im, recon_ksvd_dictionary)))
plt.savefig('recon_ksvd_dictionary.png', dpi=220)
plt.show()
