import unittest
import numpy as np
import scipy.fft
import skimage
import Deconvolvers
import matplotlib.pyplot as plt
import scipy
path = '/home/valerii/Documents/projects/Deconvolution/Presentation Images/'

def make_mask_cosine_edge(shape, edge):
    """
    Weight mask that vanishes with the cosine distance to the edge.
    """
    # no valid edge -> no masking
    if edge <= 0:
        return np.ones(shape)
    # instead of computing the mask directly, the relative distance to the nearest
    # edge within the configured width is computed. this only needs to be done
    # once for one corner and can then be mirrored accordingly.
    d = np.linspace(0.0, 1.0, num=edge)
    dx, dy = np.meshgrid(d, d)
    dxy = np.hypot(dx, dy)
    dcorner = np.where(dx < dy, dx, dy)
    dcorner = np.where(dxy < dcorner, dxy, dcorner)
    print(dcorner.shape)
    dist = np.ones(shape)
    dist[..., :edge, :] = d[:, np.newaxis]
    dist[..., -edge:, :] = d[::-1, np.newaxis]
    dist[..., :, :edge] = d
    dist[..., :, -edge:] = d[::-1]
    dist[..., :edge, :edge] = dcorner
    dist[..., -edge:, :edge] = dcorner[::-1, :]
    dist[..., :edge, -edge:] = dcorner[:, ::-1]
    dist[..., -edge:, -edge:] = dcorner[::-1, ::-1]
    # convert distance to weight
    return np.sin(0.5 * np.pi * dist)
class TestNotBlind(unittest.TestCase):
    def test_vs_skimage(self):
        # img = 1 - skimage.io.imread("images/cross.png", as_gray=True)
        img = skimage.io.imread("images/Automobile.png", as_gray=True)
        sizex, sizey = img.shape
        img /= np.sum(img)
        x, y = np.arange(sizex), np.arange(sizey)
        X, Y = np.meshgrid(x, y)
        X, Y = X - sizex // 2, Y - sizey // 2
        R = (X ** 2 + Y ** 2) ** 0.5
        psf_real = np.exp(-R ** 2 / 900)
        psf_real /= np.sum(psf_real)
        blurred = scipy.signal.convolve(img, psf_real, mode='same')
        rld = Deconvolvers.RichardsonLucy()
        restored, history = rld.deconvolve(blurred, psf_real, 50)
        rld.regularization_fitler=10**-10
        fig, axes = plt.subplots(1, 4)
        fig.subplots_adjust(wspace=0.05)
        for ax in axes:
            ax.set_axis_off()
        restored_default = skimage.restoration.richardson_lucy(blurred, psf_real, num_iter=50, filter_epsilon=10 ** -10)
        axes[0].imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(img))
        axes[0].set_title('Object')
        axes[1].imshow(blurred, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(blurred))
        axes[1].set_title('Blurred')
        axes[2].imshow(restored, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(restored))
        axes[2].set_title('RLD')
        axes[3].imshow(restored_default, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(restored_default))
        axes[3].set_title('RLD skimage')
        fig.savefig(path + 'blind_vs_default2.png')
        plt.tight_layout()
        # plt.show()

    def test_convergence(self):
        # img = 1 - skimage.io.imread("images/cross.png", as_gray=True)
        size = 11
        img = np.zeros((size, size))
        img[size//2, size//2] = 1
        sizex, sizey = size, size
        img /= np.sum(img)
        x, y = np.arange(sizex), np.arange(sizey)
        X, Y = np.meshgrid(x, y)
        X, Y = X - sizex // 2, Y - sizey // 2
        R = (X ** 2 + Y ** 2) ** 0.5
        psf_real = np.exp(-R ** 2 / 4)
        psf_real /= np.sum(psf_real)
        blurred = scipy.signal.convolve(img, psf_real, mode='same')
        rld = Deconvolvers.RichardsonLucy(regularization_filter=10**-10)
        restored, history = rld.deconvolve(blurred, psf_real, 100)

        rldb = Deconvolvers.RichardsonLucyBlind(1)
        # psf0 = np.zeros((11,11)) + 1 / 121
        psf0 = np.exp(-R**2 / 4)
        restored, psf, historyb = rldb.deconvolve(blurred, psf0, stepNumber=100)

        fig1, axes = plt.subplots(2, 3)
        fig1.subplots_adjust(wspace=0.1, hspace=0.1)
        for row in axes:
            for ax in row:
                ax.set_axis_off()
        # axes[0].imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(img))

        axes[0][0].imshow(blurred, cmap=plt.get_cmap('gray'), vmin=0, vmax=0.5)
        axes[0][1].imshow(history[9], cmap=plt.get_cmap('gray'), vmin=0, vmax=0.5)
        axes[0][2].imshow(history[99], cmap=plt.get_cmap('gray'), vmin=0, vmax=0.5)
        axes[1][0].imshow(blurred, cmap=plt.get_cmap('gray'), vmin=0, vmax=0.5)
        axes[1][1].imshow(historyb[1][9], cmap=plt.get_cmap('gray'), vmin=0, vmax=0.5)
        axes[1][2].imshow(historyb[1][99], cmap=plt.get_cmap('gray'), vmin=0, vmax=0.5)
        # fig1.savefig(path + 'convergence_point_source.png')
        # plt.show()
        mse = np.zeros(len(history))
        mseb = np.zeros(len(historyb[1]))
        # fig2, axes2 = plt.subplots(2, 5, figsize=(15, 5))
        slices = np.arange(0, len(mse), 5)
        # for i in range(10):
        #     axes2[i//5, i % 5].imshow(history[slices[i]])
        # # plt.show()
        fig3, axes3 = plt.subplots(1)
        axes3.grid()
        for i in range(len(mse)):
            mse[i] = np.sum((history[i] - img) ** 2) ** 0.5
            mseb[i] = np.sum((historyb[1][i] - img) ** 2) ** 0.5
            if i in slices:
                axes3.plot(i, mse[i], "X", color='black')
                axes3.plot(i, mseb[i], "X", color='black')
        axes3.plot(mse, label='non-blind')
        axes3.plot(mseb, label='blind')
        axes3.set_xlim(0, 100)
        axes3.set_ylim(0, 1)
        axes3.legend(loc='center right')
        # fig3.savefig(path + 'mse_point_source.png')
        plt.show()

    def test_different_objects(self):
        size = 400
        cross = 1 - skimage.io.imread("images/cross.png", as_gray=True)
        cell = skimage.io.imread("images/cell.jpg", as_gray=True)
        car = skimage.io.imread("images/Automobile.png", as_gray=True)
        images = [cross, cell, car]

        #cropping
        cropped_images = []
        for image in images:
            shape = image.shape
            cropped_images.append(image[(shape[0]+1)//2 - 200: (shape[0]+1)//2 + 200,
                (shape[1]+1)//2 - 200: (shape[1]+1)//2 + 200])

        #normalization
        for i in range(len(cropped_images)):
            cropped_images[i] /= np.sum(cropped_images[i])


        x, y = np.arange(size), np.arange(size)
        X, Y = np.meshgrid(x, y)
        X, Y = X - size // 2, Y - size // 2
        R = (X ** 2 + Y ** 2) ** 0.5
        psf_real = np.exp(-R ** 2 / 100)
        psf_real /= np.sum(psf_real)
        fig, axes = plt.subplots(len(cropped_images), 3, figsize=(9,9))
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        for row in axes:
            for ax in row:
                ax.set_axis_off()

        # psf_real = np.pad(psf_real, 50, mode='constant', constant_values=0)
        mask = make_mask_cosine_edge((size, size), 50)
        for i in range(len(cropped_images)):
            image = cropped_images[i]
            image *= mask**2
            # image = np.pad(image, 50, mode='constant', constant_values=0)
            blurred = scipy.signal.convolve(image, psf_real, mode='same')
            rld = Deconvolvers.RichardsonLucy(regularization_filter=10**-15)
            restored, history = rld.deconvolve(blurred, psf_real, 100)
            axes[i][0].imshow(image, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(image))
            axes[i][1].imshow(blurred, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(blurred))
            axes[i][2].imshow(restored, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(restored))
        plt.tight_layout()
        fig.savefig(path + 'edge_artifacts_with_mask.png')
        plt.show()
    def test_different_blur(self):
        size = 400
        cross = 1 - skimage.io.imread("images/cross.png", as_gray=True)
        cell = skimage.io.imread("images/cell.jpg", as_gray=True)
        car = skimage.io.imread("images/Automobile.png", as_gray=True)
        images = [cross, cell, car]

        #cropping
        cropped_images = []
        for image in images:
            shape = image.shape
            cropped_images.append(image[(shape[0]+1)//2 - 200: (shape[0]+1)//2 + 200,
                          (shape[1]+1)//2 - 200: (shape[1]+1)//2 + 200])

        #normalization
        for i in range(len(cropped_images)):
            cropped_images[i] /= np.sum(cropped_images[i])


        x, y = np.arange(size), np.arange(size)
        X, Y = np.meshgrid(x, y)
        X, Y = X - size // 2, Y - size // 2
        fx = np.linspace(-1 / 2, 1 / 2  - 1 / size, size)
        fy = np.linspace(-1 / 2, 1 / 2  - 1 / size, size)
        Fx, Fy = np.meshgrid(fx, fy)
        R = (X ** 2 + Y ** 2) ** 0.5
        alpha = 0.2
        csf_lens = 2 * scipy.special.j1(R * alpha)/(R * alpha)
        csf_lens[size//2, size//2] = 1
        psf_lens = csf_lens**2
        psf_lens /= np.sum(psf_lens)

        psf_gaussian = np.exp(-R ** 2 / 100)
        psf_gaussian /= np.sum(psf_gaussian)

        vx, vy = -100, 20
        T = 1
        otf_motion = 2 * T * np.sin((vx * Fx + vy * Fy))/(vx * Fx + vy * Fy)
        otf_motion = np.nan_to_num(otf_motion, nan=2)
        psf_motion = np.abs(scipy.fft.fftshift(scipy.fft.fft2(scipy.fft.fftshift(otf_motion))))
        # plt.imshow(psf_motion)
        # plt.show()
        psf_motion /= np.sum(psf_motion)

        blurs = [psf_lens, psf_gaussian, psf_motion]

        fig, axes = plt.subplots(len(cropped_images), 3, figsize=(9, 9))
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        for row in axes:
            for ax in row:
                ax.set_axis_off()

        # car = np.pad(car, 50, mode='constant', constant_values=0)

        for i in range(len(blurs)):
            blur = blurs[i]
            # blur = np.pad(blur, 50, mode='constant', constant_values=0)
            blurred = scipy.signal.convolve(car, blur, mode='same')
            rld = Deconvolvers.RichardsonLucy(regularization_filter=10**-15)
            restored, history = rld.deconvolve(blurred, blur, 100)
            axes[i][0].imshow(car, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(car))
            axes[i][1].imshow(blurred, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(car))
            axes[i][2].imshow(restored, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(car))
        plt.tight_layout()
        fig.savefig(path + 'different_blur_non_blind.png')
        plt.show()

    def test_different_noise_levels(self):
        size = 400
        cross = 1 - skimage.io.imread("images/cross.png", as_gray=True)
        cell = skimage.io.imread("images/cell.jpg", as_gray=True)
        # cell /= np.amax(cell)
        car = skimage.io.imread("images/Automobile.png", as_gray=True)
        image = cell

        #cropping
        # plt.imshow(noised1)
        # plt.show()
        # plt.imshow(noised2)
        # plt.show()
        image = image[(size+1)//2 - 200: (size+1)//2 + 200,
                          (size+1)//2 - 200: (size+1)//2 + 200]

        # noised1 = skimage.util.random_noise(cropped_images[0], mode='s&p', amount=0.01)
        # noised2 = skimage.util.random_noise(cropped_images[0], mode='speckle')
        # noised = [noised1, noised2]
        # #normalization
        # for i in range(len(noised)):
        #     noised[i] /= np.sum(noised[i])


        x, y = np.arange(size), np.arange(size)
        X, Y = np.meshgrid(x, y)
        X, Y = X - size // 2, Y - size // 2
        R = (X ** 2 + Y ** 2) ** 0.5
        alpha = 0.5
        csf_lens = 2 * scipy.special.j1(R * alpha)/(R * alpha)
        csf_lens[size//2, size//2] = 1
        psf_lens = csf_lens**2
        psf_lens /= np.sum(psf_lens)

        # psf_real = np.pad(psf_lens, 50, mode='constant', constant_values=0)
        mask = make_mask_cosine_edge((size, size), 50)
        # image = np.pad(image, 50, mode='constant', constant_values=0)
        blurred = scipy.signal.convolve(image, psf_lens, mode='same')
        noised0 = skimage.util.random_noise(blurred, mode='s&p', amount=0.00)
        noised1 = skimage.util.random_noise(blurred, mode='s&p', amount=0.01)
        noised2 = skimage.util.random_noise(blurred, mode='speckle')
        noised = [noised0, noised1, noised2]

        fig, axes = plt.subplots(len(noised), 3, figsize=(9,9))
        for row in axes:
            for ax in row:
                ax.set_axis_off()
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        plt.tight_layout()
        for i in range(len(noised)):
            noised[i] *= mask ** 2
            noised[i] /= np.sum(noised[i])
            bn = noised[i]
            rld = Deconvolvers.RichardsonLucy(regularization_filter=10**-5)
            restored, history = rld.deconvolve(bn, psf_lens, 100)
            axes[i][0].imshow(image, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(image))
            axes[i][1].imshow(bn, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(noised[i]))
            axes[i][2].imshow(restored, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(noised[i]))
        fig.savefig(path + '_rl_vs_noise.png')
        plt.show()

    def test_vs_wiener(self):
        size = 400
        cross = 1 - skimage.io.imread("images/cross.png", as_gray=True)
        cell = skimage.io.imread("images/cell.jpg", as_gray=True)
        # cell /= np.amax(cell)
        car = skimage.io.imread("images/Automobile.png", as_gray=True)
        image = cell

        # cropping
        # plt.imshow(noised1)
        # plt.show()
        # plt.imshow(noised2)
        # plt.show()
        image = image[(size + 1) // 2 - 200: (size + 1) // 2 + 200,
                (size + 1) // 2 - 200: (size + 1) // 2 + 200]

        # noised1 = skimage.util.random_noise(cropped_images[0], mode='s&p', amount=0.01)
        # noised2 = skimage.util.random_noise(cropped_images[0], mode='speckle')
        # noised = [noised1, noised2]
        # #normalization
        # for i in range(len(noised)):
        #     noised[i] /= np.sum(noised[i])

        x, y = np.arange(size), np.arange(size)
        X, Y = np.meshgrid(x, y)
        X, Y = X - size // 2, Y - size // 2
        R = (X ** 2 + Y ** 2) ** 0.5
        alpha = 0.5
        csf_lens = 2 * scipy.special.j1(R * alpha) / (R * alpha)
        csf_lens[size // 2, size // 2] = 1
        psf_lens = csf_lens ** 2
        psf_lens /= np.sum(psf_lens)

        # psf_real = np.pad(psf_lens, 50, mode='constant', constant_values=0)
        mask = make_mask_cosine_edge((size, size), 50)
        # image = np.pad(image, 50, mode='constant', constant_values=0)
        blurred = scipy.signal.convolve(image, psf_lens, mode='same')
        noised0 = skimage.util.random_noise(blurred, mode='s&p', amount=0.00)
        noised1 = skimage.util.random_noise(blurred, mode='s&p', amount=0.01)
        noised2 = skimage.util.random_noise(blurred, mode='speckle')
        noised = [noised0, noised1, noised2]

        fig, axes = plt.subplots(len(noised), 3, figsize=(9, 9))
        for row in axes:
            for ax in row:
                ax.set_axis_off()
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        plt.tight_layout()
        for i in range(len(noised)):
            noised[i] *= mask ** 2
            noised[i] /= np.sum(noised[i])
            bn = noised[i]
            noised_ft = scipy.fft.fftshift(scipy.fft.ifft2(scipy.fft.fftshift(bn)))
            otf = np.abs(scipy.fft.fftshift(scipy.fft.fft2(scipy.fft.fftshift(psf_lens))))
            noise_power = 1
            wiener_filtered_ft = otf.conjugate() * noised_ft / (otf * otf.conjugate() + noise_power)
            restored = scipy.fft.fftshift(scipy.fft.fft2(scipy.fft.fftshift(wiener_filtered_ft)))
            axes[i][0].imshow(image, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(image))
            axes[i][1].imshow(bn, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(bn))
            axes[i][2].imshow(np.abs(restored), cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(np.abs(restored)))
        fig.savefig(path + '_wiener_vs_noise.png')
        plt.show()
    def test_out_of_band_restoration(self):
        img = 1-skimage.io.imread("images/cross.png", as_gray=True)
        # img = skimage.io.imread("images/cell.jpg", as_gray=True)
        # img = skimage.io.imread("images/Automobile.png", as_gray=True)
        size = 400
        # img = np.zeros((size, size))
        # img[size//2, size//2] = 1
        img = img[img.shape[0]//2 - size//2:img.shape[0]//2 + size//2, img.shape[1]//2 - size//2:img.shape[1]//2 + size//2]
        img += 10 ** (-10)
        mask = make_mask_cosine_edge((400, 400), 50)
        img *= mask**2
        img /= np.sum(img)
        img_ft = scipy.fft.fftshift(scipy.fft.fft2(scipy.fft.fftshift(img)))
        x, y = np.arange(size), np.arange(size)
        X, Y = np.meshgrid(x, y)
        X, Y = X - size // 2, Y - size // 2
        R = (X ** 2 + Y ** 2) ** 0.5
        alpha = 0.2
        csf_lens = 2 * scipy.special.j1(R * alpha) / (R * alpha)
        csf_lens[size // 2, size // 2] = 1
        psf_lens = csf_lens ** 2
        psf_lens /= np.sum(psf_lens)
        otf = scipy.fft.fftshift(scipy.fft.fft2(scipy.fft.fftshift(psf_lens)))

        blurred = scipy.signal.convolve(img, psf_lens, mode='same')
        rld = Deconvolvers.RichardsonLucy(regularization_filter=10**-8)
        restored, history = rld.deconvolve(blurred, psf_lens, 100)
        fig1, axes = plt.subplots(1, 3)
        axes[0].imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(img))
        axes[1].imshow(blurred, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(blurred))
        axes[2].imshow(restored, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(restored))
        plt.show()
        mse = np.zeros(len(history))
        fig2, axes2 = plt.subplots(1, figsize=(10, 8))
        slices = np.arange(0, len(mse), 5)
        # axes2.axhline(y=1, color='red')
        axes2.axvline(alpha/np.pi * size, color='red', label='2NA/$\lambda$')
        axes2.plot(otf[200, 200:], color='black')
        for i in range(5):
            imgr_ft = scipy.fft.fftshift(scipy.fft.fft2(scipy.fft.fftshift(history[slices[i]])))
            freqs = np.abs(imgr_ft[200, :]/img_ft[200, :])
            axes2.plot(freqs[200:], color='black')
        axes2.grid()
        axes2.set_xlim(0, 60)
        axes2.set_ylim(bottom=0)
        # fig2.savefig(path + 'out-of-band_image.png')
        # plt.show()
        # fig3, axes3 = plt.subplots(1)
        # for i in range(len(mse)):
        #     mse[i] = np.sum((history[i] - img) ** 2) ** 0.5
        #     if i in slices:
        #         axes3.plot(i, mse[i], "X", color='black')
        # axes3.plot(mse)
        plt.legend()
        plt.show()

class TestBlind(unittest.TestCase):
    def test_vs_non_blind_same_psf(self):
        img = 1 - skimage.io.imread("images/cross.png", as_gray=True)
        sizex, sizey = img.shape
        img /= np.sum(img)
        x, y = np.arange(sizex), np.arange(sizey)
        X, Y = np.meshgrid(x, y)
        X, Y = X - sizex // 2, Y - sizey // 2
        R = (X ** 2 + Y ** 2) ** 0.5
        psf_real = np.exp(-R ** 2 / 900)
        psf_real /= np.sum(psf_real)
        blurred = scipy.signal.convolve(img, psf_real, mode='same')
        rld = Deconvolvers.RichardsonLucy()
        rldb = Deconvolvers.RichardsonLucyBlind(5)
        restored, history = rld.deconvolve(blurred, psf_real, 20)
        restored_blind, psfb, history_blind = rldb.deconvolve(blurred, psf_real, 20)
        fig, axes = plt.subplots(1, 4)
        axes[0].imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(img))
        axes[1].imshow(blurred, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(img))
        axes[2].imshow(restored, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(img))
        axes[3].imshow(restored_blind, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(img))
        fig2, axes2 = plt.subplots(1, 2)
        axes2[0].imshow(psf_real)
        axes2[1].imshow(psfb)
        plt.show()

    def test_convergence(self):
        # img = 1 - skimage.io.imread("images/cross.png", as_gray=True)
        size = 500
        img = np.zeros((size, size))
        img[size//2, size//2] = 1
        sizex, sizey = size, size
        img += 10 ** (-10)
        img /= np.sum(img)
        x, y = np.arange(sizex), np.arange(sizey)
        X, Y = np.meshgrid(x, y)
        X, Y = X - sizex // 2, Y - sizey // 2
        R = (X ** 2 + Y ** 2) ** 0.5
        psf_real = np.exp(-R ** 2 / 900)
        psf_real /= np.sum(psf_real)
        blurred = scipy.signal.convolve(img, psf_real, mode='same')
        rldb = Deconvolvers.RichardsonLucyBlind(5)
        restored, psfb, history = rldb.deconvolve(blurred, stepNumber=20)
        fig1, axes = plt.subplots(1, 3)
        axes[0].imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(img))
        axes[1].imshow(blurred, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(blurred))
        axes[2].imshow(restored, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(restored))
        plt.show()
        mse = np.zeros(len(history[1]))
        fig2, axes2 = plt.subplots(2, 5, figsize=(15, 5))
        slices = np.arange(0, len(mse), 10)
        for i in range(10):
            axes2[i//5, i % 5].imshow(history[1, slices[i]])
        # plt.show()
        fig3, axes3 = plt.subplots(1)
        for i in range(len(mse)):
            mse[i] = np.sum((history[i] - img) ** 2) ** 0.5
            if i in slices:
                axes3.plot(i, mse[i], "X", color='black')
        axes3.plot(mse)
        plt.show()

    def test_different_objects(self):
        size = 400
        cross = 1 - skimage.io.imread("images/cross.png", as_gray=True)
        cell = skimage.io.imread("images/cell.jpg", as_gray=True)
        car = skimage.io.imread("images/Automobile.png", as_gray=True)
        images = [cross, cell, car]

        #cropping
        cropped_images = []
        for image in images:
            shape = image.shape
            cropped_images.append(image[(shape[0]+1)//2 - 200: (shape[0]+1)//2 + 200,
                (shape[1]+1)//2 - 200: (shape[1]+1)//2 + 200])

        #normalization
        for i in range(len(cropped_images)):
            cropped_images[i] /= np.sum(cropped_images[i])


        x, y = np.arange(size), np.arange(size)
        X, Y = np.meshgrid(x, y)
        X, Y = X - size // 2, Y - size // 2
        R = (X ** 2 + Y ** 2) ** 0.5
        psf_real = np.exp(-R ** 2 / 100)
        psf_real /= np.sum(psf_real)
        fig, axes = plt.subplots(len(cropped_images), 3)

        psfg = np.exp(-R ** 2 / 30)
        psfg /=np.sum(psfg)
        psfg = np.pad(psfg, 50, mode='constant', constant_values=0)

        for i in range(len(cropped_images)):
            image = cropped_images[i]
            image = np.pad(image, 50, mode='constant', constant_values=0)
            blurred = scipy.signal.convolve(image, psf_real, mode='same')
            rldb = Deconvolvers.RichardsonLucyBlind(5, regularization_filter=10**-15)
            restored, psf, history = rldb.deconvolve(blurred, psfg, 20)
            axes[i][0].imshow(image, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(image))
            axes[i][1].imshow(blurred, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(blurred))
            axes[i][2].imshow(restored, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(restored))
        plt.show()
    def test_different_blur(self):
        size = 400
        cross = 1 - skimage.io.imread("images/cross.png", as_gray=True)
        cell = skimage.io.imread("images/cell.jpg", as_gray=True)
        car = skimage.io.imread("images/Automobile.png", as_gray=True)
        images = [cross, cell, car]

        #cropping
        cropped_images = []
        for image in images:
            shape = image.shape
            cropped_images.append(image[(shape[0]+1)//2 - 200: (shape[0]+1)//2 + 200,
                          (shape[1]+1)//2 - 200: (shape[1]+1)//2 + 200])

        #normalization
        for i in range(len(cropped_images)):
            cropped_images[i] /= np.sum(cropped_images[i])


        x, y = np.arange(size), np.arange(size)
        X, Y = np.meshgrid(x, y)
        X, Y = X - size // 2, Y - size // 2
        fx = np.linspace(-1 / 2, 1 / 2  - 1 / size, size)
        fy = np.linspace(-1 / 2, 1 / 2  - 1 / size, size)
        Fx, Fy = np.meshgrid(fx, fy)
        R = (X ** 2 + Y ** 2) ** 0.5
        alpha = 0.2
        csf_lens = 2 * scipy.special.j1(R * alpha)/(R * alpha)
        csf_lens[size//2, size//2] = 1
        psf_lens = csf_lens**2
        psf_lens /= np.sum(psf_lens)

        psf_gaussian = np.exp(-R ** 2 / 100)
        psf_gaussian /= np.sum(psf_gaussian)

        psf_guess = np.exp(-R ** 2 / 400)
        psf_guess /= np.sum(psf_guess)
        plt.imshow(np.log(1 + 10 ** 3 * psf_guess[size//2-50:size//2+50, size//2-50:size//2+50]),cmap=plt.get_cmap('gray'), vmin=0)
        plt.show()
        vx, vy = -100, 20
        T = 1
        otf_motion = 2 * T * np.sin((vx * Fx + vy * Fy))/(vx * Fx + vy * Fy)
        otf_motion = np.nan_to_num(otf_motion, nan=2)
        psf_motion = np.abs(scipy.fft.fftshift(scipy.fft.fft2(scipy.fft.fftshift(otf_motion))))
        # plt.imshow(psf_lens)
        # plt.show()
        psf_motion /= np.sum(psf_motion)

        blurs = [psf_lens, psf_gaussian, psf_motion]

        fig, axes = plt.subplots(len(cropped_images), 5, figsize=(15,9))
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        for row in axes:
            for ax in row:
                ax.set_axis_off()

        # car = np.pad(car, 50, mode='constant', constant_values=0)
        mask = make_mask_cosine_edge((size, size), 50)
        car *= mask**2
        for i in range(len(blurs)):
            blur = blurs[i]
            # blur = np.pad(blur, 50, mode='constant', constant_values=0)
            blurred = scipy.signal.convolve(car, blur, mode='same')
            # psf_guess = np.pad(psf_guess, 50, mode='constant', constant_values=0)
            rldb = Deconvolvers.RichardsonLucyBlind(5, regularization_filter=10**-10)
            restored, psf, history = rldb.deconvolve(blurred, psf_guess, 60)
            axes[i][0].imshow(car, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(car))
            axes[i][1].imshow(blurred, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(car))
            axes[i][2].imshow(restored, cmap=plt.get_cmap('gray'))
            axes[i][3].imshow(np.log(1 + 10 ** 3 * blur[size//2-50:size//2+50, size//2-50:size//2+50]), cmap=plt.get_cmap('gray'), vmin=0)
            axes[i][4].imshow(np.log(1 + 10 ** 3 * psf[size//2-50:size//2+50, size//2-50:size//2+50]),cmap=plt.get_cmap('gray'), vmin=0)
        plt.tight_layout()
        fig.savefig(path + 'different_blur_blind.png')
        plt.show()

    def test_different_noise_levels(self):
        size = 400
        cross = 1 - skimage.io.imread("images/cross.png", as_gray=True)
        cell = skimage.io.imread("images/cell.jpg", as_gray=True)
        # cell /= np.amax(cell)
        car = skimage.io.imread("images/Automobile.png", as_gray=True)
        images = [car, ]

        #cropping
        cropped_images = []
        for image in images:
            shape = image.shape
            cropped_images.append(image[(shape[0]+1)//2 - 200: (shape[0]+1)//2 + 200,
                          (shape[1]+1)//2 - 200: (shape[1]+1)//2 + 200])

        #normalization
        for i in range(len(cropped_images)):
            cropped_images[i] /= np.sum(cropped_images[i])


        x, y = np.arange(size), np.arange(size)
        X, Y = np.meshgrid(x, y)
        X, Y = X - size // 2, Y - size // 2
        R = (X ** 2 + Y ** 2) ** 0.5
        alpha = 0.3
        csf_lens = 2 * scipy.special.j1(R * alpha)/(R * alpha)
        csf_lens[size//2, size//2] = 1
        psf_lens = csf_lens**2
        psf_lens /= np.sum(psf_lens)

        psf_real = np.pad(psf_lens, 50, mode='constant', constant_values=0)
        image = cropped_images[0]
        image = np.pad(image, 50, mode='constant', constant_values=0)

        blurred = scipy.signal.convolve(image, psf_real, mode='same')
        noised1 = blurred + skimage.util.random_noise(blurred, 'gaussian', var=10**-10)
        noised2 = skimage.util.random_noise(blurred, 's&p', amount=0.0005)
        noised = [noised1, noised2]
        fig, axes = plt.subplots(len(noised), 5)

        for i in range(len(noised)):
            bn = noised[i]
            psf_guess = np.exp(-R**2 / 400)
            psf_guess /= np.sum(psf_guess)
            psf_guess = np.pad(psf_guess, 50, mode='constant', constant_values=0)
            rldb = Deconvolvers.RichardsonLucyBlind(5, regularization_filter=10**-15)
            restored, psf, history = rldb.deconvolve(bn, psf_guess, 100)
            axes[i][0].imshow(image, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(image))
            axes[i][1].imshow(bn, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(image))
            axes[i][2].imshow(restored, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(image))
            axes[i][3].imshow(psf_real, cmap=plt.get_cmap('gray'), vmin=0)
            axes[i][4].imshow(psf, cmap=plt.get_cmap('gray'))
        plt.show()

    def test_vs_wiener(self):
        size = 400
        cross = 1 - skimage.io.imread("images/cross.png", as_gray=True)
        cell = skimage.io.imread("images/cell.jpg", as_gray=True)
        # cell /= np.amax(cell)
        car = skimage.io.imread("images/Automobile.png", as_gray=True)
        images = [car, ]

        #cropping
        cropped_images = []
        for image in images:
            shape = image.shape
            cropped_images.append(image[(shape[0]+1)//2 - 200: (shape[0]+1)//2 + 200,
                          (shape[1]+1)//2 - 200: (shape[1]+1)//2 + 200])

        #normalization
        for i in range(len(cropped_images)):
            cropped_images[i] /= np.sum(cropped_images[i])


        x, y = np.arange(size), np.arange(size)
        X, Y = np.meshgrid(x, y)
        X, Y = X - size // 2, Y - size // 2
        R = (X ** 2 + Y ** 2) ** 0.5
        alpha = 0.3
        csf_lens = 2 * scipy.special.j1(R * alpha)/(R * alpha)
        csf_lens[size//2, size//2] = 1
        psf_lens = csf_lens**2
        psf_lens /= np.sum(psf_lens)

        psf_real = np.pad(psf_lens, 50, mode='constant', constant_values=0)
        image = cropped_images[0]
        image = np.pad(image, 50, mode='constant', constant_values=0)

        blurred = scipy.signal.convolve(image, psf_real, mode='same')
        noised = blurred + skimage.util.random_noise(blurred, 'gaussian', var=10**-10)
        fig, axes = plt.subplots(1, 4)

        rld = Deconvolvers.RichardsonLucy(regularization_filter=10**-15)
        restored, history = rld.deconvolve(noised, psf_real, 100)
        axes[0].imshow(image, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(image))
        axes[1].imshow(noised, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(image))
        axes[2].imshow(restored, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(image))

        noised_os = noised[50:450, 50:450]
        blurred_ft = scipy.fft.fftshift(scipy.fft.fft2(scipy.fft.fftshift(noised_os)))
        otf = np.abs(scipy.fft.fftshift(scipy.fft.fft2(scipy.fft.fftshift(psf_lens))))
        noise_power = 1
        wiener_filtered_ft = otf.conjugate() * blurred_ft / (otf * otf.conjugate() + noise_power)
        wiener_filtered = np.abs(scipy.fft.fftshift(scipy.fft.ifft2(scipy.fft.fftshift(wiener_filtered_ft))))
        wiener_filtered = np.pad(wiener_filtered, 50, mode='constant', constant_values=0)
        axes[3].imshow(wiener_filtered, cmap=plt.get_cmap('gray'), vmin=0)

        plt.show()
    def test_out_of_band_restoration(self):
        img = 1-skimage.io.imread("images/cross.png", as_gray=True)
        size = 400
        # img = np.zeros((size, size))
        # img[size//2, size//2] = 1
        img = img[img.shape[0]//2 - size//2:img.shape[0]//2 + size//2, img.shape[1]//2 - size//2:img.shape[1]//2 + size//2]
        img += 10 ** (-10)
        img /= np.sum(img)
        img_ft = scipy.fft.fftshift(scipy.fft.fft2(scipy.fft.fftshift(img)))
        x, y = np.arange(size), np.arange(size)
        X, Y = np.meshgrid(x, y)
        X, Y = X - size // 2, Y - size // 2
        R = (X ** 2 + Y ** 2) ** 0.5
        alpha = 0.3
        csf_lens = 2 * scipy.special.j1(R * alpha) / (R * alpha)
        csf_lens[size // 2, size // 2] = 1
        psf_lens = csf_lens ** 2
        psf_lens /= np.sum(psf_lens)
        otf = scipy.fft.fftshift(scipy.fft.fft2(scipy.fft.fftshift(psf_lens)))

        blurred = scipy.signal.convolve(img, psf_lens, mode='same')
        rld = Deconvolvers.RichardsonLucy(regularization_filter=10**-8)
        restored, history = rld.deconvolve(blurred, psf_lens, 100)
        fig1, axes = plt.subplots(1, 3)
        axes[0].imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(img))
        axes[1].imshow(blurred, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(blurred))
        axes[2].imshow(restored, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(restored))
        # plt.show()
        mse = np.zeros(len(history))
        fig2, axes2 = plt.subplots(1, figsize=(15, 5))
        slices = np.arange(0, len(mse), 10)
        axes2.axhline(y=1)
        axes2.axvline(200 + 4 * np.pi / alpha)
        axes2.plot(otf[200, 200:], label = str(0))
        for i in range(10):
            imgr_ft = scipy.fft.fftshift(scipy.fft.fft2(scipy.fft.fftshift(history[slices[i]])))
            freqs = np.abs(imgr_ft[200, :]/img_ft[200, :])
            axes2.plot(freqs[200:], label = str(slices[i]))
        # plt.show()
        fig3, axes3 = plt.subplots(1)
        for i in range(len(mse)):
            mse[i] = np.sum((history[i] - img) ** 2) ** 0.5
            if i in slices:
                axes3.plot(i, mse[i], "X", color='black')
        axes3.plot(mse)
        plt.legend()
        plt.show()

    def test_different_subiterations_number(self):
        # img = 1 - skimage.io.imread("images/cross.png", as_gray=True)
        # img = skimage.io.imread("images/cell.jpg", as_gray=True)
        img = skimage.io.imread("images/Automobile.png", as_gray=True)
        size = 400
        img = img[img.shape[0]//2 - size//2:img.shape[0]//2 + size//2, img.shape[1]//2 - size//2:img.shape[1]//2 + size//2]

        # img = np.zeros((size, size))
        # img[size // 2, size // 2] = 1
        # img += 10 ** (-10)
        img /= np.sum(img)
        x, y = np.arange(size), np.arange(size)
        X, Y = np.meshgrid(x, y)
        X, Y = X - size // 2, Y - size // 2
        R = (X ** 2 + Y ** 2) ** 0.5
        psf_real = np.exp(-R ** 2 / 100)
        psf_real /= np.sum(psf_real)

        psf_guess = np.exp(-R ** 2 / 400)
        psf_guess /= np.sum(psf_guess)

        mask = make_mask_cosine_edge((size, size), 50)
        img *= mask**2

        blurred = scipy.signal.convolve(img, psf_real, mode='same')
        # plt.imshow(blurred, cmap=plt.get_cmap('gray'))
        # plt.show()
        sn = 40
        si = 6
        rldbs = []
        objects, psfs, histories = [], [], []

        for k in range(si):
            rldb = Deconvolvers.RichardsonLucyBlind(k, regularization_filter=10**-10)
            rldbs.append(rldb)
            restored, psfb, history = rldb.deconvolve(blurred, psf_guess, stepNumber=sn)
            objects.append(restored)
            psfs.append(psfb)
            histories.append(history)

        # fig1, axes = plt.subplots(1, 3)
        # axes[0].imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(img))
        # axes[1].imshow(blurred, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(blurred))
        # axes[2].imshow(restored, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(restored))
        # plt.show()
        mse = np.zeros((si, sn))
        fig2, axes2 = plt.subplots(2, si, figsize=(15, 5))
        slices = np.arange(0, len(mse), 10)
        for i in range(si):
            axes2[0][i].set_axis_off()
            axes2[0][i].set_title("K = {}".format(i))
            axes2[0][i].imshow(objects[i], cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(objects[i]))
            axes2[1][i].set_axis_off()
            axes2[1][i].imshow(psfs[i], cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(psfs[i]))
        fig2.savefig(path + 'diff_si_car.png')
        # plt.show()
        fig3, axes3 = plt.subplots(1)
        for i in range(si):
            for j in range(sn):
                mse[i, j] = np.sum((img - histories[i][1][j])**2)
            # axes3.plot(mse[i], "X",  color='black')
            axes3.plot(mse[i], '--', label ='{}'.format(i))
        axes3.grid()
        axes3.set_xlim(left=0)
        axes3.set_ylim(bottom=0)
        plt.legend()
        axes3.set_title("Total square error")
        fig3.savefig(path + 'error_diff_si_car.png')
        plt.show()