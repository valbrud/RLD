import unittest
import numpy as np
import skimage
import Deconvolvers
import matplotlib.pyplot as plt
import scipy


class TestNotBlind(unittest.TestCase):
    def test_vs_skimage(self):
        img = 1 - skimage.io.imread("images/cross.png", as_gray=True)
        sizex, sizey = img.shape
        img += 10 ** (-10)
        img /= np.sum(img)
        x, y = np.arange(sizex), np.arange(sizey)
        X, Y = np.meshgrid(x, y)
        X, Y = X - sizex // 2, Y - sizey // 2
        R = (X ** 2 + Y ** 2) ** 0.5
        psf_real = np.exp(-R ** 2 / 900)
        psf_real /= np.sum(psf_real)
        blured = scipy.signal.convolve(img, psf_real, mode='same')
        rld = Deconvolvers.RichardsonLucy()
        restored, history = rld.deconvolve(blured, psf_real, 20)
        fig, axes = plt.subplots(1, 4)
        restored_default = skimage.restoration.richardson_lucy(blured, psf_real, num_iter=20, filter_epsilon=10 ** -10)
        axes[0].imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(img))
        axes[1].imshow(blured, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(blured))
        axes[2].imshow(restored, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(restored))
        axes[3].imshow(restored_default, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(restored_default))
        plt.show()

    def test_convergence(self):
        img = 1 - skimage.io.imread("images/cross.png", as_gray=True)
        sizex, sizey = img.shape
        img += 10 ** (-10)
        img /= np.sum(img)
        x, y = np.arange(sizex), np.arange(sizey)
        X, Y = np.meshgrid(x, y)
        X, Y = X - sizex // 2, Y - sizey // 2
        R = (X ** 2 + Y ** 2) ** 0.5
        psf_real = np.exp(-R ** 2 / 900)
        psf_real /= np.sum(psf_real)
        blured = scipy.signal.convolve(img, psf_real, mode='same')
        rld = Deconvolvers.RichardsonLucy()
        restored, history = rld.deconvolve(blured, psf_real, 100)
        fig1, axes = plt.subplots(1, 3)
        axes[0].imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(img))
        axes[1].imshow(blured, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(blured))
        axes[2].imshow(restored, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(restored))
        # plt.show()
        mse = np.zeros(len(history[1]))
        fig2, axes2 = plt.subplots(2, 5, figsize=(15,5))
        slices = np.arange(0, len(mse), 10)
        for i in range(10):
            axes2[i//5, i % 5].imshow(history[slices[i]])
        # plt.show()
        fig3, axes3 = plt.subplots(1)
        for i in range(len(mse)):
            mse[i] = np.sum((history[i] - img) ** 2) ** 0.5
            if i in slices:
                axes3.plot(i, mse[i], "X")
        plt.plot(mse)
        plt.show()
    def test_different_objects(self): ...

    def test_different_blur(self): ...

    def test_different_noise_levels(self): ...

    def test_vs_wiener(self): ...

class TestBlind(unittest.TestCase):
    def test_vs_not_blind(self): ...

    def test_convergence(self): ...

    def test_different_objects(self): ...

    def test_different_blur(self): ...

    def test_different_noise_levels(self): ...

    def test_vs_wiener(self): ...