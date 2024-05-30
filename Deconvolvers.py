import numpy as np
from abc import abstractmethod
import scipy
import skimage
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

from scipy.signal import fftconvolve


class Deconvolver:
    @abstractmethod
    def deconvolve(self, image, psf, step_number):...


class RichardsonLucy(Deconvolver):
    def __init__(self, storeIntermediate=True, regularization_filter=10**-10):
        self.storeIntermediate = storeIntermediate
        self.regularization_fitler = regularization_filter

    def deconvolve(self, image, psf, stepNumber=10):
        f0 = np.array(image)
        size = np.array(image.shape)

        history = np.zeros((stepNumber, *size)) if self.storeIntermediate else None

        f = f0
        g = psf
        for step in range(stepNumber):
            if self.storeIntermediate:
                history[step] = f
            inner_convolution = scipy.signal.convolve(g, f, 'same')
            f = scipy.signal.convolve(image / (inner_convolution + self.regularization_fitler), np.flip(g), 'same') * f
            f_ft = np.abs(scipy.fft.fftshift(scipy.fft.fft2(scipy.fft.fftshift(f))))

        object_estimated = f

        return object_estimated, history

class RichardsonLucyBlind(Deconvolver):
    def __init__(self, stepsPerIteration, storeIntermediate=True, regularization_filter=10**-10):
        self.stepsPerIteration = stepsPerIteration
        self.storeIntermediate = storeIntermediate
        self.regularization_fitler = regularization_filter

    def _iterate_psf(self, image, object, psf):
        c, f, g, = image, object, psf
        for step in range(self.stepsPerIteration):
            inner_convolution = scipy.signal.convolve(g, f, 'same')
            g = scipy.signal.convolve(c / (inner_convolution + self.regularization_fitler), np.flip(f), 'same') * g

        return g
    def _iterate_object(self, image, object, psf):
        c, f, g, = image, object, psf
        for step in range(self.stepsPerIteration):
            inner_convolution = scipy.signal.convolve(g, f, 'same')
            f = scipy.signal.convolve(c / (inner_convolution + self.regularization_fitler), np.flip(g), 'same') * f
        return f

    def deconvolve(self, image, psf=None, stepNumber=10, regularization_filter=10**-10):
        f0 = np.array(image)
        size = np.array(image.shape)

        history = np.zeros((2, stepNumber, *size)) if self.storeIntermediate else None

        if not psf is None:
            g0 = psf
        else:
            g0 = np.zeros(size)
            g0[:, :] = 1/g0.size

        f = f0
        g = g0
        for step in range(stepNumber):
            if self.storeIntermediate:
                history[0, step] = g
                history[1, step] = f
            g = self._iterate_psf(image, f, g)
            f = self._iterate_object(image, f, g)
        psf_estimated = g
        object_estimated = f

        return object_estimated, psf_estimated, history




if __name__ == "__main__":
    # img = skimage.io.imread("images/cross.png", as_gray=True)
    size = 511
    length = 50
    width = 5
    img = np.zeros((size, size))
    img[size//2 - length +1: size//2 + 1 +length, size//2+1-width:size//2+1 + width] = 1
    img[size//2+1-width:size//2+1 + width, size//2 - length +1: size//2 + 1 +length] = 1
    # img = 1 - img + 0.001
    img+=10**(-10)
    img/=np.sum(img)
    # print(img.shape)
    x, y = np.arange(size), np.arange(size)
    X, Y = np.meshgrid(x, y)
    X, Y = X - size//2, Y - size//2
    R = (X**2 + Y**2)**0.5
    psf_real = np.exp(-R**2/900)
    # fig1, axes1 = plt.subplots(1, 2)
    psf_real /= np.sum(psf_real)
    blured = scipy.signal.convolve(img, psf_real, mode='same')
    blured_ft = np.abs(np.fft.fftshift(scipy.fft.fft2(scipy.fft.fftshift(blured))))
    # axes1[0].imshow(blured)
    # axes1[1].imshow(np.log(blured_ft))
    # plt.show()
    psf0 = np.exp(-R**2/300)
    psf0 /= np.sum(psf0)
    # plt.imshow(psf0)
    # plt.show()
    rld = RichardsonLucyBlind(5)
    restored, psf_estimated, history = rld.deconvolve(blured, psf0, 40)
    fig, axes = plt.subplots(1, 4)
    restored_default = skimage.restoration.richardson_lucy(blured, psf0, num_iter=200, filter_epsilon=10**-10)
    axes[0].imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(img))
    axes[1].imshow(blured, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(blured))
    axes[2].imshow(restored, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(restored))
    axes[3].imshow(restored_default, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(restored_default))
    plt.show()
    mse = np.zeros(len(history[1]))
    for i in range(len(mse)):
        mse[i] = np.sum((history[1][i] - img)**2)**0.5
    plt.plot(mse)
    plt.show()