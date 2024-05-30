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
        blurred = scipy.signal.convolve(img, psf_real, mode='same')
        rld = Deconvolvers.RichardsonLucy()
        restored, history = rld.deconvolve(blurred, psf_real, 20)
        fig, axes = plt.subplots(1, 4)
        restored_default = skimage.restoration.richardson_lucy(blurred, psf_real, num_iter=20, filter_epsilon=10 ** -10)
        axes[0].imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(img))
        axes[1].imshow(blurred, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(blurred))
        axes[2].imshow(restored, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(restored))
        axes[3].imshow(restored_default, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(restored_default))
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
        rld = Deconvolvers.RichardsonLucy(regularization_filter=10**-15)
        restored, history = rld.deconvolve(blurred, psf_real, 100)
        fig1, axes = plt.subplots(1, 3)
        axes[0].imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(img))
        axes[1].imshow(blurred, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(blurred))
        axes[2].imshow(restored, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(restored))
        # plt.show()
        mse = np.zeros(len(history))
        fig2, axes2 = plt.subplots(2, 5, figsize=(15, 5))
        slices = np.arange(0, len(mse), 10)
        for i in range(10):
            axes2[i//5, i % 5].imshow(history[slices[i]])
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
        psf_real = np.pad(psf_real, 50, mode='constant', constant_values=0)

        for i in range(len(cropped_images)):
            image = cropped_images[i]
            image = np.pad(image, 50, mode='constant', constant_values=0)
            blurred = scipy.signal.convolve(image, psf_real, mode='same')
            rld = Deconvolvers.RichardsonLucy(regularization_filter=10**-15)
            restored, history = rld.deconvolve(blurred, psf_real, 100)
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
        alpha = 0.1
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
        plt.imshow(psf_motion)
        plt.show()
        psf_motion /= np.sum(psf_motion)

        blurs = [psf_lens, psf_gaussian, psf_motion]

        fig, axes = plt.subplots(len(cropped_images), 3)
        car = np.pad(car, 50, mode='constant', constant_values=0)

        for i in range(len(blurs)):
            blur = blurs[i]
            blur = np.pad(blur, 50, mode='constant', constant_values=0)
            blurred = scipy.signal.convolve(car, blur, mode='same')
            rld = Deconvolvers.RichardsonLucy(regularization_filter=10**-15)
            restored, history = rld.deconvolve(blurred, blur, 100)
            axes[i][0].imshow(car, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(car))
            axes[i][1].imshow(blurred, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(car))
            axes[i][2].imshow(restored, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(car))
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
        noised2 = skimage.util.random_noise(blurred, 's&p', amount = 0.0005)
        noised = [noised1, noised2]
        fig, axes = plt.subplots(len(noised), 3)

        for i in range(len(noised)):
            bn = noised[i]
            rld = Deconvolvers.RichardsonLucy(regularization_filter=10**-15)
            restored, history = rld.deconvolve(bn, psf_real, 100)
            axes[i][0].imshow(image, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(image))
            axes[i][1].imshow(bn, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(image))
            axes[i][2].imshow(restored, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(image))
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
        img = skimage.io.imread("images/Automobile.png", as_gray=True)
        size = 400
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

class TestBlind(unittest.TestCase):
    def test_vs_not_blind(self): ...

    def test_convergence(self): ...

    def test_different_objects(self): ...

    def test_different_blur(self): ...

    def test_different_noise_levels(self): ...

    def test_vs_wiener(self): ...