import PIL.Image as Image
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.signal

def align_images(im1, im2):
    # get rid of the averages, otherwise the results are not good
    im1_0 = im1 - np.mean(im1)
    im2_0 = im2 - np.mean(im2)

    # calculate the correlation image; note the flipping of onw of the images
    corr_11 = scipy.signal.fftconvolve(im1_0, im1_0[::-1,::-1], mode='same')
    corr_12 = scipy.signal.fftconvolve(im1_0, im2_0[::-1,::-1], mode='same')

    d_11 = np.unravel_index(np.argmax(corr_11), corr_11.shape)
    d_12 = np.unravel_index(np.argmax(corr_12), corr_12.shape)

    return np.array(d_11) - np.array(d_12)


def trim_image(image, delta, first_image=True):
    if not first_image:
        delta = -delta

    dx_s, dx_e = (0, -delta[0]) if delta[0] > 0 else (-delta[0], image.shape[0])
    dy_s, dy_e = (0, -delta[1]) if delta[1] > 0 else (-delta[1], image.shape[1])

    return image[dx_s: dx_e, dy_s: dy_e]

if __name__ == "__main__":
    images_folder = "./data/defective_examples"
    images = [["case1_inspected_image.tif", "case1_reference_image.tif"],
              ["case2_inspected_image.tif", "case2_reference_image.tif"]]
    for fns in images:
        fps = [os.path.join(images_folder, f) for f in fns]
        img1 = np.asarray(Image.open(fps[0]), dtype=np.float32) / 255.
        img2 = np.asarray(Image.open(fps[1]), dtype=np.float32) / 255.

        delta = align_images(img1, img2)
        aligned_img1 = trim_image(img1, delta, first_image=True)
        aligned_img2 = trim_image(img2, delta, first_image=False)

        diff_img = aligned_img1 - aligned_img2
        diff_img[np.logical_and(diff_img>-0.075, diff_img < 0.075)] = 0
        plt.figure(); plt.imshow(aligned_img1)
        plt.figure(); plt.imshow(aligned_img2)
        plt.figure(); plt.imshow(diff_img)
