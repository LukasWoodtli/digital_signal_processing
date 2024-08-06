import os
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import scipy.signal
from imageio.v3 import imread, imwrite

image = Path(__file__).resolve().parent / "image_processing_filters.png"

def map_values(x):
    WHITE = 0
    GRAY = 127
    BLACK = 255
    if x == 0:
        return GRAY
    if x < 0:
        return BLACK
    return WHITE


def test_read_image():
    im = imread(image, pilmode="L")

    filters = [np.ones((3, 3)) / 9.,
               np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
               np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
               np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
               np.array([[0, -1, 0], [-1, 0, 1], [0, 1, 0]]),
               np.array([[0, 1, 0], [-1, 0, 1], [0, -1, 0]]),
               ]

    map = np.vectorize(map_values)

    for index, filter in enumerate(filters, start=1):
        origin = [int(d / 2.) for d in filter.shape]
        res = scipy.ndimage.convolve(im, filter, mode='constant', origin=origin)
        plt.imshow(res, norm='linear', cmap='gray', vmin=0, vmax=255)

        #pyplot.show()

        dir_path = Path(__file__).parent.absolute() / "image_processing_plots"
        os.makedirs(dir_path, exist_ok=True)
        file_path = dir_path / f"f{index}.png"
        plt.savefig(file_path)
