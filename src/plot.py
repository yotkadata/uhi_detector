"""
Collection of plotting functions.
"""

from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def plot_image_channels(image: Path) -> None:
    """
    Plot an image alongside its mask.
    """
    # Read image
    img = cv2.imread(str(image))  # 3 channels / spectral bands

    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 10))

    ax1.imshow(img[:, :, 0], cmap="Reds")
    ax1.set_title("Channel 0: Red")

    ax2.imshow(img[:, :, 1], cmap="Greens")
    ax2.set_title("Channel 1: Green")

    ax3.imshow(img[:, :, 2], cmap="Blues")
    ax3.set_title("Channel 2: Blue")

    plt.show()


def plot_image_and_mask(image_path: Path, mask_path: Path) -> None:
    """
    Plot an image alongside its mask.
    """
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path.as_posix())

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))

    ax1.imshow(img)
    ax1.set_title("RGB image")

    norm = mpl.colors.Normalize(vmin=0, vmax=4)
    cmap = plt.get_cmap("viridis")

    ax2.imshow(mask[:, :, 1], cmap=cmap, norm=norm)
    ax2.set_title("Mask")

    plt.show()


def plot_rgb_histogram(band_r, band_g, band_b, bins=255) -> None:
    """
    Plot histograms for the RGB bands.
    """
    _, (axr, axg, axb) = plt.subplots(1, 3, figsize=(15, 5))

    axr.hist(band_r.flatten(), bins=bins, color="red")
    axr.set_title("Red")

    axg.hist(band_g.flatten(), bins=bins, color="green")
    axg.set_title("Green")

    axb.hist(band_b.flatten(), bins=bins, color="blue")
    axb.set_title("Blue")

    plt.show()
