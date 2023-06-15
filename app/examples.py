"""
Show some examples of predictions.
"""
from pathlib import Path

import streamlit as st
from PIL import Image
from streamlit_image_comparison import image_comparison


def make_grid(cols, rows):
    """
    Create a grid of columns and rows.
    """
    grid = [0] * cols
    for i in range(cols):
        with st.container():
            grid[i] = st.columns(rows)
    return grid


def main():
    """
    Main function.
    """

    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        st.write("### Urban Heat Island Detector")
        st.title("Examples of segmentations")

        # Create lists of images and segmentations
        images = [img for img in Path("data/predict/app/source").iterdir()]

        valid_images = []
        segmentations = []

        for image in images:
            segmentation = Path(image.parent.parent, "prediction", image.name + ".png")
            if segmentation.is_file():
                valid_images.append(image)
                segmentations.append(segmentation)

        images = valid_images

        # Show 6 images or less if there are less than 6 images
        n_images = 6 if len(images) > 6 else len(images)

        for i in range(n_images):
            image = Image.open(images[i]).convert("RGBA")
            segmentation = Image.open(segmentations[i])
            overlay = Image.alpha_composite(image, segmentation)

            image_comparison(
                img1=image,
                img2=overlay,
            )

        # st.write(images)
        # st.write(segmentations)
        # grid = make_grid(3, 2)

        # grid[0][0].image(str(images[0]), use_column_width="always")
        # grid[0][1].image(str(images[1]), use_column_width="always")
        # grid[1][0].image(str(images[2]), use_column_width="always")
        # grid[1][1].image(str(images[3]), use_column_width="always")
        # grid[2][0].image(str(images[4]), use_column_width="always")
        # grid[2][1].image(str(images[5]), use_column_width="always")


if __name__ == "__main__":
    main()
