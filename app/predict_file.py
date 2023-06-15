import os
import sys
from pathlib import Path

import branca.colormap as cm
import cv2
import numpy as np
import rasterio
import streamlit as st
from PIL import Image
from rasterio.io import MemoryFile
from streamlit_image_comparison import image_comparison

# Add app to path
module_path = os.path.abspath(os.path.join("src"))
if module_path not in sys.path:
    sys.path.append(module_path)


def main():
    """
    Main function.
    """
    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        st.write("### Urban Heat Island Detector")
        st.title("Segmentation from file")

        st.write("Upload an image file to segment it.")

        with st.spinner("Loading ..."):
            uploaded_file = st.file_uploader("Choose an image file", type="tif")

        placeholder = st.empty()

        if uploaded_file is not None:
            st.write("File uploaded")

            # Display the image
            img = Image.open(uploaded_file)

            # Save the image to a temporary directory
            with MemoryFile(uploaded_file.getvalue()) as memfile:
                with memfile.open() as dataset:
                    img_array = dataset.read()
                    img_meta = dataset.meta

            # Define file path
            input_file_path = Path(f"data/predict/app/source/{uploaded_file.name}")

            # Write the image to the temporary directory
            with rasterio.open(input_file_path, "w", **img_meta) as dst:
                dst.write(img_array)

            # Move channel information to third axis
            img_array = np.moveaxis(img_array, source=0, destination=2)

            # img_array = np.array(img)
            placeholder.image(img, caption="Uploaded Image.", use_column_width=True)

            # Show a button to start the segmentation
            if st.button("Segment"):
                with st.spinner("Segmenting ..."):
                    from predict import get_smooth_prediction_for_file

                    # Get the prediction
                    model_path = Path(
                        "data/models/landcover_25_epochs_resnet34_backbone_batch16_iou_0.76.hdf5"
                    )

                    prediction = get_smooth_prediction_for_file(
                        img_array, model_path, 5, "resnet34", patch_size=256
                    )

                # Map the values from 0-4 to RGBA colors (you can choose any colors)
                # Here, 0 is mapped to transparent
                colors = {
                    0: (0, 0, 0, 0),  # Transparent
                    1: (239, 131, 84, 200),  # Buildings
                    2: (22, 219, 147, 200),  # Trees
                    3: (38, 103, 255, 200),  # Water
                    4: (224, 202, 60, 200),  # Roads
                }

                # Prepare an empty array for the colored image (height x width x 4 for RGBA)
                colored_image = np.zeros((*prediction.shape, 4), dtype=np.uint8)

                # Apply colors
                for val, color in colors.items():
                    colored_image[prediction == val] = color

                # Convert numpy array to PIL image
                segmented_img = Image.fromarray(colored_image)

                segmented_png = f"data/predict/app/prediction/{uploaded_file.name}.png"
                segmented_img.save(segmented_png)

                overlay = Image.alpha_composite(img.convert("RGBA"), segmented_img)

                # Show image comparison in placeholder container
                with placeholder.container():
                    image_comparison(
                        img1=img,
                        img2=overlay,
                    )

                # Define file path
                output_file_path = Path(
                    f"data/predict/app/prediction/{uploaded_file.name}"
                )

                # Add dimension to final_prediction
                final_prediction = np.expand_dims(prediction, axis=0)

                img_meta.update({"count": 1})
                final_prediction = final_prediction * 255 / 4

                # Write the image to the temporary directory
                with rasterio.open(output_file_path, "w", **img_meta) as dst:
                    dst.write(final_prediction)


if __name__ == "__main__":
    main()
