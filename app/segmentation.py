"""
Segment images directly from a map area.
"""
import os
import sys
import time
from pathlib import Path

import folium
import numpy as np
import rasterio
import streamlit as st
from folium.plugins import Draw
from keras.models import load_model
from PIL import Image
from rasterio.io import MemoryFile
from samgeo import tms_to_geotiff
from streamlit_folium import st_folium
from streamlit_image_comparison import image_comparison

# Add src directory to path
module_path = os.path.abspath(os.path.join("src"))
if module_path not in sys.path:
    sys.path.append(module_path)

from predict import get_smooth_prediction_for_file
from utils import prepare_split_image

MODELS = {
    "resnet34": {
        "description": "Unet ResNet-34",
        "file_name": "landcover_50_epochs_resnet34_backbone_batch16_freeze_iou_0.78.hdf5",
        "backbone": "resnet34",
    },
    "resnet50": {
        "description": "Unet ResNet-50",
        "file_name": "landcover_25_epochs_resnet50_backbone_batch16_freeze_iou_0.86.hdf5",
        "backbone": "resnet50",
    },
}

# API Key from Mapbox
MAPBOX_URL = (
    "https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/{z}/{x}/{y}?access_token="
    + st.secrets["MAPBOX_API_KEY"]
)


@st.cache_resource
def load_model_from_file(model_path):
    """
    Load a model from a file.
    """
    model = load_model(model_path, compile=False)
    return model


def create_map(location, zoom_start=18):
    """
    Create a Folium map with a satellite layer and a drawing tool.
    """
    # Create Folium map
    folium_map = folium.Map(location=location, zoom_start=zoom_start)

    # Add the satellite layer
    folium.TileLayer(MAPBOX_URL, attr="Mapbox").add_to(folium_map)

    # Add drawing tool
    Draw(export=True).add_to(folium_map)

    return folium_map


@st.cache_data
def save_segmented_file(segmented_img, source_path, selected_model):
    """
    Save a segmented image to a png file.
    """
    segmented_png_path = (
        source_path.parent.parent
        / "prediction"
        / f"{source_path.stem}_{selected_model}.png"
    )

    # Make sure image path exists
    Path(segmented_png_path).parent.mkdir(parents=True, exist_ok=True)

    segmented_img.save(segmented_png_path)


@st.cache_data
def show_prediction(image_path, selected_model):
    """
    Get and show the prediction for a given image.
    """
    # Read image
    with rasterio.open(image_path) as dataset:
        img_array = dataset.read()

    # Move channel information to third axis
    img_array = np.moveaxis(img_array, source=0, destination=2)

    # Load model
    model = load_model_from_file(
        Path("data/models", MODELS[selected_model]["file_name"])
    )

    # Get prediction
    prediction = get_smooth_prediction_for_file(
        img_array, model, 5, MODELS[selected_model]["backbone"], patch_size=256
    )

    # Prepare images for visualization
    img, segmented_img, overlay = prepare_split_image(img_array, prediction)

    # Save segmented image
    save_segmented_file(segmented_img, image_path, selected_model)

    return img, segmented_img, overlay


def tab_live_segmentation():
    """
    Streamlit app page to segment images directly from a map area.
    """
    st.title("Live Segmentation")

    col1, col2 = st.columns([1, 1])

    with col1:
        placeholder = st.empty()

    with col2:
        # Create a select box for the model
        selected_model = st.selectbox(
            "Model to use",
            MODELS.keys(),
            format_func=lambda x: MODELS[x]["description"],
            key="model_select_live",
        )

        # location = [52.4944, 13.4348]  # Berlin
        location = [52.4175, 10.7440]  # Wolfsburg

        # Create Folium map
        folium_map = create_map(location, zoom_start=19)

        # Render map
        output = st_folium(folium_map, width=800, height=500)

        if output["all_drawings"] is not None:
            # Create image from bounding box
            if (
                len(output["all_drawings"]) > 0
                and output["all_drawings"][0]["geometry"]["type"] == "Polygon"
            ):
                with st.spinner("Extracting image..."):
                    # Get the bounding box of the drawn polygon
                    bbox = output["all_drawings"][0]["geometry"]["coordinates"][0]

                    # Convert for further use [xmin, ymin, xmax, ymax]
                    bbox = [bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]]

                    image_path = (
                        "data/predict/app/source/satellite-from-leafmap"
                        f"-{str(time.time()).replace('.', '-')}.tif"
                    )

                    # Make sure image path exists
                    Path(image_path).parent.mkdir(parents=True, exist_ok=True)

                    # Save the selection as a GeoTIFF
                    tms_to_geotiff(
                        output=image_path,
                        bbox=bbox,
                        zoom=18,
                        source=MAPBOX_URL,
                        overwrite=True,
                    )

                # Check if image was created successfully and display it
                if Path(image_path).is_file():
                    placeholder.image(image_path, caption="Extracted image", width=700)

                if st.button("Segment", key="segment_button_live"):
                    with st.spinner("Segmenting image..."):
                        img, _, overlay = show_prediction(
                            Path(image_path), selected_model
                        )

                        # Show image comparison in placeholder container
                        with placeholder.container():
                            image_comparison(img1=img, img2=overlay, width=700)


def tab_segmentation_from_file():
    """
    Page to segment images from a file.
    """
    col1, col2, col3 = st.columns([1, 3, 1])

    with col1:
        # Create a select box for the model
        selected_model = st.selectbox(
            "Model to use",
            MODELS.keys(),
            format_func=lambda x: MODELS[x]["description"],
            key="model_select_file",
        )

    with col2:
        st.title("Segmentation from file")

        with st.spinner("Loading ..."):
            uploaded_file = st.file_uploader(
                "Upload an image file to segment it:",
                type=["tif", "tiff", "jpg", "jpeg", "png"],
            )

        placeholder = st.empty()

        if uploaded_file is not None:
            # Open the image from memory
            with MemoryFile(uploaded_file.getvalue()) as memfile:
                with memfile.open() as dataset:
                    img_array = dataset.read()
                    img_meta = dataset.meta

            # Define file path
            input_file_path = Path(f"data/predict/app/source/{uploaded_file.name}")

            # Write the image to the directory
            with rasterio.open(input_file_path, "w", **img_meta) as dst:
                dst.write(img_array)

            # Move channel information to third axis
            img_array = np.moveaxis(img_array, source=0, destination=2)

            # Display the image in the placeholder container
            placeholder.image(
                img_array, caption="Uploaded Image", use_column_width=True
            )

            # Show a button to start the segmentation
            if st.button("Segment", key="segment_button_file"):
                with st.spinner("Segmenting ..."):
                    img, _, overlay = show_prediction(input_file_path, selected_model)

                    # Show image comparison in placeholder container
                    with placeholder.container():
                        image_comparison(
                            img1=img,
                            img2=overlay,
                        )

    with col3:
        st.write("")


# @st.cache_data
def tab_show_examples():
    """
    Page to show some example images.
    """
    _, col2, _ = st.columns([1, 3, 1])

    with col2:
        st.title("Examples of segmentations")

        # Create lists of images and segmentations
        image_paths = list(Path("data/predict/examples/source").iterdir())

        for model_key, model_values in MODELS.items():
            valid_images = []
            segmentations = []

            # Get images that have a source and a prediction file
            for image_path in image_paths:
                segmentation_path = (
                    image_path.parent.parent
                    / "prediction"
                    / f"{image_path.stem}_{model_key}.png"
                )

                # Skip if there is no segmentation file
                if not segmentation_path.is_file():
                    continue

                # Load image
                image = Image.open(image_path).convert("RGBA")

                # Skip if image is too small
                if image.size[0] < 700:
                    continue

                # Load segmentation
                segmentation = Image.open(segmentation_path).convert("RGBA")

                # Append to lists
                valid_images.append(image)
                segmentations.append(segmentation)

            # Show 6 images, or less if there are less than 6 images
            n_images = 6 if len(valid_images) > 6 else len(valid_images)

            if n_images > 0:
                st.subheader(model_values["description"])

            for i in range(n_images):
                # image = Image.open(valid_images[i]).convert("RGBA")
                # segmentation = Image.open(segmentations[i]).convert("RGBA")
                # overlay = Image.alpha_composite(image, segmentation)
                overlay = Image.alpha_composite(valid_images[i], segmentations[i])

                image_comparison(
                    img1=valid_images[i],
                    img2=overlay,
                    width=800,
                )


def tab_video():
    """
    Page to show a video as an example.
    """
    with open("data/presentation/screencast-live-segmentation.mp4", "rb") as video_file:
        video_bytes = video_file.read()

    st.video(video_bytes)


def main():
    """
    Main function.
    """
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Live segmentation", "From file", "Examples", "Video"]
    )

    with tab1:
        tab_live_segmentation()

    with tab2:
        tab_segmentation_from_file()

    with tab3:
        tab_show_examples()

    with tab4:
        tab_video()


if __name__ == "__main__":
    main()
