"""
Segment images directly from a map area.
"""
import os
import sys
from pathlib import Path

import folium
import numpy as np
import rasterio
import streamlit as st
from folium.plugins import Draw
from keras.models import load_model
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
    "resnet-34": {
        "description": "Unet ResNet-34",
        "file_name": "landcover_25_epochs_resnet34_backbone_batch16_iou_0.76.hdf5",
    },
    "resnet-50": {
        "description": "Unet ResNet-50",
        "file_name": "landcover_25_epochs_resnet50_backbone_batch16_iou_0.82.hdf5",
    },
}

# API Key from Mapbox
MAPBOX_URL = (
    "https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/{z}/{x}/{y}?access_token="
    + st.secrets["MAPBOX_API_KEY"]
)
IMAGE_PATH = f"data/predict/app/source/satellite-from-leafmap-{np.random.randint(10000, 99999)}.tif"


@st.cache_resource
def load_model_from_file(model_path):
    """
    Load a model from a file.
    """
    model = load_model(model_path, compile=False)
    return model


def create_map(location):
    """
    Create a Folium map with a satellite layer and a drawing tool.
    """
    # Create Folium map
    m = folium.Map(location=location, zoom_start=18)

    # Add the satellite layer
    folium.TileLayer(MAPBOX_URL, attr="Mapbox").add_to(m)

    # Add drawing tool
    Draw(export=True).add_to(m)

    return m


def main():
    st.title("Aerial Image Segmentation")

    # Create a select box for the model
    selected_model = st.selectbox(
        "Model to use",
        MODELS.keys(),
        format_func=lambda x: MODELS[x]["description"],
    )

    # Create Folium map
    m = create_map([52.49442707602441, 13.434820704132562])

    # Render map
    output = st_folium(m, width=700, height=500)

    placeholder = st.empty()

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

                # Save the selection as a GeoTIFF
                tms_to_geotiff(
                    output=IMAGE_PATH,
                    bbox=bbox,
                    zoom=18,
                    source=MAPBOX_URL,
                    overwrite=True,
                )

            # Check if image was created successfully and display it
            if Path(IMAGE_PATH).is_file():
                placeholder.image(IMAGE_PATH, caption="Extracted image")

            if st.button("Segment"):
                with st.spinner("Segmenting image..."):
                    # Read image
                    with rasterio.open(IMAGE_PATH) as dataset:
                        img_array = dataset.read()

                    # Move channel information to third axis
                    img_array = np.moveaxis(img_array, source=0, destination=2)

                    # Get the prediction
                    model = load_model_from_file(
                        Path("data/models", MODELS[selected_model]["file_name"])
                    )

                    # Get prediction
                    prediction = get_smooth_prediction_for_file(
                        img_array, model, 5, "resnet34", patch_size=256
                    )

                    # Prepare images for visualization
                    img, segmented_img, overlay = prepare_split_image(
                        img_array, prediction
                    )

                    # Show image comparison in placeholder container
                    with placeholder.container():
                        image_comparison(
                            img1=img,
                            img2=overlay,
                        )


if __name__ == "__main__":
    main()
