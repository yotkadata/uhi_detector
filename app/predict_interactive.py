# Add the parent directory to the path to make imports work
import os
import sys

module_path = os.path.abspath(os.path.join("../src"))
if module_path not in sys.path:
    sys.path.append(module_path)

import base64
from io import BytesIO
from pathlib import Path

import folium
import numpy as np
import rasterio
import requests
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_folium import folium_static
import utils

file_model = Path(
    "../data/models/landcover_25_epochs_resnet34_backbone_batch16_iou_0.76.hdf5"
)
img_file = Path("../data/app/dop20rgb_32_554_5806_2_ni_2022-05-08_patch_0_4.tif")
img_files = [
    Path("../data/app/dop20rgb_32_554_5806_2_ni_2022-05-08_patch_0_3.tif"),
    Path("../data/app/dop20rgb_32_554_5806_2_ni_2022-05-08_patch_0_4.tif"),
    Path("../data/app/dop20rgb_32_554_5806_2_ni_2022-05-08_patch_0_5.tif"),
]

# Load your trained model (replace with your model)
model = tf.keras.models.load_model(file_model)


# @st.cache(show_spinner=False)
def get_image(img_path: Path):
    target_path = img_path.parent / f"{img_path.stem}_reprojected{img_path.suffix}"
    target_crs = 4326

    if not target_path.is_file():
        src, img = utils.reproject_geotiff(img_path, target_path, target_crs)
    else:
        src = rasterio.open(target_path)
        img = src.read()

    # Move channel axis to third position
    img = np.moveaxis(img, source=0, destination=2)

    return img, src.meta, src.bounds


@st.cache(show_spinner=False)
def get_segmentation(image):
    input_image = (
        np.array(image.resize((224, 224))) / 255.0
    )  # Resize and normalize image (example)
    input_image = np.expand_dims(input_image, axis=0)  # Expand dims to simulate batch
    prediction = model.predict(input_image)[0]  # Make prediction
    return prediction


def image_overlay(map, image, bounds, prediction):
    image_uri = base64.b64encode(image.to_bytes()).decode("utf-8")
    prediction_uri = base64.b64encode(prediction.to_bytes()).decode("utf-8")

    html = (
        '<div style="position: relative;">'
        '<img src="data:image/png;base64,{}" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;">'
        '<img src="data:image/png;base64,{}" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;">'
        "</div>".format(image_uri, prediction_uri)
    )

    iframe = folium.IFrame(html, width=640, height=480)
    popup = folium.Popup(iframe, max_width=2650)

    icon = folium.Icon(color="transparent", icon="info", icon_color="black")
    marker = folium.Marker(
        location=[(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2],
        icon=icon,
        popup=popup,
    )

    map.add_child(marker)


def main():
    st.title("Aerial Image Segmentation")

    # img, img_meta, img_bounds = get_image(img_file)

    # st.write(img_meta)

    # location = [
    #     (img_bounds.top + img_bounds.bottom) / 2,
    #     (img_bounds.left + img_bounds.right) / 2,
    # ]

    location = [52.41853758716104, 9.807577312358024]
    st.write(location)

    # Initialize the map
    m = folium.Map(
        location=location,
        zoom_start=15,
        tiles="Stamen Terrain",
    )

    for img_file in img_files:
        img, img_meta, img_bounds = get_image(img_file)

        # Add the image to the map
        folium.raster_layers.ImageOverlay(
            image=img,
            name="Hannover",
            opacity=0.75,
            bounds=[
                [img_bounds.bottom, img_bounds.left],
                [img_bounds.top, img_bounds.right],
            ],
        ).add_to(m)

    # # Load an image
    # url = "http://your_url.com/path_to_your_image.png"  # Replace with your image URL
    # image = get_image(url)

    # # Display the map
    folium_static(m)

    # # If button is clicked, segment the image and display the result
    # if st.button("Segment Image"):
    #     prediction = get_segmentation(image)
    #     image_overlay(m, image, bounds, prediction)
    #     folium_static(m)


if __name__ == "__main__":
    main()
