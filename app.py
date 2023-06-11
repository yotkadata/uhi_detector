from pathlib import Path

import folium
import numpy as np
import rasterio
import streamlit as st
from streamlit_folium import st_folium

dir_data_root = Path("data/landsat/")
dir_output = Path(dir_data_root, "output")
geojson_path = Path(dir_data_root, "geojson", "berlin.geojson")
file_temperature_repr_colored = Path(dir_output, "temperature_reprojected_colored.tif")

st.set_page_config(layout="wide")
st.title("Urban Heat Island Detector")

with rasterio.open(file_temperature_repr_colored) as colored_img:
    # Load GeoTIFF as ndarray
    colored_img_array = colored_img.read()

    # Move channel axis to third position
    colored_img_array = np.moveaxis(colored_img_array, source=0, destination=2)

m = folium.Map(
    location=[
        (colored_img.bounds.bottom + colored_img.bounds.top) / 2,
        (colored_img.bounds.left + colored_img.bounds.right) / 2,
    ],
    tiles="Stamen Terrain",
    zoom_start=12,
)
folium.raster_layers.ImageOverlay(
    image=colored_img_array,
    name="Land Surface Temperature",
    opacity=0.75,
    bounds=[
        [colored_img.bounds.bottom, colored_img.bounds.left],
        [colored_img.bounds.top, colored_img.bounds.right],
    ],
).add_to(m)

folium.LayerControl().add_to(m)

st_data = st_folium(m, width=1200, height=1200 / 16 * 9)
