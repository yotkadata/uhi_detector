"""
Streamlit app page to display Landsat maps.
"""
from pathlib import Path

import folium
import numpy as np
import rasterio
import streamlit as st
from streamlit_folium import st_folium, folium_static


@st.cache_data
def load_raster(file_path):
    """
    Load GeoTIFF as ndarray.
    """
    with rasterio.open(file_path) as map_layer:
        # Load GeoTIFF as ndarray
        map_layer_array = map_layer.read()
        map_layer_bounds = map_layer.bounds

        # Move channel axis to third position
        map_layer_array = np.moveaxis(map_layer_array, source=0, destination=2)

    return map_layer_array, map_layer_bounds


def create_overlay(file_path, map, name):
    """
    Create overlay layer for map.
    """

    map_layer_array, map_layer_bounds = load_raster(file_path)

    folium.raster_layers.ImageOverlay(
        image=map_layer_array,
        name=name,
        opacity=0.75,
        bounds=[
            [map_layer_bounds.bottom, map_layer_bounds.left],
            [map_layer_bounds.top, map_layer_bounds.right],
        ],
    ).add_to(map)


def create_map(raster_path, names):
    """
    Create map with overlay.
    """
    _, map_layer_bounds = load_raster(raster_path)

    bounds = [
        (map_layer_bounds.bottom + map_layer_bounds.top) / 2,
        (map_layer_bounds.left + map_layer_bounds.right) / 2,
    ]

    m = folium.Map(
        # location=bounds,
        tiles="Stamen Terrain",
        # zoom_start=12,
    )

    m.fit_bounds(
        [
            [map_layer_bounds.bottom, map_layer_bounds.left],  # south-west
            [map_layer_bounds.top, map_layer_bounds.right],  # north-east
        ]
    )

    create_overlay(raster_path, m, names[raster_path.stem])

    folium.LayerControl().add_to(m)

    return m


def main():
    """
    Main function.
    """
    # Berlin
    # id = "LC09_L1TP_193023_20220624_20230409_02_T1"
    # Wolfsburg
    # id = "LC08_L1TP_194024_20230610_20230614_02_T1"
    # Caracas
    # id = "LC09_L1TP_004053_20221205_20230318_02_T1"
    # Istanbul
    id = "LC09_L1TP_179032_20230217_20230310_02_T1"

    dir_data_root = Path("data")
    dir_raster = dir_data_root / "raster_files"
    geojson_path = Path("data/geojson/berlin.geojson")
    file_lst = Path(dir_raster, id + "_lst_repr_colored.tif")
    file_ndvi = Path(dir_raster, id + "_ndvi_repr_colored.tif")
    file_emissivity = Path(dir_raster, id + "_emissivity_repr_colored.tif")

    names = {
        file_lst.stem: "Land Surface Temperature",
        file_ndvi.stem: "Normalized Difference Vegetation Index",
        file_emissivity.stem: "Emissivity",
    }

    st.title("Urban Heat Island Detector")

    map_selected = st.radio(
        "Select a map to show",
        (
            "Land Surface temperature",
            "Vegetation Index (NDVI)",
            "Emissivity",
        ),
        horizontal=True,
    )

    if map_selected == "Land Surface temperature":
        m = create_map(file_lst, names)

    if map_selected == "Vegetation Index (NDVI)":
        m = create_map(file_ndvi, names)

    if map_selected == "Emissivity":
        m = create_map(file_emissivity, names)

    st_folium(m, width=1200, height=1200 / 16 * 9)


if __name__ == "__main__":
    main()
