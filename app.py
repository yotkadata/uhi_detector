"""
Streamlit app for the final project.
Building a "Urban Heat Island Detector" using satellite images.
"""

import os
import sys

import streamlit as st

# Add app directory to path
module_path = os.path.abspath(os.path.join("app"))
if module_path not in sys.path:
    sys.path.append(module_path)


@st.cache_data
def display_tools():
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.image("data/presentation/logo_geopandas.png", use_column_width=True)
        st.image("data/presentation/logo_folium.png", use_column_width=True)
        st.image("data/presentation/logo_keras.png", use_column_width=True)

    with col2:
        st.image("data/presentation/logo_tensorflow.png", use_column_width=True)
        st.image("data/presentation/logo_matplot.png", use_column_width=True)
        st.image("data/presentation/logo_numpy.png", use_column_width=True)

    with col3:
        st.image("data/presentation/logo_opencv.png", use_column_width=True)
        st.image("data/presentation/logo_pandas.png", use_column_width=True)
        st.image(
            "data/presentation/logo_segmentation_models.png", use_column_width=True
        )

    with col4:
        st.image("data/presentation/logo_streamlit.png", use_column_width=True)
        st.image("data/presentation/logo_sentinel_2.png", use_column_width=True)
        st.image("data/presentation/logo_landsat_8.png", use_column_width=True)


@st.cache_data
def display_mean_temperatures():
    st.image(
        "data/presentation/berlin-germany-mean-temperature-2022_ref-1942-2022.png",
        use_column_width=True,
    )


st.set_page_config(
    page_title="Urban Heat Island Detector",
    layout="wide",
)

with st.sidebar:
    st.write("### Page selection")
    page_selected = st.radio(
        "Select a page to show",
        (
            "Segmentation from file",
            "Live segementation",
            "Start",
            "Segmentation examples",
            "Maps",
            "Mean Temperatures",
            "Tools",
        ),
    )

if page_selected == "Start":
    st.write("# Urban Heat Island Detector")
    st.markdown(
        """
        Some welcome text. Some welcome text. Some welcome text. Some welcome text. 
        Some welcome text. Some welcome text. Some welcome text. Some welcome text. 
        Some welcome text. Some welcome text. Some welcome text. Some welcome text. 
        Some welcome text. Some welcome text. Some welcome text. 
        """
    )

elif page_selected == "Maps":
    # Show app/landsat-maps.py
    import landsat_maps

    landsat_maps.main()

elif page_selected == "Live segementation":
    import predict_interactive

    predict_interactive.main()

elif page_selected == "Segmentation from file":
    import predict_file

    predict_file.main()

elif page_selected == "Segmentation examples":
    import examples

    examples.main()

elif page_selected == "Mean Temperatures":
    display_mean_temperatures()

elif page_selected == "Tools":
    st.title("Tools used (most) in the project")

    # Create a raster of images with 4 columns and 3 rows
    display_tools()
