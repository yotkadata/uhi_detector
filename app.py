"""
Streamlit app for the final project.
SPICED Academy Data Science Bootcamp 2023.
Building a "Urban Heat Island Detector" using satellite images.
By Jan Kühn, https://yotka.org
"""

import os
import sys

import streamlit as st
from streamlit_image_comparison import image_comparison

# Add app directory to path
module_path = os.path.abspath(os.path.join("app"))
if module_path not in sys.path:
    sys.path.append(module_path)

import landsat_maps
import segmentation


# @st.cache_data
def display_image(img_path, caption=None, width=None):
    """
    Display image from path.
    """
    if width:
        st.image(img_path, caption=caption, width=width)
    else:
        st.image(img_path, use_column_width=True, caption=caption)


@st.cache_data
def page_start():
    """
    Display start page.
    """
    display_image("data/presentation/heat_island_wolfsburg_slice.png")

    st.markdown(
        """
        <br /><br />
        <h1 style="text-align: center;">Urban Heat Island Detector</h1>
        <h2 style="text-align: center;">Final project for the Data Science Bootcamp</h2>
        <h3 style="text-align: center;">By Jan Kühn, June 2023</h3>
        <h3 style="text-align: center;"><em>Bergamot Encoders at SPICED Academy, Berlin</em></h3>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def page_what():
    """
    Display page for "What?" section.
    """
    st.markdown("<br /><br /><br />", unsafe_allow_html=True)
    st.title("What?")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(
            """
            <ul class="presentation-bullets">
                <li><strong>Urban Heat Island (UHI) effect:</strong> 
                Increased temperatures in urban areas compared to rural surroundings</li>
                <li><strong>Causes:</strong> Human activities, 
                changes in land use, built infrastructure</li>
                <li><strong>Color factor:</strong> Dark surfaces/roofing materials, 
                known to absorb more solar radiation</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        image_comparison(
            img1="data/presentation/uhi_roofs_black.png",
            img2="data/presentation/uhi_roofs_white.png",
            label1="",
            label2="",
            starting_position=99,
            width=700,
        )
        st.write("Source: c40knowledgehub.org")


@st.cache_data
def page_why():
    """
    Display page for "Why?" section.
    """
    col1, col2, col3 = st.columns([1, 12, 1])

    with col1:
        st.write("")

    with col2:
        st.title("Why is tackling UHI important?")

        st.markdown(
            """
            <ul class="presentation-bullets">
                <li>UHI effects <strong>exacerbate impacts of
                    climate change</strong></li>
                <li>Leads to <strong>increased energy consumption</strong> and
                    <strong>impaired air quality</strong></li>
                <li>Adverse health effects: <strong>heat-related
                    illnesses and mortality</strong></li>
            </ul>
            <br />
            <br />
            """,
            unsafe_allow_html=True,
        )

        display_image("data/presentation/euromomo-charts-pooled-by-age-group.png")

    with col3:
        st.write("")


@st.cache_data
def display_mean_temperatures():
    """
    Display mean temperatures image.
    """
    _, col2, _ = st.columns([1, 20, 1])

    with col2:
        st.title("Mean Temperatures in Berlin, Germany 2022")
        st.markdown("### Compared to Historical Daily Mean Temperatures (1942-2022)")
        display_image(
            "data/presentation/berlin-germany-mean-temperature-2022_ref-1942-2022_headless.png"
        )


@st.cache_data
def page_goal():
    """
    Display page for "Why?" section.
    """
    col1, col2, col3, col4 = st.columns([1, 6, 3, 1])

    with col1:
        st.write("")

    with col2:
        st.title("Goal")

        st.markdown(
            """
            <ul class="presentation-bullets">
                <li>Use satellite imagery to <strong>detect and visualize UHI effects</strong></li>
                <li>Build and train a <strong>deep learning model to segment building roofs</strong> 
                in urban areas from high-resolution satellite imagery</li>
                <li>Incorporate Landsat 8 <strong>Thermal Infrared Sensor (TIRS)</strong> data</li>
                <li>Identify spots for <strong>public interventions</strong> (green roofs, 
                    change of roof material/color, solar panels)</li>
            </ul>
            <br />
            """,
            unsafe_allow_html=True,
        )

    with col3:
        display_image("data/presentation/logo_sentinel_2.png", width=300)
        display_image("data/presentation/logo_landsat_8.png", width=300)

    with col4:
        st.write("")


@st.cache_data
def page_how():
    """
    Display page for "How?" section.
    """
    col1, col2, col3 = st.columns([4, 0.3, 7])

    with col1:
        display_image("data/presentation/unet-model.png", width=600)
        st.write("")
        st.write("")
        st.write("")
        display_image(
            "data/presentation/unet-architecture-building-extraction.jpg", width=600
        )

    with col2:
        st.write("")

    with col3:
        st.title("How is it done?")

        st.markdown(
            """
            <ul class="presentation-bullets">
                <li>Using a <strong>pre-trained Unet segmentation model</strong> with transfer
                    learning based on <strong>resnet-34 / resnet-50 backbone</strong>
                    and <strong>imagenet weights</strong></li>
                <li><strong>Training dataset:</strong> Landcover.ai with 40
                    high resolution labeled images (~40.000 patches of 256x256px)</li>
                <li>Selected <strong>2.000 relevant patches</strong> for training</li>
                <li>Used metric: <strong>Mean IoU</strong> (Intersection over Union)</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("# Best score: 0.86 = 86%")


@st.cache_data
def building_footprints():
    """
    Display building footprints.
    """
    # Add margins using columns
    _, main_col, _ = st.columns([1, 5, 1])

    with main_col:
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            ["Satellite", "Buildings", "Luminance", "NDVI", "Emissivity", "LST"]
        )

        with tab1:
            st.markdown("# Satellite image")
            display_image("data/presentation/berlin_satellite.jpg")

        with tab2:
            st.markdown("# Building footprints")
            display_image("data/presentation/berlin_buildings.png")

        with tab3:
            st.markdown("# Luminance")
            display_image("data/presentation/berlin_luminance.png")

        with tab4:
            st.markdown("# Vegetation Index (NDVI)")
            display_image("data/presentation/berlin_ndvi_colored.jpg")
            st.markdown("### Calculated from Landsat 8 Bands 4 & 5")

        with tab5:
            st.markdown("# Emissivity")
            display_image("data/presentation/berlin_emissivity_colored.jpg")
            st.markdown("### Calculated from Landsat 8 Bands 4 & 5")

        with tab6:
            st.markdown("# LST: Land Surface Temperature")
            display_image("data/presentation/berlin_lst_colored.jpg")
            st.markdown("### Calculated from Landsat 8 Bands 4, 5 & 10")


@st.cache_data
def page_future():
    """
    Display page "Future work".
    """
    col1, col2 = st.columns([3, 2])

    with col1:
        st.title("Future work")
        st.markdown(
            """
            <ul class="presentation-bullets">
                <li>Create <strong>location-based warnings</strong> of UHI effects incorporating
                    weather/temperature forecast data</li>
                <li>Users could include <strong>personal health information</strong> to display
                    personalized warnings especially for vulnerable groups</li>
                <li>Combine with <strong>demographic data</strong> to detect areas with most
                    need for public interventions to mitigate UHI effects</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        display_image("data/presentation/berlin-lst.png", width=600)


@st.cache_data
def display_tools():
    """
    Display tools used in the project.
    """
    st.title("Thank you Bergamot Encoders!")

    st.markdown("""### ... and Rakib & Parvin & Carmine & all the others!""")

    st.markdown(
        "<br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br />",
        unsafe_allow_html=True,
    )

    st.markdown("### Tools used (most) in the project")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.image("data/presentation/logo_geopandas.png", use_column_width=True)
        st.image("data/presentation/logo_folium.png", use_column_width=True)

    with col2:
        st.image("data/presentation/logo_tensorflow.png", use_column_width=True)
        st.image("data/presentation/logo_matplot.png", use_column_width=True)

    with col3:
        st.image("data/presentation/logo_opencv.png", use_column_width=True)
        st.image("data/presentation/logo_pandas.png", use_column_width=True)

    with col4:
        st.image("data/presentation/logo_streamlit.png", use_column_width=True)
        st.image("data/presentation/logo_sentinel_2.png", use_column_width=True)

    with col5:
        st.image("data/presentation/logo_keras.png", use_column_width=True)
        st.image("data/presentation/logo_numpy.png", use_column_width=True)

    with col6:
        st.image(
            "data/presentation/logo_segmentation_models.png", use_column_width=True
        )
        st.image("data/presentation/logo_landsat_8.png", use_column_width=True)


st.set_page_config(
    page_title="Urban Heat Island Detector",
    layout="wide",
)

with open("app/style.css", encoding="utf-8") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

with st.sidebar:
    page_selected = st.radio(
        "Pages",
        (
            "Start",
            "What?",
            "Why?",
            "Heatwaves",
            "Goal",
            "How?",
            "Segment",
            "Footprints",
            "Maps",
            "Future",
            "Finish",
        ),
    )
    display_image("data/presentation/logo_spiced_t.png")

if page_selected == "Start":
    page_start()

elif page_selected == "What?":
    page_what()

elif page_selected == "Why?":
    page_why()

elif page_selected == "Heatwaves":
    display_mean_temperatures()

elif page_selected == "Goal":
    page_goal()

elif page_selected == "How?":
    page_how()

elif page_selected == "Segment":
    segmentation.main()

elif page_selected == "Footprints":
    building_footprints()

elif page_selected == "Maps":
    landsat_maps.main()

elif page_selected == "Future":
    page_future()

elif page_selected == "Finish":
    display_tools()
