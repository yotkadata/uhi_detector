"""
Streamlit app for the final project.
Building a "Urban Heat Island Detector" using satellite images.
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
        <h1 style="text-align: center;">Urban Heat Island Detector</h1>
        <h2 style="text-align: center;">
            <strong>Final project for the Data Science Bootcamp</strong>
        </h2>
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
        # display_image(
        #     "data/presentation/uhi_roofs_black.png",
        #     caption="Source: c40knowledgehub.org",
        # )
        image_comparison(
            img1="data/presentation/uhi_roofs_black.png",
            img2="data/presentation/uhi_roofs_white.png",
            label1="",
            label2="",
            starting_position=99,
            width=700,
        )


@st.cache_data
def page_why():
    """
    Display page for "Why?" section.
    """
    col1, col2, col3 = st.columns([2, 5, 2])

    with col1:
        st.write("")

    with col2:
        st.title("Why is tackling UHI important?")

        st.markdown(
            """
            <ul class="presentation-bullets">
                <li>UHI effect <strong>exacerbates impacts of 
                    climate change</strong></li>
                <li>Leads to <strong>increased energy consumption</strong> and 
                    <strong>impaired air quality</strong></li>
                <li>Adverse health effects: <strong>heat-related 
                    illnesses and mortality</strong></li>
            </ul>
            <br />
            <br />
            <br />
            """,
            unsafe_allow_html=True,
        )

        display_image(
            "data/presentation/euromomo-charts-pooled-by-age-group.png", width=900
        )

    with col3:
        st.write("")


@st.cache_data
def page_goal():
    """
    Display page for "Why?" section.
    """
    col1, col2, col3, col4 = st.columns([1, 4, 2, 1])

    with col1:
        st.write("")

    with col2:
        st.title("Goal")

        st.markdown(
            """
            <ul class="presentation-bullets">
                <li>Use satellite imagery to <strong>detect and visualize UHI effects</strong>.</li>
                <li>Build and train a <strong>deep learning model to segment building roofs</strong> 
                in urban areas from high-resolution satellite imagery.</li>
                <li>Calculate <strong>brightness of building</strong> patterns.</li>
                <li>Incorporate Landsat 8 <strong>Thermal Infrared Sensor (TIRS)</strong> data.</li>
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
    col1, col2 = st.columns([1, 1])

    with col1:
        display_image("data/presentation/unet-model.png", width=600)
        display_image(
            "data/presentation/unet-architecture-building-extraction.jpg", width=600
        )

    with col2:
        st.title("How is it done?")

        st.markdown(
            """
            <ul class="presentation-bullets">
                <li>Using a <strong>pre-trained Unet segmentation model</strong> 
                    with transfer learning</li>
                <li><strong>Training dataset:</strong> Landcover.ai with 40 
                    high resolution labeled images (~40.000 patches of 256x256px)</li>
                <li>Selected <strong>2.000 patches</strong> with relevant information for training</li>
                <li>Model based on <strong>resnet-34 / resnet-50 backbone</strong> 
                    and <strong>imagenet weights</strong>.</li>
                <li>Used metric: <strong>Mean IoU</strong> (Intersection over Union). Best
                    score on test data: 0.86</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )


@st.cache_data
def building_footprints():
    """
    Display building footprints.
    """
    col1, col2, col3 = st.columns([2, 5, 2])

    with col2:
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            ["Satellite", "Buildings", "Luminance", "NDVI", "Emissivity", "LST"]
        )

        with tab1:
            display_image("data/presentation/wolfsburg-satellite-tight.jpg")

        with tab2:
            display_image("data/presentation/wolfsburg-buildings-tight.png")

        with tab3:
            display_image("data/presentation/wolfsburg-luminance-tight.png")

        with tab4:
            display_image("data/presentation/wolfsburg-ndvi-tight_manual.png")

        with tab5:
            display_image("data/presentation/wolfsburg-emissivity-tight_manual.png")

        with tab6:
            display_image("data/presentation/wolfsburg-lst-tight_manual.png")


@st.cache_data
def display_tools():
    """
    Display tools used in the project.
    """
    st.title("Thank you Bergamot Encoders!")

    st.markdown("""### ... and Rakib & Parvin & Carmine & all the others!""")

    st.markdown("<br /><br /><br /><br /><br /><br />", unsafe_allow_html=True)

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


@st.cache_data
def display_mean_temperatures():
    """
    Display mean temperatures image.
    """
    _, col2, _ = st.columns([1, 8, 1])

    with col2:
        st.image(
            "data/presentation/berlin-germany-mean-temperature-2022_ref-1942-2022.png",
            use_column_width=True,
        )


st.set_page_config(
    page_title="Urban Heat Island Detector",
    layout="wide",
)

with open("app/style.css", encoding="utf-8") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

with st.sidebar:
    # st.write("### Page selection")
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
            "Tools",
        ),
    )

if page_selected == "Start":
    page_start()

elif page_selected == "What?":
    page_what()

elif page_selected == "Why?":
    page_why()

elif page_selected == "Goal":
    page_goal()

elif page_selected == "How?":
    page_how()

elif page_selected == "Maps":
    landsat_maps.main()

elif page_selected == "Footprints":
    building_footprints()

elif page_selected == "Segment":
    segmentation.main()

elif page_selected == "Heatwaves":
    display_mean_temperatures()

elif page_selected == "Tools":
    display_tools()
