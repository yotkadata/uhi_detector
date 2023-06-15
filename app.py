import streamlit as st
import os
import sys

# Add app to path
module_path = os.path.abspath(os.path.join("app"))
if module_path not in sys.path:
    sys.path.append(module_path)

st.set_page_config(
    page_title="Urban Heat Island Detector",
    # page_icon="👋",
    layout="wide",
)

with st.sidebar:
    st.write("### Page selection")
    page_selected = st.radio(
        "Select a page to show",
        (
            "Start",
            "Segmentation from file",
            "Segmentation examples",
            "Maps",
            "Life segementation",
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

elif page_selected == "Life segementation":
    # Show app/landsat-maps.py
    st.title("Life segementation")

elif page_selected == "Segmentation from file":
    import predict_file

    predict_file.main()

elif page_selected == "Segmentation examples":
    import examples

    examples.main()
