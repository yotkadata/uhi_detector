"""
Collection of functions used for prediction.
"""
import cv2
import numpy as np
import os
import sys
import segmentation_models as sm
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st

module_path = os.path.abspath(os.path.join("../src"))
if module_path not in sys.path:
    sys.path.append(module_path)

from smooth_tiled_predictions import predict_img_with_smooth_windowing


@st.cache_data
def get_smooth_prediction_for_file(img, _model, n_classes, backbone, patch_size=256):
    """
    Predict on a single file.
    """
    # model = load_model(model_path, compile=False)
    scaler = MinMaxScaler()
    preprocess_input = sm.get_preprocessing(backbone)

    if not isinstance(img, np.ndarray):
        img = cv2.imread(str(img))

    # Normalize the image to values between 0 and 1
    input_img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)

    # Preprocess the image
    input_img = preprocess_input(input_img)

    # Predict using smooth blending
    # The `pred_func` is passed and will process all the image 8-fold by tiling small
    # patches with overlap, called once with all those image as a batch outer dimension.
    # Note that model.predict(...) accepts a 4D tensor of shape (batch, x, y,
    # nb_channels), such as a Keras model.
    predictions_smooth = predict_img_with_smooth_windowing(
        input_img,
        window_size=patch_size,
        subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
        nb_classes=n_classes,
        pred_func=(lambda img_batch_subdiv: _model.predict((img_batch_subdiv))),
    )

    final_prediction = np.argmax(predictions_smooth, axis=2)

    return final_prediction
