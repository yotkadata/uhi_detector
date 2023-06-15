"""
Collection of functions used in the prediction process.
"""

# Add the parent directory to the path to make imports work
import os
import sys

module_path = os.path.abspath(os.path.join("../src"))
if module_path not in sys.path:
    sys.path.append(module_path)

from pathlib import Path

import cv2
import numpy as np
import rasterio
import segmentation_models as sm
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import shutil

from smooth_tiled_predictions import predict_img_with_smooth_windowing
import prepare_data

CREATE_PATCHES = True
PREDICT = True

# Directory paths
dir_data_root = Path("data/predict/")

# Original images
dir_source = Path(dir_data_root, "source")

# Patches generated from the original images
dir_patches = Path(dir_data_root, "patches")

# Files used as input smooth predictions
dir_input = Path(dir_data_root, "input")

# Predictions
dir_output = Path(dir_data_root, "output")

# Model to be used for prediction
file_model = Path(
    "data/models/landcover_25_epochs_resnet34_backbone_batch16_iou_0.76.hdf5"
)

# Backbone used for the model, needed for preprocessing
BACKBONE = "resnet34"

# Size of patches
PATCH_SIZE_1 = 512  # Used as input to smooth_tiled_predictions
PATCH_SIZE_2 = 256  # Used as input to model.predict

# Number of patches to predict on
N_PATCHES = 1

classes = {0: "Not classified", 1: "Building", 2: "Woodland", 3: "Water", 4: "Roads"}

# Create patches
if CREATE_PATCHES:
    # prepare_data.create_patches(
    #     dir_source, dir_patches, filetypes=[file.suffix], patch_size=PATCH_SIZE_1
    # )
    prepare_data.create_geo_patches(dir_source, dir_patches, patch_size=PATCH_SIZE_1)

if PREDICT:
    scaler = MinMaxScaler()
    model = load_model(file_model, compile=False)
    preprocess_input = sm.get_preprocessing(BACKBONE)

    counter = 0

    # Use the patches as input to the model if patches are created
    input_dir = dir_patches if CREATE_PATCHES else dir_input

    for file in input_dir.iterdir():
        # Skip entries that are not files
        if not file.is_file():
            continue

        img = cv2.imread(str(file))

        # Normalize the image to values between 0 and 1
        input_img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(
            img.shape
        )

        # Preprocess the image
        input_img = preprocess_input(input_img)

        # Predict using smooth blending
        # The `pred_func` is passed and will process all the image 8-fold by tiling small
        # patches with overlap, called once with all those image as a batch outer dimension.
        # Note that model.predict(...) accepts a 4D tensor of shape (batch, x, y,
        # nb_channels), such as a Keras model.
        predictions_smooth = predict_img_with_smooth_windowing(
            input_img,
            window_size=PATCH_SIZE_2,
            subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
            nb_classes=len(classes),
            pred_func=(lambda img_batch_subdiv: model.predict((img_batch_subdiv))),
        )

        final_prediction = np.argmax(predictions_smooth, axis=2)

        # Use rasterio to save the prediction as a GeoTIFF
        with rasterio.open(file) as src:
            profile = src.profile.copy()
            profile.update(count=1)

        new_path = Path(dir_output, file.stem + "_prediction" + file.suffix)

        # Add dimension to final_prediction
        final_prediction = np.expand_dims(final_prediction, axis=0)

        with rasterio.open(new_path, "w", **profile) as dst:
            dst.write(final_prediction)

        counter += 1

        # Copy used patch to input folder for visualisation of the prediction
        shutil.copy(file, dir_input / file.name)

        print(f"Prediction on {file.name} finished.")

        # Stop after N_PATCHES
        if counter == N_PATCHES:
            break
