# Add the parent directory to the path to make imports work
import os
import sys

module_path = os.path.abspath(os.path.join("../src"))
if module_path not in sys.path:
    sys.path.append(module_path)

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import segmentation_models as sm
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

import plot
import prepare_data
from smooth_tiled_predictions import predict_img_with_smooth_windowing

# Directory paths
dir_data_root = Path("../data/orthophotos/")
dir_input_files = Path(dir_data_root, "input")
dir_output = Path(dir_data_root, "output")
dir_results = Path(dir_data_root, "predict", "results")
dir_models = Path("../models")

file_model = Path(
    dir_models, f"landcover_25_epochs_resnet34_backbone_batch16_iou_0.76.hdf5"
)

classes = {0: "Not classified", 1: "Building", 2: "Woodland", 3: "Water", 4: "Roads"}

BACKBONE = "resnet34"
preprocess_input = sm.get_preprocessing(BACKBONE)

# Size of patches
patch_size = 256

# Number of classes
n_classes = len(classes)

scaler = MinMaxScaler()

model = load_model(file_model, compile=False)
for file in dir_input_files.iterdir():
    # Skip entries that are not files
    if not file.is_file():
        continue

    img = cv2.imread(str(file))
    input_img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
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
        nb_classes=len(classes),
        pred_func=(lambda img_batch_subdiv: model.predict((img_batch_subdiv))),
    )

    final_prediction = np.argmax(predictions_smooth, axis=2)

    # Use rasterio to save the prediction as a GeoTIFF
    with rasterio.open(file) as src:
        profile = src.profile
        profile.update(count=1)
        new_path = Path(dir_results, file.stem + "_prediction" + file.suffix)

        # Add dimension to final_prediction
        final_prediction = np.expand_dims(final_prediction, axis=0)

        with rasterio.open(new_path, "w", **profile) as dst:
            dst.write(final_prediction)
