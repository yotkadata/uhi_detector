"""
Script to prepare the LST layer to be displayed in Folium.
"""

from pathlib import Path

import branca.colormap as cm
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.mask
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from pylandtemp import single_window
from pyproj import Transformer
from rasterio.plot import show
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import shape

# ID of the Landsat 8 files
id = "LC08_L1TP_193023_20220803_20220806_02_T1"

# Directory paths
dir_data_root = Path("data/landsat/")
dir_files = Path(dir_data_root, id)
dir_output = Path(dir_data_root, "output")

# File paths
geojson_path = Path(dir_data_root, "geojson", "berlin.geojson")
mtl_path = Path(dir_files, id + "_MTL.txt")

# Bands
band_4_path = Path(dir_files, id + "_B4.TIF")
band_5_path = Path(dir_files, id + "_B5.TIF")
band_10_path = Path(dir_files, id + "_B10.TIF")

# Output file paths
file_temperature = Path(dir_output, "land_surface_temperature.tif")
file_temperature_repr = Path(dir_output, "temperature_reprojected.tif")
file_temperature_repr_colored = Path(dir_output, "temperature_reprojected_colored.tif")


def calc_lst(band_4_path, band_5_path, band_10_path, target_path):
    """
    Calculates the land surface temperature (LST) from Landsat 8
    bands 4, 5 and 10 using pylandtemp library.
    """
    with rasterio.open(band_4_path) as src:
        band_4 = src.read(1)

    with rasterio.open(band_5_path) as src:
        band_5 = src.read(1)

    with rasterio.open(band_10_path) as src:
        band_10 = src.read(1)
        out_meta = src.meta.copy()

    lst_image_array = single_window(band_10, band_4, band_5, unit="celsius")

    # For some reason, the values are in Kelvin, so we need to convert them to Celsius
    lst_image_array = lst_image_array - 273.15

    out_meta.update(
        {
            "height": lst_image_array.shape[0],
            "width": lst_image_array.shape[1],
            "transform": src.transform,
            "dtype": lst_image_array.dtype,
        }
    )

    with rasterio.open(target_path, "w", **out_meta) as dst:
        dst.write(lst_image_array, 1)

    print("Saved Land Surface Temperature (LST) to", target_path)

    return lst_image_array


def exaggerate(temp_array, factor=2):
    """
    Exaggerate the values of an array.
    """
    # Calculate the mean temperature
    mean_temp = np.mean(temp_array)

    valid_mask = (temp_array != 0) & (temp_array > mean_temp)

    deviation = np.where(valid_mask, temp_array - mean_temp, 0)

    # Calculate the deviation from the mean
    # deviation = temp_array - mean_temp

    # Apply an exaggeration function to the deviation
    exaggerated_deviation = np.power(deviation, factor)

    # Add the exaggerated deviation back to the mean temperature
    exaggerated_temperature = mean_temp + exaggerated_deviation

    return exaggerated_temperature


def clip_to_geojson(band_path, path_geojson, target_dir):
    """
    Clip a GeoTIFF file to an Area of Interest (AOI) from a Geojson file.
    """
    # Load a GeoJSON or shapefile of the area of interest
    aoi = gpd.read_file(path_geojson)

    # Set correct CRS (obtained from:
    # check = rasterio.open(band_path)
    # check.crs
    aoi = aoi.to_crs("EPSG:32633")  # TODO Take directly from src.crs

    # Transform to a Shapely geometry
    aoi_shape = [shape(aoi.geometry.loc[0])]

    # Open the Landsat band file
    with rasterio.open(band_path) as src:
        # Clip the raster file with the polygon using mask function
        out_image, out_transform = rasterio.mask.mask(src, aoi_shape, crop=True)
        out_meta = src.meta

    # Update the metadata to include the new transform and shape
    out_meta.update(
        {
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        }
    )

    file_path_new = target_dir / f"{band_path.stem}_clipped{band_path.suffix}"

    # Save the clipped raster to a new GeoTiff file
    with rasterio.open(file_path_new, "w", **out_meta) as dest:
        dest.write(out_image)

    print(f"Saved clipped file to {file_path_new}")

    return str(file_path_new)


#
# Prepare files for Folium layer
#


def convert_values_to_colors(values, cmap):
    """
    Convert image values to colors using a colormap.

    Inputs:
        - values: 2D NumPy array of image values
        - cmap: a branca.colormap.LinearColormap object
    Output:
        - 3D NumPy array representing colored image
    """
    colors_arr = np.zeros((*values.shape, 4))
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isnan(value):
                colors_arr[i, j, :] = [0, 0, 0, 0]  # Transparent color for NaN values
            else:
                color = colors.to_rgba(cmap(value), alpha=0.7)
                colors_arr[i, j, :] = color
    return colors_arr


def reproject_geotiff(src_path, target_path, target_crs):
    """
    Function to reproject a GeoTiff to a different CRS.
    """
    # Open the raster file
    with rasterio.open(src_path) as src:
        img = src.read().astype(np.float32)

        # Set the nodata values to NaN
        img[img == src.nodata] = np.nan

        # Define target CRS (e.g., WGS84)
        target_crs = rasterio.crs.CRS.from_epsg(target_crs)

        # Calculate the default transform for the reprojected image
        transform, width, height = rasterio.warp.calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )

        # Create an array to hold the reprojected image
        reprojected_img = np.zeros((img.shape[0], height, width), dtype=img.dtype)

        # Reproject the image to match the transformed bounds
        reproject(
            src.read(),
            reprojected_img,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=target_crs,
        )

        # Update the metadata for the reprojected image
        reprojected_meta = src.meta.copy()
        reprojected_meta.update(
            {
                "crs": target_crs,
                "transform": transform,
                "width": width,
                "height": height,
            }
        )

        # Set the nodata values to NaN
        reprojected_img[reprojected_img == src.nodata] = np.nan

        # Write the reprojected image to a new GeoTIFF file
        with rasterio.open(target_path, "w", **reprojected_meta) as dst:
            dst.write(reprojected_img)

        return dst, reprojected_img


def create_rgba_color_image(src_path, target_path):
    """
    Function to map raster values to rgba.
    """
    with rasterio.open(src_path) as src:
        img = src.read()  # .astype(np.float32)

        # Set the nodata values to NaN
        # img[img == src.nodata] = np.nan

        vmin = np.floor(np.nanmin(img))
        vmax = np.ceil(np.nanmax(img))

        colormap = cm.linear.RdBu_11.scale(vmin, vmax)
        colormap.colors.reverse()

        colored_img = convert_values_to_colors(img[0], colormap)

        # Update src.meta
        colored_meta = src.meta.copy()
        colored_meta.update({"count": 4})

        # Write the reprojected image to a new GeoTIFF file
        with rasterio.open(target_path, "w", **colored_meta) as dst:
            # Move channel information to first axis
            colored_img_rio = np.moveaxis(colored_img, source=2, destination=0)
            dst.write(colored_img_rio)

        return dst, colored_img


def main():
    """
    Main function.
    """
    # Create files clipped to GeoJson
    band_4_clipped_path = clip_to_geojson(band_4_path, geojson_path, dir_output)
    band_5_clipped_path = clip_to_geojson(band_5_path, geojson_path, dir_output)
    band_10_clipped_path = clip_to_geojson(band_10_path, geojson_path, dir_output)
    # TODO: Somehow clip_to_geojson() deletes MTL file

    # Calculate Land Surface Temperature
    lst = calc_lst(
        band_4_clipped_path, band_5_clipped_path, band_10_clipped_path, file_temperature
    )

    # Reproject the temperature file to WGS84 and save to new file
    reprojected_img, reprojected_img_array = reproject_geotiff(
        file_temperature, file_temperature_repr, 4326
    )
    with rasterio.open(file_temperature_repr) as src:
        temperature_reprojected = src.read(1)

    # Recalculate the colors and save to new file
    colored_img, colored_img_array = create_rgba_color_image(
        reprojected_img.name, file_temperature_repr_colored
    )
    with rasterio.open(file_temperature_repr_colored) as src:
        temperature_reprojected_colored = src.read(1)


if __name__ == "__main__":
    main()
