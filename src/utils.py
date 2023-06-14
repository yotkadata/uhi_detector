"""
Collection of utility functions.
"""

from pathlib import Path

import branca.colormap as cm
import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask
from matplotlib import colors
from pylandtemp import single_window, ndvi
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.windows import get_data_window, shape, transform
from shapely.geometry import shape


def clip_to_geojson(band_path, geojson_path, target_dir=None):
    """
    Clip a GeoTIFF file to an Area of Interest (AOI) from a Geojson file.
    """

    # Use the same directory as the band file if no target directory is specified
    if target_dir is None:
        target_dir = band_path.parent

    # Load a GeoJSON or shapefile of the area of interest
    aoi = gpd.read_file(geojson_path)

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


def clip_jp2_to_geojson(
    band_path,
    geojson_path,
    target_dir=None,
    driver_in="JP2OpenJPEG",
    driver_out="GTiff",
):
    """
    Clip a JP2 file to an Area of Interest (AOI) from a Geojson file.
    """

    # Use the same directory as the band file if no target directory is specified
    if target_dir is None:
        target_dir = band_path.parent

    # Load a GeoJSON or shapefile of the area of interest
    aoi = gpd.read_file(geojson_path)

    # Open the Landsat band file
    with rasterio.open(band_path, driver=driver_in) as src:
        # Set correct CRS for the AOI
        aoi = aoi.to_crs(src.crs)

        # Transform to a Shapely geometry
        aoi_shape = [shape(aoi.geometry.loc[0])]

        # Clip the raster file with the polygon using the mask function
        out_image, out_transform = rasterio.mask.mask(src, aoi_shape, crop=True)
        out_meta = src.meta

    # Update the metadata to include the new transform and shape
    out_meta.update(
        {
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "crs": src.crs,
            "driver": driver_out,
        }
    )

    file_path_new = target_dir / f"{band_path.stem}_clipped.tiff"  # {band_path.suffix}"

    # Save the clipped raster to a new GeoTiff file
    with rasterio.open(file_path_new, "w", **out_meta) as dest:
        dest.write(out_image)

    print(f"Saved clipped file to {file_path_new}")

    return str(file_path_new)


def normalize(band, stretch=True):
    """
    Normalize a band to the range 0-1.
    """
    band_min, band_max = (band.min(), band.max())

    with np.errstate(divide="ignore", invalid="ignore"):
        band_norm = (band - band_min) / (band_max - band_min)

    # Stretch the band to the range 0-255
    if stretch:
        band_norm = band_norm * 255

    return band_norm


def brighten(band, alpha, beta=0):
    """
    Brighten a band by adding a constant.
    """
    return np.clip(alpha * band + beta, 0, 255)


def gamma_corr(band, gamma):
    """
    Gamma correct a band.
    """
    return np.power(band, 1 / gamma)


def create_rgb_composite(
    band_red_path: Path,
    band_green_path: Path,
    band_blue_path: Path,
    target_path: Path,
    stretch: bool = True,
    gamma: float = 1,
    alpha: float = 1,
    beta: float = 0,
    driver: str = "GTiff",
) -> tuple[np.ndarray, str]:
    """
    Create a RGB composite from three bands.
    """

    with rasterio.open(band_blue_path, driver=driver) as src:
        band_blue = src.read(1)

    with rasterio.open(band_green_path, driver=driver) as src:
        band_green = src.read(1)

    with rasterio.open(band_red_path, driver=driver) as src:
        band_red = src.read(1)

    # Apply gamma correction (default: gamma=1 means no change)
    red_g = gamma_corr(band_red, gamma=gamma)
    blue_g = gamma_corr(band_blue, gamma=gamma)
    green_g = gamma_corr(band_green, gamma=gamma)

    # Apply brightness correction (default: alpha=1 means no change)
    red_gb = brighten(red_g, alpha=alpha, beta=beta)
    blue_gb = brighten(blue_g, alpha=alpha, beta=beta)
    green_gb = brighten(green_g, alpha=alpha, beta=beta)

    # Normalize the bands to the range 0-1
    red_gbn = normalize(red_gb, stretch=stretch)
    green_gbn = normalize(green_gb, stretch=stretch)
    blue_gbn = normalize(blue_gb, stretch=stretch)

    # Create the RGB composite
    rgb_composite_gbn = np.dstack((red_gbn, green_gbn, blue_gbn))

    # Create output directory if it does not exist
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the RGB composite to a new GeoTiff file
    with rasterio.open(
        target_path,
        "w",
        driver=src.driver,
        height=band_red.shape[0],
        width=band_red.shape[1],
        count=3,
        dtype="uint8",
        crs=src.crs,
        transform=src.transform,
        nodata=0,
    ) as dst:
        # Move channel information to first axis
        out_array = np.moveaxis(rgb_composite_gbn, source=2, destination=0)
        dst.write(out_array.astype("uint8"))

    if target_path.is_file():
        print(f"RGB composite saved to {target_path}")

    return rgb_composite_gbn.astype("uint8"), target_path


def create_rgb_composite_2(
    band_red_path: Path,
    band_green_path: Path,
    band_blue_path: Path,
    target_path: Path,
    normal: bool = True,
    stretch: bool = True,
    gamma: float = None,
    alpha: float = None,
    beta: float = 0,
    driver_in: str = "GTiff",
    driver_out: str = "GTiff",
):
    """
    Create a RGB composite from three bands.
    """
    band_red = rasterio.open(band_red_path, driver=driver_in)
    band_red_read = band_red.read(1)

    band_green = rasterio.open(band_green_path, driver=driver_in)
    band_green_read = band_green.read(1)

    band_blue = rasterio.open(band_blue_path, driver=driver_in)
    band_blue_read = band_blue.read(1)

    if gamma is not None:
        # Apply gamma correction (default: gamma=1 means no change)
        band_red_read = gamma_corr(band_red_read, gamma=gamma)
        band_green_read = gamma_corr(band_green_read, gamma=gamma)
        band_blue_read = gamma_corr(band_blue_read, gamma=gamma)

    if alpha is not None:
        # Apply brightness correction (default: alpha=0 means no change)
        band_red_read = brighten(band_red_read, alpha=alpha, beta=beta)
        band_green_read = brighten(band_green_read, alpha=alpha, beta=beta)
        band_blue_read = brighten(band_blue_read, alpha=alpha, beta=beta)

    if normal:
        # Normalize the bands to the range 0-1 or 0-255
        band_red_read = normalize(band_red_read, stretch=stretch)
        band_green_read = normalize(band_green_read, stretch=stretch)
        band_blue_read = normalize(band_blue_read, stretch=stretch)

    # Create output directory if it does not exist
    target_path.parent.mkdir(parents=True, exist_ok=True)

    rgb = rasterio.open(
        target_path,
        "w",
        driver=driver_out,
        width=band_red.width,
        height=band_red.height,
        count=3,
        crs=band_red.crs,
        transform=band_red.transform,
        dtype="uint16",
    )

    rgb.write(band_red_read, 1)
    rgb.write(band_green_read, 2)
    rgb.write(band_blue_read, 3)
    rgb.close()

    if target_path.is_file():
        print(f"RGB composite saved to {target_path}")

    return target_path


def resample_image(input_file, output_file, target_resolution):
    """
    Resample an image to a target ground resolution (in meters).
    """
    with rasterio.open(input_file) as dataset:
        current_resolution = dataset.res[0]  # assuming x and y resolutions are the same
        downsampling_factor = current_resolution / target_resolution

        width = int(dataset.width / downsampling_factor)
        height = int(dataset.height / downsampling_factor)

        profile = dataset.profile
        profile.update(
            width=width,
            height=height,
            transform=dataset.transform * dataset.transform.scale(downsampling_factor),
        )

        with rasterio.open(output_file, "w", **profile) as dst:
            for i in range(1, dataset.count + 1):
                data = dataset.read(
                    i, out_shape=(height, width), resampling=Resampling.bilinear
                )
                dst.write(data, i)


def calc_lst(
    band_4_path: Path, band_5_path: Path, band_10_path: Path, target_path: Path
) -> np.ndarray:
    """
    Calculate land surface temperature (LST) from Landsat 8
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


def calc_ndvi(band_4_path: Path, band_5_path: Path, target_path: Path) -> np.ndarray:
    """
    Calculate Normalized Difference Vegetation Index (NDVI) from Landsat 8
    bands 4 and 5 using pylandtemp library.
    """
    with rasterio.open(band_4_path) as src:
        band_4 = src.read(1)

    with rasterio.open(band_5_path) as src:
        band_5 = src.read(1)
        out_meta = src.meta.copy()

    mask = band_4 == 0
    ndvi_image_array = ndvi(band_5, band_4, mask=mask)

    out_meta.update(
        {
            "height": ndvi_image_array.shape[0],
            "width": ndvi_image_array.shape[1],
            "transform": src.transform,
            "dtype": ndvi_image_array.dtype,
        }
    )

    with rasterio.open(target_path, "w", **out_meta) as dst:
        dst.write(ndvi_image_array, 1)

    print("Saved NDVI to", target_path)

    return ndvi_image_array


def exaggerate(input_array: np.ndarray, factor: float = 2) -> np.ndarray:
    """
    Exaggerate the values of an array.
    """
    # Calculate the mean temperature
    mean_temp = np.mean(input_array)

    valid_mask = (input_array != 0) & (input_array > mean_temp)

    deviation = np.where(valid_mask, input_array - mean_temp, 0)

    # Calculate the deviation from the mean
    # deviation = input_array - mean_temp

    # Apply an exaggeration function to the deviation
    exaggerated_deviation = np.power(deviation, factor)

    # Add the exaggerated deviation back to the mean temperature
    exaggerated_temperature = mean_temp + exaggerated_deviation

    return exaggerated_temperature


def convert_values_to_colors(values: np.ndarray, cmap) -> np.ndarray:
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


def create_rgba_color_image(src_path: Path, target_path: Path):
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


def clip_to_remove_nodata(input_path: Path, output_path: Path = None) -> None:
    """
    Clip a raster to the data window to remove nodata values.
    TODO: Doesn't clip. https://gis.stackexchange.com/a/428982
    """
    if output_path is None:
        output_path = Path(
            input_path.parent, input_path.stem + "_clipped" + input_path.suffix
        )

    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        data_window = get_data_window(src.read(masked=True))
        data_transform = transform(data_window, src.transform)
        profile.update(
            transform=data_transform,
            height=data_window.height,
            width=data_window.width,
        )

        data = src.read(window=data_window)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data)

    if output_path.is_file():
        print(f"Clipped file saved to {output_path}")
