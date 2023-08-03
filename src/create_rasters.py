"""
Create raster files to be used in the web app.
"""

from pathlib import Path

import utils


def create_paths(dir_paths, geojson_path):
    """
    Create paths to the files needed for the analysis.
    """
    file_paths = {}

    for dir_id in dir_paths["source"].iterdir():
        # Skip files
        if not dir_id.is_dir():
            continue

        # Skip directory names that are not the right length
        if len(dir_id.name) != 40:
            continue

        paths = {}

        # Metadata
        paths["mtl"] = Path(dir_id, dir_id.name + "_MTL.txt")

        # Bands
        for i in range(1, 12):
            paths[f"band_{i}"] = Path(dir_id, dir_id.name + f"_B{i}.TIF")

        # Paths clipped to GeoJson
        output_dir = Path(dir_paths["output"], dir_id.name)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for i in range(1, 12):
            paths[f"band_{i}_clipped"] = utils.clip_to_geojson(
                paths[f"band_{i}"], geojson_path, output_dir
            )

        # Output file paths
        for name in ["lst", "ndvi", "emissivity"]:
            paths[f"{name}"] = Path(
                dir_paths["raster"], dir_id.name + "_" + f"{name}.tif"
            )
            paths[f"{name}_colored"] = Path(
                dir_paths["raster"], dir_id.name + "_" + f"{name}_colored.tif"
            )
            paths[f"{name}_repr"] = Path(
                dir_paths["raster"], dir_id.name + "_" + f"{name}_repr.tif"
            )
            paths[f"{name}_repr_colored"] = Path(
                dir_paths["raster"], dir_id.name + "_" + f"{name}_repr_colored.tif"
            )

        # Add paths to dictionary
        file_paths[dir_id.name] = paths

    return file_paths


def create_raster_ndvi(file_paths):
    """
    Create raster file for NDVI.
    """
    _ = utils.calc_ndvi(
        file_paths["band_4_clipped"],
        file_paths["band_5_clipped"],
        file_paths["ndvi"],
    )
    # Save colored image as files
    _, _ = utils.create_rgba_color_image(
        file_paths["ndvi"], file_paths["ndvi_colored"], colormap="Greens"
    )
    # Reproject and save
    _, _ = utils.reproject_geotiff(file_paths["ndvi"], file_paths["ndvi_repr"], 4326)
    # Save colored reprojected image as files
    _, _ = utils.create_rgba_color_image(
        file_paths["ndvi_repr"], file_paths["ndvi_repr_colored"], colormap="Greens"
    )


def create_raster_emissivity(file_paths):
    """
    Create raster file for Emissivity.
    """
    _ = utils.calc_emissivity(
        file_paths["band_4_clipped"],
        file_paths["band_5_clipped"],
        file_paths["emissivity"],
    )
    # Save colored image as files
    _, _ = utils.create_rgba_color_image(
        file_paths["emissivity"], file_paths["emissivity_colored"], colormap="Blues"
    )
    # Reproject and save
    _, _ = utils.reproject_geotiff(
        file_paths["emissivity"], file_paths["emissivity_repr"], 4326
    )
    # Save colored reprojected image as files
    _, _ = utils.create_rgba_color_image(
        file_paths["emissivity_repr"],
        file_paths["emissivity_repr_colored"],
        colormap="Blues",
    )


def create_raster_lst(file_paths):
    """
    Create raster file for Land Surface Temperature (LST).
    """
    _ = utils.calc_lst(
        file_paths["band_4_clipped"],
        file_paths["band_5_clipped"],
        file_paths["band_10_clipped"],
        file_paths["lst"],
    )
    # Save colored image as files
    _, _ = utils.create_rgba_color_image(
        file_paths["lst"], file_paths["lst_colored"], colormap="RdBu_r"
    )
    # Reproject and save
    _, _ = utils.reproject_geotiff(file_paths["lst"], file_paths["lst_repr"], 4326)

    # Save colored reprojected image as files
    _, _ = utils.create_rgba_color_image(
        file_paths["lst_repr"], file_paths["lst_repr_colored"], colormap="RdBu_r"
    )


def main():
    """
    Main function.
    """
    # Directory paths
    dir_paths = {
        "root": Path().cwd(),
        "data": Path().cwd() / "data",
        "data_landsat": Path().cwd() / "data/landsat",
        "source": Path().cwd() / "data/landsat/source",
        "output": Path().cwd() / "data/landsat/output",
        "raster": Path().cwd() / "data/raster_files",
    }

    geojson_path = Path(dir_paths["data"]) / "geojson" / "berlin.geojson"

    # Print file paths
    file_paths = create_paths(dir_paths, geojson_path)

    # Create raster files
    for _, dir_id in file_paths.items():
        create_raster_ndvi(dir_id)
        create_raster_emissivity(dir_id)
        create_raster_lst(dir_id)


if __name__ == "__main__":
    main()
