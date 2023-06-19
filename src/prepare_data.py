"""
Functions to prepare data for training.
"""

import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio
from patchify import patchify
from rasterio.windows import Window


def create_patches(
    image_dir: Path, patch_dir: Path, filetypes: list = None, patch_size: int = 256
) -> None:
    """
    Function to create patches from large images.
    """
    # Set default file types
    if filetypes is None:
        filetypes = [".tif", ".tiff"]

    # Make sure patch_dir exists
    Path(patch_dir).mkdir(parents=True, exist_ok=True)

    # Loop through image directory
    for file in image_dir.rglob("*"):
        if file.is_file() and file.suffix in filetypes:
            # Read each image as BGR
            image = cv2.imread(str(file), 1)

            # Calculate nearest size divisible by patch size
            size_x = (image.shape[1] // patch_size) * patch_size
            size_y = (image.shape[0] // patch_size) * patch_size

            # Crop image (y comes first!)
            image = image[0:size_y, 0:size_x]

            # Extract patches from each image (Step=256 for 256 patches means no overlap)
            print("Patchifying image:", file)
            patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)

            counter_create, counter_skip = 0, 0

            for i in range(patches_img.shape[0]):
                for j in range(patches_img.shape[1]):
                    single_patch_img = patches_img[i, j, :, :]

                    # Drop extra unecessary dimension that patchify adds.
                    single_patch_img = single_patch_img[0]

                    patch_path = Path(
                        patch_dir, f"{file.stem}_patch_{str(i)}_{str(j)}{file.suffix}"
                    )

                    if not patch_path.is_file():
                        cv2.imwrite(str(patch_path), single_patch_img)
                        counter_create += 1
                    else:
                        counter_skip += 1

            print(f"Created {counter_create} patches in {patch_dir}.")
            print(f"Skipped {counter_skip} already existing patches.")


def create_geo_patches(
    input_dir: Path,
    patch_dir: Path,
    filetypes: list = None,
    patch_size: int = 256,
):
    """
    Function to create patches from large images, maintaining geodata.
    """

    # Set default file types
    if filetypes is None:
        filetypes = [".tif", ".tiff"]

    # Check if output folder exists, if not, create it
    patch_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over all files in the input folder
    for filepath in input_dir.glob("*.tif"):
        # Skip if not a file or not correct file type
        if not filepath.is_file() or not filepath.suffix in filetypes:
            continue

        # Open the GeoTIFF file
        with rasterio.open(filepath) as src:
            # Get the metadata of the original GeoTIFF file
            meta = src.meta

            # Calculate the number of patches in row and column directions
            n_patches_x = src.width // patch_size
            n_patches_y = src.height // patch_size

            counter_create, counter_skip = 0, 0

            # Iterate over all patches
            for i in range(n_patches_y):
                for j in range(n_patches_x):
                    # Define the output filename
                    output_filename = patch_dir / f"{filepath.stem}_patch_{i}_{j}.tif"

                    if not output_filename.is_file():
                        # Define the window of the patch
                        window = Window(
                            j * patch_size, i * patch_size, patch_size, patch_size
                        )

                        # Read the patch
                        patch = src.read(window=window)

                        # Update the metadata for the patch
                        meta.update(
                            {
                                "height": patch_size,
                                "width": patch_size,
                                "transform": rasterio.windows.transform(
                                    window, src.transform
                                ),
                            }
                        )

                        # Write the patch to a new GeoTIFF file
                        with rasterio.open(output_filename, "w", **meta) as dst:
                            dst.write(patch)
                        counter_create += 1
                    else:
                        counter_skip += 1

            print(f"Created {counter_create} patches in {patch_dir}.")
            print(f"Skipped {counter_skip} already existing patches.")


def create_patch_inventory(
    patch_dir: Path, csv_path: Path, classes: dict, force: bool = False
) -> pd.DataFrame:
    """
    Function to create an inventory of classes in the patches.
    Returns a DataFrame with infomation how often each class appears in mask patches.
    """

    # Check if file already exists
    if csv_path.is_file() and not force:
        print(f"File {csv_path} already exists. Loading DataFrame...")
        return pd.read_csv(csv_path, index_col=0)

    # Create empty dataframe
    classes_pct = [f"{str(k)}_pct" for k in classes.keys()]
    cols = (
        list(["img"] + list(classes.keys())) + ["total"] + classes_pct + ["useful_pct"]
    )
    inventory = pd.DataFrame(columns=cols)

    for file in patch_dir.iterdir():
        mask = cv2.imread(file.as_posix(), 1)

        # Count how often each class exists as pixel value
        val, counts = np.unique(mask, return_counts=True)

        # Create a dict to be added as row to the DataFrame
        row = dict(zip(val, counts))
        row["total"] = sum(row.values())
        pixels_cat_0 = counts[0] if 0 in row.keys() else 0
        row["useful_pct"] = 1 - pixels_cat_0 / counts.sum()
        row["img"] = file.name

        # Calculate percentages
        for v in val:
            row[f"{str(v)}_pct"] = row[v] / row["total"]

        df_tmp = pd.DataFrame(row, index=[0])
        inventory = pd.concat([inventory, df_tmp], ignore_index=True).fillna(0)

    inventory.to_csv(csv_path)
    print(f"Saved patch inventory to {csv_path}")

    # Save smaller version excluding useless images
    inventory_short = (
        inventory[inventory["useful_pct"] >= 0.5]
        .sort_values("useful_pct", ascending=False)
        .reset_index(drop=True)
    )

    csv_short_path = Path(csv_path.parent, f"{csv_path.stem}_short{csv_path.suffix}")
    inventory_short.to_csv(csv_short_path)
    print(f"Shorter version saved to {csv_short_path}")

    return inventory


def select_useful_patches(
    dir_patch_img: Path,  # Directory with image patches
    dir_patch_mask: Path,  # Directory with masks
    dir_patch_useful_img: Path,  # Target dir for useful patches
    dir_patch_useful_mask: Path,  # Target dir for useful masks
    inventory_path: Path,  # Path to inventory csv
    classes: dict,  # Dict with classes
) -> None:
    """
    Function to select useful patches and move them to a folder.
    """
    # Delete and recreate directories
    for dir in [dir_patch_useful_img, dir_patch_useful_mask]:
        # Delete directory if it exists
        if dir.is_dir():
            shutil.rmtree(str(dir))
        # Create directory
        Path(dir).mkdir(parents=True, exist_ok=True)

    # Load inventory
    inventory = pd.read_csv(inventory_path, index_col=0)

    # # Get image names with most buildings
    # accepted_images = (
    #     inventory[inventory["1_pct"] >= 0.01]
    #     .sort_values("1_pct", ascending=False)
    #     .reset_index()
    #     .loc[:1999, "img"]
    #     .tolist()
    # )
    # print(f"Added {len(accepted_images)} images with focus on buildings.")

    # Get image names with most useful information
    # accepted_images = (
    #     inventory.sort_values("useful_pct", ascending=False)
    #     .reset_index()
    #     .loc[:1999, "img"]
    #     .tolist()
    # )

    # Make sure all classes are represented
    accepted_images = []
    for class_name in classes.keys():
        if not class_name in [0, 2]:
            selection = (
                inventory[~inventory["img"].isin(accepted_images)]
                .sort_values(f"{class_name}_pct", ascending=False)
                .reset_index()
                .loc[:999, "img"]
                .tolist()
            )
            accepted_images += selection
            print(f"Added {len(selection)} images with class {class_name}.")

    count_useful, count_source_missing = 0, 0

    # Loop through image directory
    for file_name in accepted_images:
        img = Path(dir_patch_img, file_name)
        mask = Path(dir_patch_mask, file_name)

        if img.is_file():
            new_file_img = Path(dir_patch_useful_img, img.name)
            new_file_mask = Path(dir_patch_useful_mask, mask.name)

            if not new_file_img.is_file():
                shutil.copy(str(img), str(new_file_img))
                shutil.copy(str(mask), str(new_file_mask))
                count_useful += 1
        else:
            count_source_missing += 1

    print(
        f"{len(accepted_images)} useful images in list.\n"
        f"{count_source_missing} missing in source.\n"
        f"{count_useful} copied."
    )
