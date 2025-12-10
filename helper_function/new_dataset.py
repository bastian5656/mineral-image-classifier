"""
Takes an int (between 0-100) telling what amount of the data (in percentage) you want to extract from the original dataset.
Furthermore, you can set what subfolders you want to extract if no all. The options are: 
[azurite, baryte, beryl, calcite, cerussite, copper, fluorite, gypsum, hematite, malachite, pyrite, pyromorphite, quartz, smithsonite, wulfenite]
"""
import random
import shutil

from pathlib import Path
from typing import List

def new_dataset(perc_amount: int,
                subfolders: List[str] = ["copper", "azurite", "malachite", "pyrite", "quartz"]):
    # Original directory
    original_data_path = Path("data/")
    # Create new directory
    str_percentage = str(perc_amount)
    new_data_path = Path(f"data_{str_percentage}_percent")
    # Checking if directory already exits
    if new_data_path.exists():
        print(f"Directory {new_data_path} already exits. Stopping function...")
        return
    # Continuig function in case it doesn't exits
    new_data_path.mkdir(exist_ok=True)

    # Folder to split
    to_split = ["train", "test"]
    # Classes selected
    selected_classes = subfolders

    for split in to_split:
        # Select directory to split
        split_path = original_data_path / split
        # Create new directory for the selected percentage data for split
        new_split_path = new_data_path / split
        new_split_path.mkdir(exist_ok=True)

        # Iterate over selected classes
        for select_class in selected_classes:
            # Iterate over existing clases to find the right directory
            for class_dir in split_path.iterdir():
                # See if they match, otherwise pass
                if class_dir.name != select_class:
                    pass
                else:
                    # Create new class dir
                    new_class_dir = new_split_path / class_dir.name
                    new_class_dir.mkdir(exist_ok=True)

                    # Create list of images
                    images = list(class_dir.glob("*"))
                    total_images = len(images)

                    # Calculate the amount of data corresponding to the percentage selected
                    sample_size = round(total_images * (perc_amount / 100))
                    selected_samples = random.sample(images, k=sample_size)

                    # Copy selected samples
                    print(f"Copying {str_percentage}% of {select_class} data into {new_split_path}")
                    for img in selected_samples:
                        shutil.copy(img, new_class_dir / img.name)

    return print(f"✔ Done! Folder 'data_{str_percentage}_percent' created with {str_percentage}% of the original dataset.")
