import glob
import os
import shutil
from pathlib import Path

from my_config import (
    temperature_summary_folder,
    hd_path_daily_temperature_summary,
)


# moved file from computer to external hard drive if not backed up already
def move_summary_files(backup=True, delete_original=True):
    """
    Move the summary files to the backup location.

    If backup is True, the files backed up and a copy is kept in the original location.
    If delete_original is True, the original files are deleted.
    """

    for file in glob.glob(str(temperature_summary_folder) + "/*.nc"):
        move_location = file.replace(
            str(temperature_summary_folder), str(hd_path_daily_temperature_summary)
        )
        move_location = Path(move_location)

        if backup:
            if move_location.exists():
                print("File already backed up", file)
            else:
                print("Backing up file:", file)
                shutil.copy(file, move_location)
        if delete_original:
            print("Removing original file:", file)
            os.remove(file)


if __name__ == "__main__":
    move_summary_files(backup=True, delete_original=False)
