import glob
import os
import shutil
from pathlib import Path

import my_config


# moved file from computer to external hard drive if not backed up already
def move_summary_files(backup=True, delete_original=True):
    """
    Move the summary files to the backup location.

    If backup is True, the files backed up and a copy is kept in the original location.
    If delete_original is True, the original files are deleted.
    """

    number_files_total = 0
    number_file_backup = 0
    for file in glob.glob(str(my_config.dir_pop_raw) + "/*.tif"):
        move_location = file.replace(
            str(my_config.dir_pop_raw), str(my_config.dir_ssd_path_population)
        )
        move_location = Path(move_location)
        number_files_total += 1

        if backup:
            if move_location.exists():
                print("File already backed up", file)
            else:
                print("Backing up file:", file)
                number_file_backup += 1
                shutil.copy(file, move_location)
        if delete_original:
            print("Removing original file:", file)
            os.remove(file)

    print(f"Number of files: {number_files_total}")
    print(f"Number of files backed up: {number_file_backup}")


# moved file from computer to external hard drive if not backed up already
def copy_back_laptop():
    file_to_move = 0
    for file in glob.glob(str(my_config.dir_ssd_path_population) + "/*.tif"):
        move_location = file.replace(
            str(my_config.dir_ssd_path_population), str(my_config.dir_pop_raw)
        )
        move_location = Path(move_location)
        if move_location.exists():
            # print("File already exist", file)
            pass
        else:
            print("Copying file:", file)
            file_to_move += 1
            # if file_to_move > 15:
            #     break
            shutil.copy(file, move_location)

    print(f"Number of files to move: {file_to_move}")


if __name__ == "__main__":
    # move_summary_files(backup=True, delete_original=False)

    copy_back_laptop()
