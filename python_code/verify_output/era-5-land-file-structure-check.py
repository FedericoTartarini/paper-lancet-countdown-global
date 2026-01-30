"""
This code checks the file structure of ERA5-Land data files in a specified directory.
The file is copied from Gadi to a local directory for verification using the following command:
scp ft8695@gadi-dm.nci.org.au:/g/data/zz93/era5-land/reanalysis/2t/1950/2t_era5-land_oper_sfc_19500101-19500131.nc ~/Downloads/
"""

import xarray as xr
from pathlib import Path
import os


def check_file_structure():
    # Path to the downloaded file
    file_path = Path(
        os.path.expanduser("~/Downloads/2t_era5-land_oper_sfc_19500101-19500131.nc")
    )

    print(f"Attempting to read file: {file_path}")

    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        print("Please check the file path and ensure the file has been downloaded.")
        return

    try:
        # Open the dataset
        ds = xr.open_dataset(file_path)

        print("\n" + "=" * 50)
        print("DATASET SUMMARY")
        print("=" * 50)
        print(ds)

        print("\n" + "=" * 50)
        print("TIME DIMENSION ANALYSIS")
        print("=" * 50)
        # Check time frequency
        if "time" in ds.coords:
            times = ds.time.values
            print(f"Number of time steps: {len(times)}")
            print(f"First time step: {times[0]}")
            print(f"Last time step:  {times[-1]}")

            if len(times) > 1:
                time_diff = times[1] - times[0]
                print(f"Time interval: {time_diff}")

                # Heuristic for hourly vs daily
                diff_hours = time_diff.astype("timedelta64[h]").astype(int)

                if diff_hours == 1:
                    print("\nCONCLUSION: Data is HOURLY.")
                    print("You will need to resample this to daily min/mean/max.")
                elif diff_hours == 24:
                    print("\nCONCLUSION: Data is DAILY.")
                else:
                    print(f"\nCONCLUSION: Data has a frequency of {diff_hours} hours.")
        else:
            print("No 'time' coordinate found.")

        print("\n" + "=" * 50)
        print("VARIABLES")
        print("=" * 50)
        for var_name in ds.data_vars:
            var = ds[var_name]
            print(f"Variable: {var_name}")
            print(f"  Long Name: {var.attrs.get('long_name', 'N/A')}")
            print(f"  Units: {var.attrs.get('units', 'N/A')}")
            print(f"  Shape: {var.shape}")

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")


if __name__ == "__main__":
    check_file_structure()
