"""
The old WorldPop data had separate files for male and female populations.
This script combines those files into total population files.

This code was only needed for the 2026 report, and it can be removed later.

This version preserves the original source file naming convention (numeric age
strings like '65_70_75_80') when reading male/female inputs, but writes the
combined total files using the requested new labels:
- [0] -> 'under_1'
- [65, ...] -> '65_over'
- [75, 80] -> '75_over'

So for example, it will read:
  f_0_2015_era5_compatible.nc and m_0_2015_era5_compatible.nc
and write:
  t_under_1_2015_era5_compatible.nc

This avoids trying to read non-existent files while producing the renamed outputs.
"""

import xarray as xr
from tqdm import tqdm
from my_config import DirsLocal  # Assuming you have this config


def _output_age_label(age_group):
    """Return the desired output age label for a given age_group list.

    Mapping rules:
    - [0] -> 'under_1'
    - startswith 65 and length > 1 -> '65_over'
    - startswith 75 (e.g., [75,80]) -> '75_over'
    - otherwise fallback to the numeric joined string (e.g., '30_34')
    """
    if len(age_group) == 1 and age_group[0] == 0:
        return "under_1"

    if len(age_group) >= 2 and age_group[0] == 65:
        return "65_over"

    if len(age_group) >= 2 and age_group[0] == 75:
        return "75_over"

    # default
    return "_".join(map(str, age_group))


def _source_age_string(age_group):
    """Return the source filename age string (numeric) used in existing files."""
    return "_".join(map(str, age_group))


def create_total_sex_files(
    ages_array, years_array, directory=DirsLocal.dir_pop_era_grid
):
    """
    Combines existing male ('m') and female ('f') NetCDF files into a total ('t') file.

    Args:
        ages_array (list of lists): The age groups to process, e.g. [[0], [65, 70]].
        years_array (list): List of years to process.
        directory (Path): The folder containing the regridded .nc files.
    """
    # Calculate total iterations for progress bar
    total_ops = len(ages_array) * len(years_array)

    print(f"Starting combination of Male + Female -> Total in {directory}")

    with tqdm(total=total_ops, desc="Creating Totals") as pbar:
        for age_group in ages_array:
            # Source string (what existing files are named with)
            src_age_str = _source_age_string(age_group)
            # Output string (what we will name the total files)
            out_age_str = _output_age_label(age_group)

            for year in years_array:
                # 1. Define filenames
                f_file = directory / f"f_{src_age_str}_{year}_era5_compatible.nc"
                m_file = directory / f"m_{src_age_str}_{year}_era5_compatible.nc"
                t_file = directory / f"t_{out_age_str}_{year}_era5_compatible.nc"

                # 2. Check existence
                if not f_file.exists() or not m_file.exists():
                    # print(f"Skipping {year} ages {age_group}: Missing source files.")
                    pbar.update(1)
                    continue

                if t_file.exists():
                    # print(f"Skipping {year} ages {age_group}: Total already exists.")
                    pbar.update(1)
                    continue

                try:
                    # 3. Open Datasets
                    # using open_dataset ensures we get coordinates/metadata
                    ds_f = xr.open_dataset(f_file)
                    ds_m = xr.open_dataset(m_file)

                    # 4. Sum variables
                    # Xarray aligns coordinates automatically.
                    # If grids match exactly (which they should), this is fast.
                    ds_total = ds_f + ds_m

                    # 5. Save
                    # Ensure zlib compression is used to save space
                    encoding = {"pop": {"zlib": True, "complevel": 5}}
                    ds_total.to_netcdf(t_file, encoding=encoding)

                except Exception as e:
                    print(f"Error processing {year} ages {age_group}: {e}")

                pbar.update(1)


# Example Usage Block
if __name__ == "__main__":
    # Define the same arrays used in your main processing script
    ages_to_combine = [[0], [65, 70, 75, 80], [75, 80]]
    years_to_combine = list(range(2000, 2021))  # Adjust as needed

    create_total_sex_files(ages_to_combine, years_to_combine)
