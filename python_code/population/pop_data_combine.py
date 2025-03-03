import numpy as np
import xarray as xr

import matplotlib.pyplot as plt

from my_config import (
    dir_pop_era_grid,
    dir_results,
    year_max_analysis,
)


def load_population_data(group, gender, years, suffix):
    return {
        year: xr.open_dataset(
            dir_pop_era_grid / f"{gender}_{group}_{str(year)}_{suffix}"
        )
        for year in years
    }


# Function to sum male and female datasets and rename 'time' dimension to 'year'
def combine_and_rename(data_m, data_f):
    combined_data = {year: data_f[year] + data_m[year] for year in data_m}
    for year in combined_data:
        combined_data[year] = combined_data[year].rename({"time": "year"})
    return xr.concat(combined_data.values(), dim="year")


def concatenate_and_extrapolate(old_data, new_data):
    combined_data = xr.concat([old_data, new_data], dim="year")
    return xr.concat(
        [
            combined_data,
            combined_data.interp(
                year=extrapolated_years, kwargs={"fill_value": "extrapolate"}
            ),
        ],
        "year",
    ).load()


if __name__ == "__main__":

    # Load and combine infant and elderly population data for 2000-2020
    years_range = np.arange(2000, 2020)
    infants_m = load_population_data("0", "m", years_range, "era5_compatible.nc")
    infants_f = load_population_data("0", "f", years_range, "era5_compatible.nc")
    elderly_m = load_population_data(
        "65_70_75_80", "m", years_range, "era5_compatible.nc"
    )
    elderly_f = load_population_data(
        "65_70_75_80", "f", years_range, "era5_compatible.nc"
    )

    # Combine and process data for both infants and elderly
    infants_worldpop_2000_2020 = combine_and_rename(infants_m, infants_f)
    elderly_worldpop_2000_2020 = combine_and_rename(elderly_m, elderly_f)

    # Load and combine infant and elderly population data for 1950-1999
    demographics_totals_file = (
        dir_results / "hybrid_pop" / "Hybrid Demographics 1950-2020.nc"
    )  # files generated for lancet report 2023

    demographics_totals = xr.open_dataarray(demographics_totals_file)
    population_infants_1950_1999 = demographics_totals.sel(age_band_lower_bound=0).sel(
        year=slice(1950, 1999)
    )
    population_infants_1950_1999 /= 5  # Divide by 5 to get the number of infants
    demographics_totals = xr.open_dataarray(demographics_totals_file)
    elderly_1950_1999 = demographics_totals.sel(age_band_lower_bound=65).sel(
        year=slice(1950, 1999)
    )

    # fig, axs = plt.subplots(1, 2, sharey=True)
    # inf_sum = infants_worldpop_2000_2020.sum(dim=["latitude", "longitude"])
    # eld_sum = elderly_worldpop_2000_2020.sum(dim=["latitude", "longitude"])
    # axs[1].scatter(inf_sum.year, inf_sum.pop / 10**6, label="Infants")
    # axs[1].scatter(eld_sum.year, eld_sum.pop / 10**6, label="Elderly")
    # axs[1].set(xlabel="Year", ylabel="")
    # axs[1].legend()
    # for age_band in [0, 65]:
    #     pop_data = demographics_totals.sel(age_band_lower_bound=age_band).sel(
    #         year=slice(1950, 1999)
    #     )
    #     pop_data = pop_data.sum(dim=["latitude", "longitude"])
    #     if age_band == 0:
    #         pop_data /= 5
    #     axs[0].scatter(pop_data.year, pop_data / 10**6, label=f"Age band {age_band}")
    # axs[0].set(xlabel="Year", ylabel="Population (millions)")
    # axs[0].legend()
    # plt.tight_layout()
    # plt.show()

    # Combine data for all years (1950-2020) and extrapolate to 2023
    extrapolated_years = np.arange(2020, year_max_analysis + 1)
    population_infants_1950_1999 = population_infants_1950_1999.to_dataset().rename(
        {"demographic_totals": "infants"}
    )
    infants_worldpop_2000_2020 = infants_worldpop_2000_2020.rename({"pop": "infants"})
    population_infants_worldpop = concatenate_and_extrapolate(
        population_infants_1950_1999, infants_worldpop_2000_2020
    )
    population_infants_worldpop = population_infants_worldpop.transpose(
        "year", "latitude", "longitude"
    )

    elderly_1950_1999 = elderly_1950_1999.to_dataset().rename(
        {"demographic_totals": "elderly"}
    )
    elderly_worldpop_2000_2020 = elderly_worldpop_2000_2020.rename({"pop": "elderly"})

    population_elderly_worldpop = concatenate_and_extrapolate(
        elderly_1950_1999, elderly_worldpop_2000_2020
    )
    population_elderly_worldpop = population_elderly_worldpop.transpose(
        "year", "latitude", "longitude"
    )

    # Save the results to NetCDF files
    population_infants_worldpop.to_netcdf(
        dir_results
        / f"hybrid_pop"
        / f"worldpop_infants_1950_{year_max_analysis}_era5_compatible.nc"
    )
    population_elderly_worldpop.to_netcdf(
        dir_results
        / f"hybrid_pop"
        / f"worldpop_elderly_1950_{year_max_analysis}_era5_compatible.nc"
    )

    fig, ax = plt.subplots()
    plot_data = population_infants_worldpop.sum(dim=["latitude", "longitude"])
    ax.scatter(plot_data.year, plot_data.infants)
    plt.show()

    fig, ax = plt.subplots()
    plot_data = population_elderly_worldpop.sum(dim=["latitude", "longitude"])
    ax.scatter(plot_data.year, plot_data.elderly)
    plt.show()
