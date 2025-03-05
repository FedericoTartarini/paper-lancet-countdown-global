import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from shapely.geometry import box

from my_config import (
    dir_pop_era_grid,
    year_worldpop_end,
    year_worldpop_start,
    dir_file_population_before_2000,
    dir_figures_interim,
    year_report,
    dir_pop_infants_file,
    dir_pop_elderly_file,
    dir_pop_above_75_file,
)


def load_population_data(age_group, sex, years, suffix="era5_compatible.nc"):
    return {
        year: xr.open_dataset(dir_pop_era_grid / f"{sex}_{age_group}_{year}_{suffix}")
        for year in years
    }


# Function to sum male and female datasets and rename 'time' dimension to 'year'
def combine_and_rename(data_m, data_f):
    combined_data = {year: data_f[year] + data_m[year] for year in data_m}
    for year in combined_data:
        combined_data[year] = combined_data[year].rename({"time": "year"})
    return xr.concat(combined_data.values(), dim="year")


def concatenate_and_extrapolate(old_data, new_data, years_range):
    combined_data = xr.concat([old_data, new_data], dim="year")
    return xr.concat(
        [
            combined_data,
            combined_data.interp(
                year=years_range, kwargs={"fill_value": "extrapolate"}
            ),
        ],
        "year",
    ).load()


def plot_population_data(data, label, year=2001, bounds=(0, 35, 20, 47), v_max=20000):
    # Create a figure and axis with a cartopy projection centered on longitude 0
    fig, ax = plt.subplots(
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=0)},
        constrained_layout=True,
    )
    # Add features to the map
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, edgecolor="black")
    ax.add_feature(cfeature.OCEAN)

    # Select the data for the specified year
    data = data.sel(year=year)

    # Set the CRS for the data
    data = data.rio.write_crs("EPSG:4326")

    if bounds:
        # Create a GeoDataFrame with the specified bounds
        gdf = gpd.GeoDataFrame(geometry=[box(*bounds)], crs="EPSG:4326")

        # Clip the data using the GeoDataFrame
        data = data.rio.clip(gdf.geometry.values, gdf.crs, drop=True)

    # Filter the data to remove negative values
    data = data.pop.where(data.pop >= 0)

    # Plot the population data
    data.plot(
        vmax=v_max,
        cmap="viridis",
        ax=ax,
        cbar_kwargs={"orientation": "horizontal", "pad": 0.05, "label": "Population"},
    )

    plt.title(f"{label.capitalize()} population in {year}")
    # Show the plot
    plt.savefig(dir_figures_interim / f"pop_data_{label}_{year}.png")
    plt.show()


def plot_population_trends(inf_worldpop, eld_worldpop, eld_75, totals_lancet):
    fig, axs = plt.subplots(1, 2, sharey=True)
    inf_sum = inf_worldpop.sum(dim=["latitude", "longitude"])
    eld_sum = eld_worldpop.sum(dim=["latitude", "longitude"])
    eld_75 = eld_75.sum(dim=["latitude", "longitude"])
    axs[1].scatter(inf_sum.year, inf_sum.pop / 10**6, label="Infants")
    axs[1].scatter(eld_sum.year, eld_sum.pop / 10**6, label="Above 65")
    axs[1].scatter(eld_75.year, eld_75.pop / 10**6, label="Above 75")
    axs[1].set(xlabel="Year", ylabel="")
    axs[1].legend()
    for age_band in [0, 65]:
        pop_data = totals_lancet.sel(age_band_lower_bound=age_band).sel(
            year=slice(1950, 1999)
        )
        pop_data = pop_data.sum(dim=["latitude", "longitude"])
        if age_band == 0:
            pop_data /= 5
        axs[0].scatter(pop_data.year, pop_data / 10**6, label=f"Age band {age_band}")
    axs[0].set(xlabel="Year", ylabel="Population (millions)")
    axs[0].legend()
    plt.tight_layout()
    plt.savefig(dir_figures_interim / "pop_data_trends.png")
    plt.show()


def load_and_combine_population_data(age_group, years_range):
    data_m = load_population_data(age_group=age_group, sex="m", years=years_range)
    data_f = load_population_data(age_group=age_group, sex="f", years=years_range)
    return combine_and_rename(data_m=data_m, data_f=data_f)


def main(plot=True):
    # Load and combine infant and elderly population data for 2000-2020
    years_range = np.arange(year_worldpop_start, year_worldpop_end + 1)
    infants_worldpop = load_and_combine_population_data(
        age_group="0", years_range=years_range
    )
    elderly_worldpop = load_and_combine_population_data(
        age_group="65_70_75_80", years_range=years_range
    )

    # Load and combine infant and elderly population data for 1950-1999
    demographics_totals = xr.open_dataarray(dir_file_population_before_2000)
    infants_lancet = demographics_totals.sel(age_band_lower_bound=0).sel(
        year=slice(1950, 1999)
    )
    infants_lancet /= 5  # Divide by 5 to get the number of infants

    elderly_lancet = demographics_totals.sel(age_band_lower_bound=65).sel(
        year=slice(1950, 1999)
    )

    # Combine data for all years (1950-2020) and extrapolate to 2023
    extrapolated_years = np.arange(year_worldpop_end, year_report)

    infants_lancet = infants_lancet.to_dataset().rename({"demographic_totals": "pop"})
    infants_pop_analysis = concatenate_and_extrapolate(
        infants_lancet, infants_worldpop, years_range=extrapolated_years
    )
    infants_pop_analysis = infants_pop_analysis.transpose(
        "year", "latitude", "longitude"
    )

    elderly_lancet = elderly_lancet.to_dataset().rename({"demographic_totals": "pop"})

    elderly_pop_analysis = concatenate_and_extrapolate(
        elderly_lancet, elderly_worldpop, years_range=extrapolated_years
    )
    elderly_pop_analysis = elderly_pop_analysis.transpose(
        "year", "latitude", "longitude"
    )

    # Save the results to NetCDF files
    infants_pop_analysis = infants_pop_analysis.rename({"pop": "infants"})
    elderly_pop_analysis = elderly_pop_analysis.rename({"pop": "elderly"})

    infants_pop_analysis.to_netcdf(dir_pop_infants_file)
    elderly_pop_analysis.to_netcdf(dir_pop_elderly_file)

    # elderly above 75
    elderly_worldpop_75 = load_and_combine_population_data(
        age_group="75_80", years_range=years_range
    )
    elderly_worldpop_75.to_netcdf(dir_pop_above_75_file)

    if not plot:
        return

    plot_population_trends(
        inf_worldpop=infants_worldpop,
        eld_worldpop=elderly_worldpop,
        eld_75=elderly_worldpop_75,
        totals_lancet=demographics_totals,
    )

    for year in years_range:
        plot_population_data(
            infants_worldpop, label="inf", year=int(year), v_max=20000, bounds=None
        )
        plot_population_data(
            elderly_worldpop, label="eld", year=int(year), v_max=20000, bounds=None
        )

    fig, ax = plt.subplots()
    plot_data = infants_pop_analysis.sum(dim=["latitude", "longitude"])
    ax.scatter(plot_data.year, plot_data.infants)
    plt.show()

    fig, ax = plt.subplots()
    plot_data = elderly_pop_analysis.sum(dim=["latitude", "longitude"])
    ax.scatter(plot_data.year, plot_data.elderly)
    plt.show()


if __name__ == "__main__":
    main(plot=True)
    pass
