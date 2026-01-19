import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr
from shapely.geometry import box

from my_config import VarsWorldPop, Dirs, Vars


def load_population_data(age_group, years, suffix="era5_compatible.nc"):
    datasets = []
    for year in years:
        # Load t_{age}_{year} file
        file_path = Dirs.dir_pop_era_grid / f"t_{age_group}_{year}_{suffix}"
        if file_path.exists():
            ds = xr.open_dataset(file_path)
            # Standardize time dimension to 'year'
            if "time" in ds.dims:
                ds = ds.rename({"time": "year"})
            datasets.append(ds)

    if not datasets:
        raise FileNotFoundError(f"No files found for age group: {age_group}")

    # Concatenate over time
    return xr.concat(datasets, dim="year")  # todo this is returning a warning


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
    plt.savefig(Dirs.dir_figures_interim / f"pop_data_{label}_{year}.png")
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
    plt.savefig(Dirs.dir_figures_interim / "pop_data_trends.png")
    plt.show()


def main(plot=True):
    years_range = range(2000, 2026)  # todo this should not be hardcoded

    infants_worldpop = load_population_data(age_group="under_1", years=years_range)
    elderly_worldpop = load_population_data(age_group="65_over", years=years_range)

    # Load and process data before 2000
    demographics_totals = xr.open_dataarray(Dirs.dir_file_population_before_2000)
    demographics_totals = demographics_totals.sel(year=slice(1950, 1999))
    infants_lancet = demographics_totals.sel(age_band_lower_bound=0).sel(
        year=VarsWorldPop.get_slice_years(period="before")
    )
    infants_lancet /= 5  # Divide by 5 to get the number of infants

    elderly_lancet = demographics_totals.sel(age_band_lower_bound=65).sel(
        year=VarsWorldPop.get_slice_years(period="before")
    )

    # Prepare historical datasets
    # Note: We keep variable name as 'pop' to avoid confusion and allow generic handling
    infants_lancet = infants_lancet.to_dataset().rename({"demographic_totals": "pop"})
    elderly_lancet = elderly_lancet.to_dataset().rename({"demographic_totals": "pop"})

    # Combine: simply concatenate historical (pre-2000) and new (2000-2025)
    # No extrapolation needed as we have data up to 2025
    infants_pop_analysis = xr.concat(
        [infants_lancet, infants_worldpop], dim="year"
    ).transpose("year", "latitude", "longitude")  # todo FutureWarning

    elderly_pop_analysis = xr.concat(
        [elderly_lancet, elderly_worldpop], dim="year"
    ).transpose("year", "latitude", "longitude")

    # Save the results to NetCDF files
    if Dirs.dir_pop_infants_file.exists():
        os.remove(Dirs.dir_pop_infants_file)
    infants_pop_analysis.to_netcdf(Dirs.dir_pop_infants_file)

    if Dirs.dir_pop_elderly_file.exists():
        os.remove(Dirs.dir_pop_elderly_file)
    elderly_pop_analysis.to_netcdf(Dirs.dir_pop_elderly_file)

    # elderly above 75 (only available for recent years in this flow)
    elderly_worldpop_75 = load_population_data(age_group="75_over", years=years_range)
    # Just save, no extrapolation
    if Dirs.dir_pop_above_75_file.exists():
        os.remove(Dirs.dir_pop_above_75_file)
    elderly_worldpop_75.to_netcdf(Dirs.dir_pop_above_75_file)

    if not plot:
        return

    plot_population_trends(
        # todo the numbers below should not be hardcoded
        inf_worldpop=infants_pop_analysis.sel(year=slice(2000, 2025)),
        eld_worldpop=elderly_pop_analysis.sel(year=slice(2000, 2025)),
        eld_75=elderly_worldpop_75,
        totals_lancet=demographics_totals,
    )

    # Sample plots for the latest available year in range
    sample_year = Vars.year_max_analysis
    plot_population_data(
        data=infants_worldpop, label="inf", year=sample_year, v_max=20000, bounds=None
    )
    plot_population_data(
        elderly_worldpop, label="eld", year=sample_year, v_max=20000, bounds=None
    )

    fig, ax = plt.subplots()
    plot_data = infants_pop_analysis.sum(dim=["latitude", "longitude"])
    ax.scatter(plot_data.year, plot_data.pop)
    ax.set_title("Total Infants (pop)")
    plt.show()
    plt.savefig(Dirs.dir_figures_interim / "total_infants_over_time.png")

    fig, ax = plt.subplots()
    plot_data = elderly_pop_analysis.sum(dim=["latitude", "longitude"])
    ax.scatter(plot_data.year, plot_data.pop)
    ax.set_title("Total Elderly (pop)")
    plt.show()
    plt.savefig(Dirs.dir_figures_interim / "total_elderly_over_time.png")


if __name__ == "__main__":
    main(plot=True)
