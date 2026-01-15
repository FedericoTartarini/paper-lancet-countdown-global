import copy

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from cartopy import crs as ccrs
from shapely.geometry import Point

from my_config import Vars, Dirs

gdf_countries = gpd.read_file(Dirs.dir_file_detailed_boundaries)


def plot_population_data(data_lancet, data_worldpop, data_isimip, population="infants"):
    data_gpw_sum = data_lancet.sum(dim=("latitude", "longitude")).sel(
        year=slice(2000, 2020)
    )

    fig, ax = plt.subplots(constrained_layout=True)

    # Plot GPW data
    ax.plot(data_gpw_sum.year, data_gpw_sum / 1e6, label="GPW")

    # Plot WorldPop data
    ax.plot(data_worldpop.year, data_worldpop[population] / 1e6, label="WorldPop")

    ax.plot(
        data_isimip.year,
        data_isimip[population] / 1e6,
        label="ISIMIP",
        linestyle="--",
    )

    ax.legend()
    ax.set_title(f"{population} Global Population")
    ax.set_ylabel("Population (millions)")
    plt.savefig(Dirs.dir_figures.value / f"worldpop_vs_gpw_global_{population}.pdf")
    plt.show()


def process_and_combine_data(data, year, population="infants"):
    _data = data.sel(year=year).to_dataframe().reset_index()

    _data["adjusted_longitude"] = _data["longitude"].apply(
        lambda x: x - 360 if x > 180 else x
    )

    geometry = [Point(xy) for xy in zip(_data.adjusted_longitude, _data.latitude)]

    _data = gpd.GeoDataFrame(_data, crs="EPSG:4326", geometry=geometry)

    _data = gpd.sjoin(_data, gdf_countries, how="inner", predicate="within")

    if "demographic_totals" in _data.columns:
        _data = _data.rename(columns={"demographic_totals": population})

    return _data[[population, "ISO_3_CODE"]].groupby("ISO_3_CODE").sum()


def plot_map_comparison(data, population, year):
    # Reset index of diff_gdf if needed and merge it with gdf_countries
    diff_gdf = pd.merge(gdf_countries[["geometry", "ISO_3_CODE"]], data.reset_index())
    diff_gdf = gpd.GeoDataFrame(diff_gdf, geometry=diff_gdf.geometry)
    # Create the plot
    fig, ax = plt.subplots(
        1, 1, figsize=(8, 6), subplot_kw=dict(projection=Vars.map_projection)
    )  # You can adjust the dimensions as needed

    # Plotting the data with specified vmin and vmax
    diff_gdf.plot(
        "diff (%)", vmin=-50, vmax=50, ax=ax, cmap="bwr", transform=ccrs.PlateCarree()
    )

    # Create a ScalarMappable with the colormap and norm specified
    norm = mcolors.Normalize(vmin=-50, vmax=50)
    cbar = plt.cm.ScalarMappable(cmap="bwr", norm=norm)

    # Add the colorbar to the figure based on the ScalarMappable
    fig.colorbar(cbar, ax=ax, orientation="horizontal").set_label(
        "Relative Difference GPW - WorldPop (%)"
    )  # Customize your label here

    # Save the figure
    plt.savefig(
        Dirs.dir_figures.value / f"{population}_{year}_worldpop_vs_gpw_by_country.pdf"
    )
    plt.show()


def main(year_map_comparison=2019):
    # Load and combine infant and elderly population data for 1950-1999
    infants_worldpop = xr.open_dataset(Dirs.dir_pop_infants_file.value)

    infants_isimip_sum = infants_worldpop.sum(dim=("latitude", "longitude")).sel(
        year=slice(1950, 1999)
    )
    infants_worldpop_sum = infants_worldpop.sum(dim=("latitude", "longitude")).sel(
        year=slice(2000, 2020)
    )

    demographics_totals = xr.open_dataarray(Dirs.dir_file_population_before_2000.value)
    population_infants_1950_1999 = demographics_totals.sel(age_band_lower_bound=0)
    population_infants_1950_1999 /= 5  # Divide by 5 to get the number of infants

    plot_population_data(
        population_infants_1950_1999, infants_worldpop_sum, infants_isimip_sum
    )

    infants_gpw_2019_gdf_countries = process_and_combine_data(
        data=population_infants_1950_1999,
        year=year_map_comparison,
    )

    infants_worldpop_2019_countries = process_and_combine_data(
        infants_worldpop, year_map_comparison
    )

    diff_gdf = copy.deepcopy(infants_gpw_2019_gdf_countries).rename(
        columns={"infants": "infants_gpw"}
    )
    diff_gdf["infants_worldpop"] = infants_worldpop_2019_countries["infants"]
    diff_gdf["diff (%)"] = (
        (
            infants_gpw_2019_gdf_countries["infants"]
            - infants_worldpop_2019_countries["infants"]
        )
        / infants_gpw_2019_gdf_countries["infants"]
        * 100
    )

    print(diff_gdf.loc[["USA", "CHE", "IND", "CHN"]])

    plot_map_comparison(data=diff_gdf, population="infants", year=year_map_comparison)

    # Over 65
    elderly_gpw = demographics_totals.sel(age_band_lower_bound=65)

    elderly_worldpop = xr.open_dataset(Dirs.dir_pop_elderly_file.value)

    elderly_isimip_sum = elderly_worldpop.sum(dim=("latitude", "longitude")).sel(
        year=slice(1950, 1999)
    )
    elderly_worldpop_sum = elderly_worldpop.sum(dim=("latitude", "longitude")).sel(
        year=slice(2000, 2020)
    )

    plot_population_data(
        elderly_gpw, elderly_worldpop_sum, elderly_isimip_sum, population="elderly"
    )

    elderly_gpw_2019_gdf_countries = process_and_combine_data(
        data=elderly_gpw,
        year=year_map_comparison,
        population="elderly",
    )

    elderly_worldpop_2019_countries = process_and_combine_data(
        data=elderly_worldpop,
        year=year_map_comparison,
        population="elderly",
    )

    diff_gdf = copy.deepcopy(elderly_gpw_2019_gdf_countries).rename(
        columns={"elderly": "elderly_gpw"}
    )
    diff_gdf["elderly_worldpop"] = elderly_worldpop_2019_countries["elderly"]
    diff_gdf["diff (%)"] = (
        (
            elderly_gpw_2019_gdf_countries["elderly"]
            - elderly_worldpop_2019_countries["elderly"]
        )
        / elderly_gpw_2019_gdf_countries["elderly"]
        * 100
    )

    print(diff_gdf.loc[["USA", "CHE", "IND", "CHN", "FRA", "AUS", "DEU"]])

    plot_map_comparison(data=diff_gdf, population="elderly", year=year_map_comparison)


if __name__ == "__main__":
    main(year_map_comparison=2020)
    pass
