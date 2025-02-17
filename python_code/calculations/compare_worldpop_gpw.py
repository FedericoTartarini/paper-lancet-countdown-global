import matplotlib.pyplot as plt
import xarray as xr

from my_config import dir_results, DATA_SRC
from shapely.geometry import Point
import geopandas as gpd
import copy
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from cartopy import crs as ccrs


# Load and combine infant and elderly population data for 1950-1999
demographics_totals_file = (
    dir_results / "hybrid_pop" / "Hybrid Demographics 1950-2020.nc"
)  # files generated for lancet report 2023

infants_worldpop = xr.open_dataset(
    dir_results / f"hybrid_pop" / f"worldpop_infants_1950_2024_era5_compatible.nc"
)

infants_isimip_sum = infants_worldpop.sum(dim=("latitude", "longitude")).sel(
    year=slice(1950, 1999)
)
infants_worldpop_sum = infants_worldpop.sum(dim=("latitude", "longitude")).sel(
    year=slice(2000, 2020)
)
demographics_totals = xr.open_dataarray(demographics_totals_file)
population_infants_1950_1999 = demographics_totals.sel(age_band_lower_bound=0)
population_infants_1950_1999 /= 5  # Divide by 5 to get the number of infants
infants_gpw_sum = population_infants_1950_1999.sum(dim=("latitude", "longitude")).sel(
    year=slice(2000, 2020)
)

fig, ax = plt.subplots(figsize=(6, 4))

# Plot GPW data
ax.plot(infants_gpw_sum.year, infants_gpw_sum / 1e6, label="GPW")

# Plot WorldPop data
ax.plot(infants_worldpop_sum.year, infants_worldpop_sum.infants / 1e6, label="WorldPop")

ax.plot(
    infants_isimip_sum.year,
    infants_isimip_sum.infants / 1e6,
    label="ISIMIP",
    linestyle="--",
)

ax.legend()
ax.set_title("Infants Global Population")
ax.set_ylabel("Population (millions)")
plt.savefig("python_code/figures/infants_worldpop_vs_gpw_global.pdf")
plt.show()


gdf_countries = gpd.read_file(DATA_SRC / "admin_boundaries" / "Detailed_Boundary_ADM0")

infants_gpw_2019 = population_infants_1950_1999.sel(year=2020)

infants_gpw_2019_gdf = infants_gpw_2019.to_dataframe().reset_index()

infants_gpw_2019_gdf["adjusted_longitude"] = infants_gpw_2019_gdf["longitude"].apply(
    lambda x: x - 360 if x > 180 else x
)

geometry = [
    Point(xy)
    for xy in zip(
        infants_gpw_2019_gdf.adjusted_longitude, infants_gpw_2019_gdf.latitude
    )
]

infants_gpw_2019_gdf = gpd.GeoDataFrame(
    infants_gpw_2019_gdf, crs="EPSG:4326", geometry=geometry
)

infants_gpw_2019_gdf = gpd.sjoin(
    infants_gpw_2019_gdf, gdf_countries, how="inner", predicate="within"
)

infants_gpw_2019_gdf_countries = (
    infants_gpw_2019_gdf[["demographic_totals", "ISO_3_CODE"]]
    .groupby("ISO_3_CODE")
    .sum()
)
infants_worldpop_2019 = infants_worldpop.sel(year=2020).to_dataframe().reset_index()

infants_worldpop_2019["adjusted_longitude"] = infants_worldpop_2019["longitude"].apply(
    lambda x: x - 360 if x > 180 else x
)

geometry = [
    Point(xy)
    for xy in zip(
        infants_worldpop_2019.adjusted_longitude, infants_worldpop_2019.latitude
    )
]

infants_worldpop_2019_gdf = gpd.GeoDataFrame(
    infants_worldpop_2019, crs="EPSG:4326", geometry=geometry
)

infants_worldpop_2019_gdf = gpd.sjoin(
    infants_worldpop_2019_gdf, gdf_countries, how="inner", predicate="within"
)

infants_worldpop_2019_countries = (
    infants_worldpop_2019_gdf[["infants", "ISO_3_CODE"]].groupby("ISO_3_CODE").sum()
)

diff_gdf = copy.deepcopy(infants_gpw_2019_gdf_countries).rename(
    columns={"infants": "infants_gpw"}
)
diff_gdf["infants_worldpop"] = infants_worldpop_2019_countries["infants"]
diff_gdf["diff (%)"] = (
    (
        infants_gpw_2019_gdf_countries["demographic_totals"]
        - infants_worldpop_2019_countries["infants"]
    )
    / infants_gpw_2019_gdf_countries["demographic_totals"]
    * 100
)

print(diff_gdf.loc[["USA", "CHE", "IND", "CHN"]])


MAP_PROJECTION = ccrs.EckertIII()

# Assuming gdf_countries and diff_gdf are previously defined and populated

# Reset index of diff_gdf if needed and merge it with gdf_countries
diff_gdf = pd.merge(gdf_countries[["geometry", "ISO_3_CODE"]], diff_gdf.reset_index())
diff_gdf = gpd.GeoDataFrame(diff_gdf, geometry=diff_gdf.geometry)

# Create the plot
fig, ax = plt.subplots(
    1, 1, figsize=(8, 6), subplot_kw=dict(projection=MAP_PROJECTION)
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
plt.savefig("python_code/figures/infants_worldpop_vs_gpw_by_country.pdf")
plt.show()


# Over 65
elderly_gpw = demographics_totals.sel(age_band_lower_bound=65)

elderly_worldpop = xr.open_dataset(
    dir_results / f"hybrid_pop" / f"worldpop_elderly_1950_2024_era5_compatible.nc"
)

elderly_isimip_sum = elderly_worldpop.sum(dim=("latitude", "longitude")).sel(
    year=slice(1950, 1999)
)
elderly_worldpop_sum = elderly_worldpop.sum(dim=("latitude", "longitude")).sel(
    year=slice(2000, 2020)
)
elderly_gpw_sum = elderly_gpw.sum(dim=("latitude", "longitude")).sel(
    year=slice(2000, 2020)
)

fig, ax = plt.subplots(figsize=(6, 4))

# Plot GPW data
ax.plot(elderly_gpw_sum.year, elderly_gpw_sum / 1e6, label="GPW")

# Plot WorldPop data
ax.plot(elderly_worldpop_sum.year, elderly_worldpop_sum.elderly / 1e6, label="WorldPop")

# Plot ISIMIP data (dashed line before year 2000)

ax.plot(
    elderly_isimip_sum.year,
    elderly_isimip_sum.elderly / 1e6,
    label="ISIMIP",
    linestyle="--",
)

ax.legend()
ax.set_title("Elderly Global Population")
ax.set_ylabel("Population (millions)")

plt.savefig("python_code/figures/elderlies_worldpop_vs_gpw_global.pdf")
plt.show()

elderly_gpw_2019 = elderly_gpw.sel(
    year=2020
)  # Assuming 'elderly_gpw' is your data variable for the elderly population

elderly_gpw_2019_gdf = elderly_gpw_2019.to_dataframe().reset_index()

elderly_gpw_2019_gdf["adjusted_longitude"] = elderly_gpw_2019_gdf["longitude"].apply(
    lambda x: x - 360 if x > 180 else x
)

geometry = [
    Point(xy)
    for xy in zip(
        elderly_gpw_2019_gdf.adjusted_longitude, elderly_gpw_2019_gdf.latitude
    )
]

elderly_gpw_2019_gdf = gpd.GeoDataFrame(
    elderly_gpw_2019_gdf, crs="EPSG:4326", geometry=geometry
)

elderly_gpw_2019_gdf = gpd.sjoin(
    elderly_gpw_2019_gdf, gdf_countries, how="inner", predicate="within"
)

elderly_gpw_2019_gdf_countries = (
    elderly_gpw_2019_gdf[["demographic_totals", "ISO_3_CODE"]]
    .groupby("ISO_3_CODE")
    .sum()
)

elderly_worldpop_2019 = elderly_worldpop.sel(year=2020).to_dataframe().reset_index()

elderly_worldpop_2019["adjusted_longitude"] = elderly_worldpop_2019["longitude"].apply(
    lambda x: x - 360 if x > 180 else x
)

geometry = [
    Point(xy)
    for xy in zip(
        elderly_worldpop_2019.adjusted_longitude, elderly_worldpop_2019.latitude
    )
]

elderly_worldpop_2019_gdf = gpd.GeoDataFrame(
    elderly_worldpop_2019, crs="EPSG:4326", geometry=geometry
)

elderly_worldpop_2019_gdf = gpd.sjoin(
    elderly_worldpop_2019_gdf, gdf_countries, how="inner", predicate="within"
)

elderly_worldpop_2019_countries = (
    elderly_worldpop_2019_gdf[["elderly", "ISO_3_CODE"]].groupby("ISO_3_CODE").sum()
)

diff_gdf = copy.deepcopy(elderly_gpw_2019_gdf_countries).rename(
    columns={"demographic_totals": "elderly_gpw"}
)
diff_gdf["elderly_worldpop"] = elderly_worldpop_2019_countries["elderly"]
diff_gdf["diff (%)"] = (
    (
        elderly_gpw_2019_gdf_countries["demographic_totals"]
        - elderly_worldpop_2019_countries["elderly"]
    )
    / elderly_gpw_2019_gdf_countries["demographic_totals"]
    * 100
)

print(diff_gdf.loc[["USA", "CHE", "IND", "CHN", "FRA", "AUS", "DEU"]])

# Reset index of diff_gdf if needed and merge it with gdf_countries
diff_gdf = pd.merge(gdf_countries[["geometry", "ISO_3_CODE"]], diff_gdf.reset_index())
diff_gdf = gpd.GeoDataFrame(diff_gdf, geometry=diff_gdf.geometry)

# Create the plot
fig, ax = plt.subplots(
    1, 1, figsize=(8, 6), subplot_kw=dict(projection=MAP_PROJECTION)
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
plt.savefig("python_code/figures/elderlies_worldpop_vs_gpw_by_country.pdf")
plt.show()
