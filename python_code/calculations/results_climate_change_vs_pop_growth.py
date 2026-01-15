# from climada.entity import Exposures
import warnings

import cartopy
import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point

from my_config import Vars, Dirs
from python_code.calculations.region_raster import plot_world_map

# Suppress the specific RuntimeWarning from icecream
warnings.filterwarnings(
    "ignore", message="Failed to access the underlying source code for analysis"
)


def calculate_effect_climate_change_compared_to_pop_change(
    year_max: int = Vars.year_max_analysis.value,
):
    heatwave_metrics_files = sorted(Dirs.dir_results_heatwaves_days.value.glob("*.nc"))
    hw = xr.open_mfdataset(heatwave_metrics_files, combine="by_coords")
    elderly = xr.open_dataset(Dirs.dir_pop_elderly_file.value)
    elderly = elderly.sel(year=slice(Vars.year_min_analysis.value, year_max))
    infants = xr.open_dataset(Dirs.dir_pop_infants_file.value)
    infants = infants.sel(year=slice(Vars.year_min_analysis.value, year_max))

    """
    1. select two periods
    2. get average heatwave day during each period per grid point
    3. get average number of people during each period per grid point
    4. multiply each combination per grid point --> average heatwave day * person per grid point
    5. sum --> average heatwave day * person globally
    6. relative difference
    """

    hw = hw.sortby("latitude")
    # mean number of heatwave days per grid point for each period
    print(
        "Comparison period in the past is from",
        Vars.year_reference_start.value,
        "to",
        Vars.year_reference_end.value,
    )
    climate_past = (
        hw.transpose("latitude", "longitude", "year")["heatwaves_days"]
        .sel(year=slice(Vars.year_reference_start.value, Vars.year_reference_end.value))
        .mean(dim="year")
    )
    print("Current period is from", Vars.year_reference_end.value + 1, "to", year_max)
    climate_recent = (
        hw.transpose("latitude", "longitude", "year")["heatwaves_days"]
        .sel(year=slice(Vars.year_reference_end.value + 1, year_max))
        .mean(dim="year")
    )
    # mean number of people per grid point for each period
    elderly_past = elderly.sel(
        year=slice(Vars.year_reference_start.value, Vars.year_reference_end.value)
    ).mean(dim="year")
    elderly_recent = elderly.sel(
        year=slice(Vars.year_reference_end.value + 1, year_max)
    ).mean(dim="year")
    infants_past = infants.sel(
        year=slice(Vars.year_reference_start.value, Vars.year_reference_end.value)
    ).mean(dim="year")
    infants_recent = infants.sel(
        year=slice(Vars.year_reference_end.value + 1, year_max)
    ).mean(dim="year")

    def adjust_longitude(data):
        _data = data.assign_coords(
            longitude=(((data.longitude + 180) % 360) - 180)
        ).sortby("longitude")
        # Set areas with a population of 0 to NaN for elderly
        return _data.where(_data > 1, np.nan)

    climate_past = adjust_longitude(climate_past)
    climate_recent = adjust_longitude(climate_recent)
    elderly_past = adjust_longitude(elderly_past)
    elderly_recent = adjust_longitude(elderly_recent)
    infants_past = adjust_longitude(infants_past)
    infants_recent = adjust_longitude(infants_recent)

    # Calculating the effect for elderly
    days_person_elderly_past = climate_past * elderly_past
    days_person_elderly_recent = climate_recent * elderly_recent
    # Calculating the effect for infants
    days_person_infants_past = climate_past * infants_past
    days_person_infants_recent = climate_recent * infants_recent

    # Calculating the increase due to climate change for each group
    increase_elderly_climate = elderly_past * (climate_recent - climate_past)
    increase_infants_climate = infants_past * (climate_recent - climate_past)

    # Calculating the increase due to population growth for each group
    increase_elderly_population = (elderly_recent - elderly_past) * climate_past
    increase_infants_population = (infants_recent - infants_past) * climate_past

    # Calculating the combined effect of climate change and population growth for each group
    combined_increase_elderly = days_person_elderly_recent - days_person_elderly_past
    combined_increase_infants = days_person_infants_recent - days_person_infants_past

    percentage_increase_elderly_population = (
        increase_elderly_population.sum() / (days_person_elderly_past.sum()) * 100
    )
    percentage_increase_infants_population = (
        increase_infants_population.sum() / (days_person_infants_past.sum()) * 100
    )

    percentage_increase_elderly_climate = (
        increase_elderly_climate.sum() / (days_person_elderly_past.sum()) * 100
    )
    percentage_increase_infants_climate = (
        increase_infants_climate.sum() / (days_person_infants_past.sum()) * 100
    )

    percentage_increase_elderly = (
        combined_increase_elderly.sum() / (days_person_elderly_past.sum()) * 100
    )
    percentage_increase_infants = (
        combined_increase_infants.sum() / (days_person_infants_past.sum()) * 100
    )

    elderly_past_gdf = (
        elderly_past.to_dataframe()
        .reset_index()
        .rename(columns={"elderly": "Over 65 mean past"})
    )
    elderly_recent_gdf = (
        elderly_recent.to_dataframe()
        .reset_index()
        .rename(columns={"elderly": "Over 65 mean recent"})
    )
    infants_past_gdf = (
        infants_past.to_dataframe()
        .reset_index()
        .rename(columns={"infants": "Infants mean past"})
    )
    infants_recent_gdf = (
        infants_recent.to_dataframe()
        .reset_index()
        .rename(columns={"infants": "Infants mean recent"})
    )
    hw_past_gdf = (
        climate_past.to_dataframe()
        .reset_index()
        .rename(columns={"heatwaves_days": "Mean heatwaves days/year past"})
    )
    hw_recent_gdf = (
        climate_recent.to_dataframe()
        .reset_index()
        .rename(columns={"heatwaves_days": "Mean heatwaves days/year recent"})
    )

    # Starting with the first merge between the elderly datasets
    merged_df = pd.merge(elderly_past_gdf, elderly_recent_gdf)
    merged_df = merged_df.drop(columns="age_band_lower_bound")
    merged_df["Infants mean past"] = infants_past_gdf["Infants mean past"]
    merged_df["Infants mean recent"] = infants_recent_gdf["Infants mean recent"]
    merged_df["Mean heatwaves days/year past"] = hw_past_gdf[
        "Mean heatwaves days/year past"
    ]
    merged_df["Mean heatwaves days/year recent"] = hw_recent_gdf[
        "Mean heatwaves days/year recent"
    ]

    merged_df = merged_df.dropna()
    geometry = [Point(xy) for xy in zip(merged_df.longitude, merged_df.latitude)]

    merged_gdf = gpd.GeoDataFrame(merged_df, crs="EPSG:4326", geometry=geometry)

    merged_gdf["vulnerable population past"] = (
        merged_gdf["Over 65 mean past"] + merged_gdf["Infants mean past"]
    )

    merged_gdf["vulnerable population recent"] = (
        merged_gdf["Over 65 mean recent"] + merged_gdf["Infants mean recent"]
    )

    vulm_hw_no_cc_recent = (
        merged_gdf["vulnerable population recent"]
        * merged_gdf["Mean heatwaves days/year past"]
    ).sum() / merged_gdf["vulnerable population recent"].sum()
    print(f"vulm_hw_no_cc_recent={vulm_hw_no_cc_recent:.1f}")

    vulm_hw_cc_recent = (
        merged_gdf["vulnerable population recent"]
        * merged_gdf["Mean heatwaves days/year recent"]
    ).sum() / merged_gdf["vulnerable population recent"].sum()
    print(f"vulm_hw_cc_recent={vulm_hw_cc_recent:.1f}")

    print(
        f"percentage decrease no climate change "
        f"{((vulm_hw_cc_recent - vulm_hw_no_cc_recent) / vulm_hw_cc_recent * 100):.1f}%",
    )

    print(
        f"percentage increase with climate change "
        f"{((vulm_hw_cc_recent - vulm_hw_no_cc_recent) / vulm_hw_no_cc_recent * 100):.1f}%",
    )

    infants_hw_cc_recent = (
        merged_gdf["Infants mean recent"]
        * merged_gdf["Mean heatwaves days/year recent"]
    ).sum() / merged_gdf["Infants mean recent"].sum()
    print(f"infants_hw_cc_recent={infants_hw_cc_recent:.1f}")

    infants_hw_past = (
        merged_gdf["Infants mean past"] * merged_gdf["Mean heatwaves days/year past"]
    ).sum() / merged_gdf["Infants mean past"].sum()
    print(f"infants_hw_past={infants_hw_past:.1f}")

    infants_hw_no_cc_recent = (
        merged_gdf["Infants mean recent"] * merged_gdf["Mean heatwaves days/year past"]
    ).sum() / merged_gdf["Infants mean recent"].sum()
    print(f"infants_hw_no_cc_recent={infants_hw_no_cc_recent:.1f}")

    over65_hw_cc_recent = (
        merged_gdf["Over 65 mean recent"]
        * merged_gdf["Mean heatwaves days/year recent"]
    ).sum() / merged_gdf["Over 65 mean recent"].sum()
    print(f"over65_hw_cc_recent={over65_hw_cc_recent:.1f}")

    over65_hw_past = (
        merged_gdf["Over 65 mean past"] * merged_gdf["Mean heatwaves days/year past"]
    ).sum() / merged_gdf["Over 65 mean past"].sum()
    print(f"over65_hw_past={over65_hw_past:.1f}")

    over65_hw_no_cc_recent = (
        merged_gdf["Over 65 mean recent"] * merged_gdf["Mean heatwaves days/year past"]
    ).sum() / merged_gdf["Over 65 mean recent"].sum()
    print(f"over65_hw_no_cc_recent={over65_hw_no_cc_recent:.1f}")

    # Data for plotting
    impact_factors = ["Only Climate Change", "Combined"]
    elderly_values = [
        percentage_increase_elderly_climate["elderly"].data,
        percentage_increase_elderly["elderly"].data,
    ]
    infants_values = [
        percentage_increase_infants_climate["infants"].data,
        percentage_increase_infants["infants"].data,
    ]

    x = np.arange(len(impact_factors))  # the label locations

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Plotting for Elderly
    axs[0].bar(x, infants_values, color=["cornflowerblue", "coral", "gray"])
    axs[0].set_title("Infants")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(impact_factors)
    axs[0].set_ylabel("Rel. Increase in Person Heatwave Days (%)")

    # Plotting for Infants
    axs[1].bar(x, elderly_values, color=["steelblue", "coral", "gray"])
    axs[1].set_title("Elderlies")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(impact_factors)

    # Adding some general titles and labels
    plt.suptitle("Impact of Climate Change and Population Growth on Heatwave Days")
    plt.tight_layout(
        rect=[0, 0.03, 1, 0.95]
    )  # Adjust the rect to make room for the suptitle

    plt.savefig(Dirs.dir_figures.value / "barplots_dominant_effect_change.pdf")
    plt.show()

    return (
        climate_recent,
        climate_past,
        increase_infants_population,
        increase_elderly_population,
        increase_elderly_climate,
        increase_infants_climate,
        combined_increase_elderly,
        combined_increase_infants,
    )


def plots(year_max: int = Vars.year_max_analysis):
    (
        climate_recent,
        climate_past,
        increase_infants_population,
        increase_elderly_population,
        increase_elderly_climate,
        increase_infants_climate,
        combined_increase_elderly,
        combined_increase_infants,
    ) = calculate_effect_climate_change_compared_to_pop_change(year_max=year_max)

    countries_raster = xr.open_dataset(Dirs.dir_file_country_raster_report.value)
    land_mask = countries_raster["OBJECTID"] < 2000

    diff = climate_recent - climate_past
    diff = diff.assign_coords(longitude=(((diff.longitude + 180) % 360) - 180)).sortby(
        "longitude"
    )

    diff = diff * land_mask

    plot_world_map(diff, v_min_max=[-5, 5])

    plot_world_map(increase_infants_population["infants"], v_min_max=[-1000, 1000])
    plot_world_map(increase_elderly_population["elderly"], v_min_max=[-1000, 1000])

    increase_elderly_population_gdf = (
        increase_elderly_population.to_dataframe().reset_index()
    )
    increase_infants_population_gdf = (
        increase_infants_population.to_dataframe().reset_index()
    )

    increase_elderly_climate_gdf = increase_elderly_climate.to_dataframe().reset_index()
    increase_infants_climate_gdf = increase_infants_climate.to_dataframe().reset_index()

    combined_increase_elderly_gdf = (
        combined_increase_elderly.to_dataframe().reset_index()
    )
    combined_increase_infants_gdf = (
        combined_increase_infants.to_dataframe().reset_index()
    )

    # dominant effect: check per grid point what is the dominant effect based on the average of this period
    geometry = [
        Point(xy)
        for xy in zip(
            increase_elderly_population_gdf.longitude,
            increase_elderly_population_gdf.latitude,
        )
    ]
    increase_elderly_population_gdf = gpd.GeoDataFrame(
        increase_elderly_population_gdf, crs="EPSG:4326", geometry=geometry
    )
    increase_elderly_climate_gdf = gpd.GeoDataFrame(
        increase_elderly_climate_gdf, crs="EPSG:4326", geometry=geometry
    )

    gdf_countries = gpd.read_file(Dirs.dir_file_detailed_boundaries)

    increase_elderly_population_country = gpd.sjoin(
        increase_elderly_population_gdf, gdf_countries, how="inner", predicate="within"
    )
    increase_elderly_population_country = increase_elderly_population_country[
        ["geometry", "elderly", "ISO_3_CODE"]
    ]

    # Exclude the geometry column from the aggregation
    increase_elderly_population_country_agg = (
        increase_elderly_population_country.drop(columns="geometry")
        .groupby("ISO_3_CODE")
        .sum()
        .reset_index()
    )

    # Merge the aggregated data back with the geometry
    increase_elderly_population_country = increase_elderly_population_country_agg.merge(
        increase_elderly_population_country[
            ["ISO_3_CODE", "geometry"]
        ].drop_duplicates(),
        on="ISO_3_CODE",
    )

    # increase_elderly_population_country = (
    #     increase_elderly_population_country.groupby("ISO_3_CODE").sum().reset_index()
    # )
    increase_elderly_climate_country = gpd.sjoin(
        increase_elderly_climate_gdf, gdf_countries, how="inner", predicate="within"
    )
    increase_elderly_climate_country = increase_elderly_climate_country[
        ["geometry", "elderly", "ISO_3_CODE"]
    ]

    increase_elderly_climate_country = increase_elderly_climate_country.rename(
        columns={"elderly": "climate_change"}
    )
    increase_elderly_population_country = increase_elderly_population_country.rename(
        columns={"elderly": "population_change"}
    )

    dominant_effect_elderly = pd.merge(
        increase_elderly_climate_country, increase_elderly_population_country
    )

    # Determine the dominant effect for each country
    dominant_effect_elderly["dominant_effect"] = dominant_effect_elderly.apply(
        lambda x: (
            "Population Change"
            if x["population_change"] > x["climate_change"]
            else "Climate Change"
        ),
        axis=1,
    )

    dominant_effect_elderly = pd.merge(
        gdf_countries[["geometry", "ISO_3_CODE"]], dominant_effect_elderly, how="outer"
    )
    dominant_effect_elderly = gpd.GeoDataFrame(
        dominant_effect_elderly, geometry=dominant_effect_elderly.geometry
    )
    # Merge the geospatial data with your dataframe on the ISO codes
    color_map = {"Climate Change": "coral", "Population Change": "steelblue"}
    # Plot the map

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(8, 6),
        subplot_kw=dict(projection=Vars.map_projection),
        constrained_layout=True,
    )
    # merged_gdf.plot(column='dominant_effect', ax=ax, legend=True)
    patches = []
    for effect, color in color_map.items():
        dominant_effect_elderly[
            dominant_effect_elderly["dominant_effect"] == effect
        ].plot(
            ax=ax,
            color=color,
            transform=ccrs.PlateCarree(),
            markersize=0.01,
        )
        patches.append(mpatches.Patch(color=color, label=effect))

    # Add the legend with the custom patches
    ax.legend(handles=patches, loc="lower left")

    # Optional: Set additional options for the plot
    ax.set_title("Over 65")
    # ax.set_axis_off()
    plt.savefig(Dirs.dir_figures.value / "dominant_effect_change_countries_elderly.pdf")
    plt.show()

    geometry = [
        Point(xy)
        for xy in zip(
            increase_infants_population_gdf.longitude,
            increase_infants_population_gdf.latitude,
        )
    ]
    # increase_elderly_population_gdf = increase_elderly_population_gdf.drop(['Lon', 'Lat'], axis=1)
    increase_infants_population_gdf = gpd.GeoDataFrame(
        increase_infants_population_gdf, crs="EPSG:4326", geometry=geometry
    )
    increase_infants_climate_gdf = gpd.GeoDataFrame(
        increase_infants_climate_gdf, crs="EPSG:4326", geometry=geometry
    )
    combined_increase_infants_gdf = gpd.GeoDataFrame(
        combined_increase_infants_gdf, crs="EPSG:4326", geometry=geometry
    )

    increase_infants_population_country = gpd.sjoin(
        increase_infants_population_gdf, gdf_countries, how="inner", predicate="within"
    )
    increase_infants_population_country = increase_infants_population_country[
        ["geometry", "infants", "ISO_3_CODE"]
    ]
    # Exclude the geometry column from the aggregation
    increase_infants_population_country_agg = (
        increase_infants_population_country.drop(columns="geometry")
        .groupby("ISO_3_CODE")
        .sum()
        .reset_index()
    )

    # Merge the aggregated data back with the geometry
    increase_infants_population_country = increase_infants_population_country_agg.merge(
        increase_infants_population_country[
            ["ISO_3_CODE", "geometry"]
        ].drop_duplicates(),
        on="ISO_3_CODE",
    )
    increase_infants_climate_country = gpd.sjoin(
        increase_infants_climate_gdf, gdf_countries, how="inner", predicate="within"
    )
    increase_infants_climate_country = increase_infants_climate_country[
        ["geometry", "infants", "ISO_3_CODE"]
    ]
    # Exclude the geometry column from the aggregation
    increase_infants_population_country_agg = (
        increase_infants_population_country.drop(columns="geometry")
        .groupby("ISO_3_CODE")
        .sum()
        .reset_index()
    )

    # Merge the aggregated data back with the geometry
    increase_infants_population_country = increase_infants_population_country_agg.merge(
        increase_infants_population_country[
            ["ISO_3_CODE", "geometry"]
        ].drop_duplicates(),
        on="ISO_3_CODE",
    )

    increase_infants_climate_country = increase_infants_climate_country.rename(
        columns={"infants": "climate_change"}
    )
    increase_infants_population_country = increase_infants_population_country.rename(
        columns={"infants": "population_change"}
    )

    dominant_effect_infants = pd.merge(
        increase_infants_climate_country, increase_infants_population_country
    )

    # Determine the dominant effect for each country
    dominant_effect_infants["dominant_effect"] = dominant_effect_infants.apply(
        lambda x: (
            "Population Change"
            if x["population_change"] > x["climate_change"]
            else "Climate Change"
        ),
        axis=1,
    )

    dominant_effect_infants = pd.merge(
        gdf_countries[["geometry", "ISO_3_CODE"]], dominant_effect_infants
    )
    dominant_effect_infants = gpd.GeoDataFrame(
        dominant_effect_infants, geometry=dominant_effect_infants.geometry
    )
    # Merge the geospatial data with your dataframe on the ISO codes
    color_map = {"Climate Change": "coral", "Population Change": "steelblue"}
    # Plot the map

    # fig, ax = plt.subplots(
    #     1, 1, figsize=(8, 6), subplot_kw=dict(projection=Vars.map_projection)
    # )
    # # merged_gdf.plot(column='dominant_effect', ax=ax, legend=True)
    # patches = []
    # for effect, color in color_map.items():
    #     dominant_effect_infants[
    #         dominant_effect_infants["dominant_effect"] == effect
    #     ].plot(ax=ax, color=color, transform=ccrs.PlateCarree())
    #     patches.append(mpatches.Patch(color=color, label=effect))
    #
    # # Add the legend with the custom patches
    # ax.legend(handles=patches, loc="lower left")
    #
    # # Optional: Set additional options for the plot
    # ax.set_title("Infants")
    # # ax.set_axis_off()
    #
    # plt.savefig(dir_figures / "dominant_effect_change_countries_infants.pdf")

    # Step 1: Rename columns for clarity
    increase_infants_population_gdf.rename(
        columns={"infants": "infants_population_growth"}, inplace=True
    )
    increase_infants_population_gdf.rename(
        columns={"infants": "infants_climate_effect"}, inplace=True
    )

    # Assuming the GeoDataFrames are spatially aligned, we directly combine the data.
    # If they are not aligned, you might need a spatial join or another method to ensure alignment.

    # Step 2: Combine data into a new DataFrame
    combined_df = increase_infants_population_gdf[
        ["geometry", "infants_population_growth"]
    ].copy()
    combined_df["infants_climate_effect"] = increase_infants_climate_gdf[
        "infants"
    ].values

    # Drop rows where either effect is NaN to ensure we only compare complete data
    combined_df.dropna(
        subset=["infants_population_growth", "infants_climate_effect"], inplace=True
    )

    # Step 3: Determine the dominant effect
    combined_df["dominant_effect"] = combined_df.apply(
        lambda row: (
            "Population Change"
            if row["infants_population_growth"] > row["infants_climate_effect"]
            else "Climate Change"
        ),
        axis=1,
    )

    # Step 4: Create a new GeoDataFrame with the dominant effect
    dominant_effect_gdf_infants = gpd.GeoDataFrame(combined_df, geometry="geometry")
    dominant_effect_gdf_infants.crs = (
        dominant_effect_gdf_infants.crs
    )  # Match the CRS with the original GDF

    # Optionally, to view or save this GeoDataFrame:
    print(dominant_effect_gdf_infants.head())  # To display the first few rows
    # dominant_effect_gdf_infants.to_file("path_to_save_file.geojson", driver='GeoJSON')  # To save to a file

    # Assuming dominant_effect_gdf_elderly is your GeoDataFrame for the elderly population
    # and it has a column named 'dominant_effect'

    # Filter the GeoDataFrame for each dominant effect
    population_growth_dominant = dominant_effect_gdf_infants[
        dominant_effect_gdf_infants["dominant_effect"] == "Population Change"
    ]
    climate_effect_dominant = dominant_effect_gdf_infants[
        dominant_effect_gdf_infants["dominant_effect"] == "Climate Change"
    ]

    # Create the plot with Cartopy
    fig, ax = plt.subplots(
        figsize=(8, 6),
        subplot_kw={"projection": Vars.map_projection},
        constrained_layout=True,
    )
    ax.coastlines()  # Add coastlines

    # Transform the geometries to the correct projection with `.to_crs()`
    climate_effect_dominant.to_crs(epsg=4326).plot(
        ax=ax,
        marker="o",
        color="coral",
        markersize=0.01,
        label="Climate Change",
        categorical=True,
        alpha=1,
        transform=ccrs.PlateCarree(),
    )
    population_growth_dominant.to_crs(epsg=4326).plot(
        ax=ax,
        marker="o",
        color="steelblue",
        markersize=0.01,
        label="Population Change",
        categorical=True,
        alpha=1,
        transform=ccrs.PlateCarree(),
    )

    # Customize the plot
    ax.set_title("Dominant Effect")
    ax.set_global()  # Optionally set a global extent

    # Increase the size of the points in the legend
    plt.legend(markerscale=30)  # Increase marker scale in legend for better visibility

    plt.savefig(
        Dirs.dir_figures.value / "dominant_effect_change_grid_infants.jpeg", dpi=1200
    )
    plt.show()

    # Step 1: Rename columns for clarity
    increase_elderly_population_gdf.rename(
        columns={"elderly": "elderly_population_growth"}, inplace=True
    )
    increase_elderly_climate_gdf.rename(
        columns={"elderly": "elderly_climate_effect"}, inplace=True
    )

    # Assuming the GeoDataFrames are spatially aligned, we directly combine the data.
    # If they are not aligned, you might need a spatial join or another method to ensure alignment.

    # Step 2: Combine data into a new DataFrame
    combined_df = increase_elderly_population_gdf[
        ["geometry", "elderly_population_growth"]
    ].copy()
    combined_df["elderly_climate_effect"] = increase_elderly_climate_gdf[
        "elderly_climate_effect"
    ].values

    # Drop rows where either effect is NaN to ensure we only compare complete data
    combined_df.dropna(
        subset=["elderly_population_growth", "elderly_climate_effect"], inplace=True
    )

    # Step 3: Determine the dominant effect
    combined_df["dominant_effect"] = combined_df.apply(
        lambda row: (
            "Population Change"
            if row["elderly_population_growth"] > row["elderly_climate_effect"]
            else "Climate Change"
        ),
        axis=1,
    )

    # Step 4: Create a new GeoDataFrame with the dominant effect
    dominant_effect_gdf_elderly = gpd.GeoDataFrame(combined_df, geometry="geometry")
    dominant_effect_gdf_elderly.crs = (
        increase_elderly_population_gdf.crs
    )  # Match the CRS with the original GDF

    # Optionally, to view or save this GeoDataFrame:
    print(dominant_effect_gdf_elderly.head())  # To display the first few rows
    # dominant_effect_gdf_infants.to_file("path_to_save_file.geojson", driver='GeoJSON')  # To save to a file

    # Assuming dominant_effect_gdf_elderly is your GeoDataFrame for the elderly population
    # and it has a column named 'dominant_effect'

    # Filter the GeoDataFrame for each dominant effect
    population_growth_dominant = dominant_effect_gdf_elderly[
        dominant_effect_gdf_elderly["dominant_effect"] == "Population Change"
    ]
    climate_effect_dominant = dominant_effect_gdf_elderly[
        dominant_effect_gdf_elderly["dominant_effect"] == "Climate Change"
    ]

    # Create the plot with Cartopy
    fig, ax = plt.subplots(
        figsize=(8, 6),
        subplot_kw={"projection": Vars.map_projection},
        constrained_layout=True,
    )
    ax.coastlines()  # Add coastlines

    # Transform the geometries to the correct projection with `.to_crs()`
    climate_effect_dominant.to_crs(epsg=4326).plot(
        ax=ax,
        marker="o",
        color="coral",
        markersize=0.001,
        label="Climate Change",
        categorical=True,
        alpha=1,
        transform=ccrs.PlateCarree(),
    )

    population_growth_dominant.to_crs(epsg=4326).plot(
        ax=ax,
        marker="o",
        color="steelblue",
        markersize=0.001,
        label="Population Change",
        categorical=True,
        alpha=1,
        transform=ccrs.PlateCarree(),
    )

    # Customize the plot
    ax.set_title("Dominant Effect")
    ax.set_global()  # Optionally set a global extent

    # Increase the size of the points in the legend
    plt.legend(markerscale=30)  # Increase marker scale in legend for better visibility

    plt.savefig(
        Dirs.dir_figures.value / "dominant_effect_change_grid_elderly.jpeg", dpi=1200
    )
    plt.show()

    # Assuming dominant_effect_gdf_infants and dominant_effect_gdf_elderly are your GeoDataFrames
    # and they have a column named 'dominant_effect'

    # Filter the GeoDataFrame for each dominant effect for infants
    population_growth_dominant_infants = dominant_effect_gdf_infants[
        dominant_effect_gdf_infants["dominant_effect"] == "Population Change"
    ]
    climate_effect_dominant_infants = dominant_effect_gdf_infants[
        dominant_effect_gdf_infants["dominant_effect"] == "Climate Change"
    ]

    # Filter the GeoDataFrame for each dominant effect for the elderly
    population_growth_dominant_elderly = dominant_effect_gdf_elderly[
        dominant_effect_gdf_elderly["dominant_effect"] == "Population Change"
    ]
    climate_effect_dominant_elderly = dominant_effect_gdf_elderly[
        dominant_effect_gdf_elderly["dominant_effect"] == "Climate Change"
    ]

    # Create the plot with Cartopy for both age groups side by side
    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(16, 6),
        subplot_kw={"projection": Vars.map_projection},
        constrained_layout=True,
    )

    # Plot for infants
    axs[0].coastlines()  # Add coastlines
    climate_effect_dominant_infants.to_crs(epsg=4326).plot(
        ax=axs[0],
        marker="o",
        color="coral",
        markersize=0.01,
        label="Climate Change",
        categorical=True,
        transform=ccrs.PlateCarree(),
    )
    population_growth_dominant_infants.to_crs(epsg=4326).plot(
        ax=axs[0],
        marker="o",
        color="steelblue",
        markersize=0.01,
        label="Population Growth",
        categorical=True,
        transform=ccrs.PlateCarree(),
    )
    axs[0].set_title("Infants")
    axs[0].add_feature(
        cartopy.feature.COASTLINE, linestyle=":"
    )  # Add country borders for better context
    axs[0].legend(markerscale=40)  # Adjust marker scale in legend for better visibility

    # Plot for elderly
    axs[1].coastlines()  # Add coastlines
    climate_effect_dominant_elderly.to_crs(epsg=4326).plot(
        ax=axs[1],
        marker="o",
        color="coral",
        markersize=0.01,
        label="Climate Change",
        categorical=True,
        transform=ccrs.PlateCarree(),
    )
    population_growth_dominant_elderly.to_crs(epsg=4326).plot(
        ax=axs[1],
        marker="o",
        color="steelblue",
        markersize=0.01,
        label="Population Change",
        categorical=True,
        transform=ccrs.PlateCarree(),
    )
    axs[1].set_title("Over 65")
    axs[1].add_feature(
        cartopy.feature.COASTLINE, linestyle=":"
    )  # Add country borders for better context
    axs[1].legend(markerscale=40)  # Adjust marker scale in legend for better visibility

    # Optionally, set a global extent for both plots (or adjust as needed)
    # axs[0].set_global()
    # axs[1].set_global()

    plt.tight_layout()  # Adjust the layout to make sure everything fits without overlapping
    plt.savefig(
        Dirs.dir_figures.value / "combined_dominant_effect_change_grid.jpeg", dpi=1200
    )
    plt.show()

    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(16, 6),
        subplot_kw={"projection": Vars.map_projection},
        constrained_layout=True,
    )
    increase_infants_climate_gdf.dropna().plot(
        "infants",
        ax=axs[0],
        markersize=0.01,
        vmin=1e2,
        vmax=1e4,
        marker="o",
        cmap="viridis",
        label="Climate Change",
        transform=ccrs.PlateCarree(),
        legend=True,
    )
    axs[0].set_title("Change in Heatwaves Only")

    combined_increase_infants_gdf.dropna().plot(
        "infants",
        ax=axs[1],
        markersize=0.01,
        vmin=1e2,
        vmax=1e4,
        marker="o",
        cmap="viridis",
        label="Climate Change and Population",
        transform=ccrs.PlateCarree(),
        legend=True,
    )
    axs[1].set_title("Change in Heatwave Days and Population ")
    axs[1].add_feature(
        cartopy.feature.COASTLINE, linestyle=":"
    )  # Add country borders for better context

    combined_increase_infants_gdf["rel_diff"] = (
        100
        * (
            combined_increase_infants_gdf["infants"]
            - increase_infants_climate_gdf["infants"]
        )
        / combined_increase_infants_gdf["infants"]
    )
    plt.show()

    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(16, 6),
        subplot_kw={"projection": Vars.map_projection},
    )
    combined_increase_infants_gdf.dropna().plot(
        "rel_diff",
        vmax=100,
        vmin=-100,
        ax=axs[0],
        markersize=0.001,
        marker="o",
        cmap="viridis",
        label="Climate Change",
        transform=ccrs.PlateCarree(),
        legend=True,
    )
    axs[0].set_title("")
    plt.show()

    climate_recent_ = climate_recent.assign_coords(
        longitude=(((climate_recent.longitude + 180) % 360) - 180)
    ).sortby("longitude", ascending=False)
    climate_recent_ = climate_recent_ * land_mask
    climate_recent_gdf = climate_recent_.to_dataframe("").reset_index()

    geometry = [
        Point(xy)
        for xy in zip(climate_recent_gdf.longitude, climate_recent_gdf.latitude)
    ]
    climate_recent_gdf = gpd.GeoDataFrame(
        climate_recent_gdf, crs="EPSG:4326", geometry=geometry
    )

    fig, ax = plt.subplots(
        figsize=(8, 6), subplot_kw={"projection": Vars.map_projection}
    )

    climate_recent_gdf.plot("", cmap="Reds", vmin=0, vmax=20, ax=ax)

    ax.coastlines()
    plt.show()


if __name__ == "__main__":
    _ = calculate_effect_climate_change_compared_to_pop_change(
        year_max=Vars.year_max_analysis.value
    )
    # plots(year_max=Vars.year_max_analysis)
    pass
