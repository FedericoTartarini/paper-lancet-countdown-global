"""
# Exposure to change in heatwave occurrance
"""

# todo not sure what this file is for. I have not cleaned the code

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from my_config import (
    year_reference_start,
    year_reference_end,
    dir_results_pop_exposure,
    dir_results_heatwaves_days,
)
from python_code.shared_functions import read_pop_data_processed


def main():

    population_infants_worldpop, population_elderly_worldpop, population_worldpop = (
        read_pop_data_processed()
    )

    heatwave_metrics_files = sorted(dir_results_heatwaves_days.glob("*.nc"))
    heatwave_metrics = xr.open_mfdataset(heatwave_metrics_files, combine="by_coords")

    heatwaves_metrics_reference = heatwave_metrics.sel(
        year=slice(year_reference_start, year_reference_end)
    ).mean(dim="year")
    heatwave_metrics_delta = heatwave_metrics - heatwaves_metrics_reference

    population_elderly_worldpop = population_elderly_worldpop.sortby(
        "latitude", ascending=False
    )

    exposures_over65 = (
        heatwave_metrics_delta["heatwaves_days"].transpose(
            "year", "latitude", "longitude"
        )
        * population_elderly_worldpop.transpose("year", "latitude", "longitude")[
            "elderly"
        ]
    )

    exposures_over65 = exposures_over65.drop_vars("age_band_lower_bound")
    exposures_over65 = exposures_over65.rename("heatwaves_days")

    # exposures_over65.to_netcdf(
    #     dir_results_pop_exposure
    #     / f"heatwave_exposure_change_over65_multi_threshold_{year_min_analysis}-{year_max_analysis}_worldpop.nc"
    # )

    exposures_infants = (
        heatwave_metrics_delta["heatwaves_days"].transpose(
            "year", "latitude", "longitude"
        )
        * population_infants_worldpop.transpose("year", "latitude", "longitude")[
            "infants"
        ]
    )

    # exposures_infants = exposures_infants.drop_vars("age_band_lower_bound")
    # exposures_infants = exposures_infants.rename("heatwaves_days")
    #
    # exposures_infants.to_netcdf(
    #     dir_results_pop_exposure
    #     / f"heatwave_exposure_change_infants_multi_threshold_{year_min_analysis}-{year_max_analysis}_worldpop.nc"
    # )
    #
    # total_exposures_over65 = exposures_over65.sum(
    #     dim=["latitude", "longitude"]
    # ).to_dataframe()
    # total_exposures_infants = exposures_infants.sum(
    #     dim=["latitude", "longitude"]
    # ).to_dataframe()
    #
    # total_exposures_over65.to_csv(
    #     dir_results_pop_exposure / "heatwave_exposure_change_totals_over65_worldpop.csv"
    # )
    # total_exposures_infants.to_csv(
    #     dir_results_pop_exposure
    #     / "heatwave_exposure_change_totals_infants_worldpop.csv"
    # )

    weighted_mean_over65 = (
        (
            exposures_over65
            / population_elderly_worldpop.sum(dim=["latitude", "longitude"])
        )
        .where(~np.isnan(population_elderly_worldpop))
        .sum(dim=["latitude", "longitude"], skipna=True)
        .to_dataframe()
    ).drop("age_band_lower_bound", axis=1)
    weighted_mean_infants = (
        (
            exposures_infants
            / population_infants_worldpop.sum(dim=["latitude", "longitude"])
        )
        .where(~np.isnan(population_infants_worldpop))
        .sum(dim=["latitude", "longitude"], skipna=True)
        .to_dataframe()
        .drop("age_band_lower_bound", axis=1)
    )
    weighted_mean_over65.to_csv(
        dir_results_pop_exposure / "heatwave_days_change_weighted_over65_worldpop.csv"
    )
    weighted_mean_infants.to_csv(
        dir_results_pop_exposure / "heatwave_days_change_weighted_infants_worldpop.csv"
    )

    # Get the grid weighting factor from the latitude
    cos_lat = np.cos(np.radians(heatwave_metrics.latitude))

    heatwave_metrics_delta_mean = (
        (heatwave_metrics_delta * cos_lat)
        .where(~np.isnan(population_elderly_worldpop.max(dim="year")))
        .mean(dim=["latitude", "longitude"], skipna=True)
        .to_dataframe()
        .drop("age_band_lower_bound", axis=1)
    )

    heatwave_metrics_delta_mean = (
        (heatwave_metrics_delta["heatwaves_days"] * cos_lat)
        .where(~np.isnan(population_elderly_worldpop.max(dim="year")))
        .mean(dim=["latitude", "longitude"], skipna=True)
        .to_dataframe()
        .drop("age_band_lower_bound", axis=1)
    )

    population_elderly_worldpop.max(dim="year")

    f, ax = plt.subplots(1, 1)

    # heatwave_metrics_delta_mean.plot(ax=ax, label="Global mean (land)")

    weighted_mean_infants.infants.plot(ax=ax, label="Weighted mean (infants)")
    weighted_mean_over65.elderly.plot(ax=ax, label="Weighted mean (over-65)")
    ax.legend()
    ax.set(
        ylabel="Heatwave days",
        title="Change in heatwave days relative to 1986-2005 baseline",
    )
    plt.show()
    # f.savefig(RESULTS_FOLDER / 'weighted heatwave change comparison worldpop.png', dpi=300)
    # f.savefig(RESULTS_FOLDER / 'weighted heatwave change comparison worldpop.pdf')

    f, ax = plt.subplots(1, 1)
    n = 10
    # heatwave_metrics_delta_mean.rolling(n).mean().plot(ax=ax, label="Global mean (land)")

    weighted_mean_infants.rolling(n).mean().plot(ax=ax, label="Weighted mean (infants)")
    weighted_mean_over65.rolling(n).mean().plot(ax=ax, label="Weighted mean (over-65)")
    ax.legend()
    ax.set(
        ylabel="Heatwave days",
        title=f"Change in heatwave days relative to 1986-2005 baseline,\n {n}-year rolling mean",
    )
    plt.show()
    # f.savefig(RESULTS_FOLDER / 'weighted rolling heatwave change comparison worldpop.png', dpi=300)
    # f.savefig(RESULTS_FOLDER / 'weighted rolling heatwave change comparison worldpop.pdf')

    print(heatwave_metrics_delta_mean.rolling(n).mean())

    n = 10
    rolling_stats = pd.DataFrame(
        {
            "heatwave_days_change_land": heatwave_metrics_delta_mean.rolling(n).mean()[
                "elderly"
            ],
            "heatwave_days_change_infants": weighted_mean_infants.rolling(n).mean()[
                "infants"
            ],
            "heatwave_days_change_over65": weighted_mean_over65.rolling(n).mean()[
                "elderly"
            ],
        }
    ).dropna()
    rolling_stats.to_csv(
        dir_results_pop_exposure
        / "heatwave_days_change_10_year_rolling_mean_worldpop.csv"
    )
    (weighted_mean_infants - heatwave_metrics_delta_mean).rolling(n).mean()[
        "infants"
    ].plot()
    plt.show()

    (weighted_mean_over65 - heatwave_metrics_delta_mean).rolling(n).mean()[
        "elderly"
    ].plot()
    plt.show()


if __name__ == "__main__":
    main()
    pass
