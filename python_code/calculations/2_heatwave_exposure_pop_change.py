"""
# Exposure to change in heatwave occurrence

> Using upscaled population data for pre-2000, this is an approximation! Needs to be shown as such on the graphs
> Using VERY ROUGH ESTIMATE of yearly newborn pop, this is EVEN MORE ROUGH for the pre-2000 data
"""

import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr

from my_config import (
    Vars,
    Dirs,
)
from python_code.shared_functions import (
    read_pop_data_processed,
    calculate_exposure_population,
)

# Set seaborn style for cleaner plots
sns.set_theme(style="whitegrid")


def load_exposure_change_data():
    """Load the exposure change files."""
    path_days_eld_change = Dirs.dir_file_elderly_exposure_change
    path_days_inf_change = Dirs.dir_file_infants_exposure_change

    ds_days_eld = xr.open_dataset(path_days_eld_change)["heatwave_days"]
    ds_days_inf = xr.open_dataset(path_days_inf_change)["heatwave_days"]

    return {
        "days_eld_change": ds_days_eld,
        "days_inf_change": ds_days_inf,
    }


def main():
    # 1. Load Population Data
    print("Reading Population Data...")
    # We use get_pop_75=True to get all data, though we primarily save infants and elderly (65+)
    pop_inf, pop_eld, _, pop_75 = read_pop_data_processed(get_pop_75=True)

    # 2. Load Heatwave Metrics
    print("Reading Heatwave Metrics...")
    heatwave_metrics_files = sorted(Dirs.dir_results_heatwaves.glob("*.nc"))
    heatwave_metrics = xr.open_mfdataset(
        heatwave_metrics_files, combine="by_coords", parallel=True
    )

    # 3. Calculate Deltas
    #    Difference from the mean number of days and number of events in reference period
    print(
        f"Calculating Baseline ({Vars.year_reference_start}-{Vars.year_reference_end})..."
    )

    heatwaves_metrics_reference = heatwave_metrics.sel(
        year=slice(Vars.year_reference_start, Vars.year_reference_end)
    ).mean(dim="year")

    heatwave_metrics_delta = heatwave_metrics - heatwaves_metrics_reference

    # --- METRIC A: CHANGE IN PERSON-DAYS ---

    print("\n--- Calculating Exposure to Change in Heatwave Days ---")

    # Infants
    exp_days_inf_change = calculate_exposure_population(
        data=pop_inf, heatwave_metrics=heatwave_metrics_delta, metric="heatwave_days"
    )

    # Elderly (>65)
    exp_days_eld_change = calculate_exposure_population(
        data=pop_eld, heatwave_metrics=heatwave_metrics_delta, metric="heatwave_days"
    )

    # Save
    print(f"Saving Exposure Change to {Dirs.dir_file_elderly_exposure_change}...")
    exp_days_eld_change.to_netcdf(Dirs.dir_file_elderly_exposure_change)

    print(f"Saving Exposure Change to {Dirs.dir_file_infants_exposure_change}...")
    exp_days_inf_change.to_netcdf(Dirs.dir_file_infants_exposure_change)

    # --- PLOTTING ---
    # We plot directly here using the in-memory data to calculate weighted means

    print("\n--- Generating Plots ---")

    def calculate_weighted_mean_change(exposure, population):
        """
        Calculates Global Weighted Mean Change = Sum(Exposure Change) / Sum(Population)
        This represents: "How many MORE heatwave days the average person experienced compared to baseline"
        """
        total_exposure = exposure.sum(dim=["latitude", "longitude"], skipna=True)
        total_pop = population.sum(dim=["latitude", "longitude"], skipna=True)
        return total_exposure / total_pop

    weighted_mean_infants = calculate_weighted_mean_change(exp_days_inf_change, pop_inf)
    weighted_mean_over65 = calculate_weighted_mean_change(exp_days_eld_change, pop_eld)

    # Plotting
    f, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    weighted_mean_infants.to_pandas()["pop"].plot(
        ax=ax, label="Infants (<1 year)", linewidth=2
    )
    weighted_mean_over65.to_pandas()["pop"].plot(
        ax=ax, label="Elderly (≥65 years)", linewidth=2
    )

    ax.legend()
    ax.set_ylabel("Change in Heatwave Days (per person)")
    ax.set_xlabel("Year")
    ax.set_title(
        f"Change in Heatwave Exposure relative to {Vars.year_reference_start}-{Vars.year_reference_end} Baseline\n(Population Weighted Mean)"
    )

    # Add a zero line for reference
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    plt.savefig(
        Dirs.dir_figures / "exposure_change_heatwave_days_population_weighted_mean.pdf"
    )

    plt.show()

    print("\n✅ Exposure change calculation and plotting complete.")


if __name__ == "__main__":
    main()
