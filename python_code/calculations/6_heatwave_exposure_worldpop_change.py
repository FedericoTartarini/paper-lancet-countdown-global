import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
from my_config import Vars, Dirs
from python_code.shared_functions import read_pop_data_processed

# Set style for professional plots
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


def calculate_area_weighted_mean(da_metric):
    """
    Calculates the global mean over land, weighted by grid cell size.
    """
    # 1. Create weights based on latitude
    weights = np.cos(np.deg2rad(da_metric.latitude))
    weights.name = "weights"

    # 2. Apply weights
    da_weighted = da_metric.weighted(weights)

    # 3. Mean over Lat/Lon (Preserving Year)
    return da_weighted.mean(dim=["latitude", "longitude"])


def calculate_pop_weighted_mean(da_metric, da_pop):
    """
    Calculates the mean weighted by population density.
    """
    # Align dimensions explicitly
    da_pop = da_pop.transpose("year", "latitude", "longitude")
    da_metric = da_metric.transpose("year", "latitude", "longitude")

    # Mask to ensure validity
    mask = da_pop.notnull() & da_metric.notnull()

    total_pop_per_year = da_pop.where(mask).sum(dim=["latitude", "longitude"])
    total_exposure_per_year = (
        (da_metric * da_pop).where(mask).sum(dim=["latitude", "longitude"])
    )

    return total_exposure_per_year / total_pop_per_year


def main():
    print("Loading Data...")
    # 1. Load Population (75+ is not needed for this plot)
    pop_inf, pop_eld, _, _ = read_pop_data_processed(get_pop_75=True)

    # 2. Load Heatwaves
    heatwave_metrics_files = sorted(Dirs.dir_results_heatwaves.glob("*.nc"))
    ds_hw = xr.open_mfdataset(
        heatwave_metrics_files, combine="by_coords", parallel=True
    )
    hw_days = ds_hw["heatwave_days"]

    # 3. Calculate Change relative to Baseline
    print(
        f"Calculating Baseline ({Vars.year_reference_start}-{Vars.year_reference_end})..."
    )
    baseline = hw_days.sel(
        year=slice(Vars.year_reference_start, Vars.year_reference_end)
    ).mean(dim="year")

    hw_delta = hw_days - baseline

    # 4. Calculate Weighted Means
    print("Calculating Weighted Means...")

    # Explicitly compute() to load into memory
    ts_climate = calculate_area_weighted_mean(hw_delta).compute()
    ts_pop_inf = calculate_pop_weighted_mean(hw_delta, pop_inf).compute()
    ts_pop_eld = calculate_pop_weighted_mean(hw_delta, pop_eld).compute()

    # --- FIX: ROBUST DATA EXTRACTION ---
    # Safely extract the 'year' array
    if "year" in ts_climate.coords:
        years = ts_climate.coords["year"].values
    elif "time" in ts_climate.coords:
        years = ts_climate.coords["time"].dt.year.values
    else:
        raise ValueError("Could not find 'year' coordinate in results.")

    # Create DataFrame using .to_numpy() to strictly ensure numerical arrays
    df = pd.DataFrame(
        {
            "Year": years,
            "Global Land Mean (Climate)": ts_climate.to_numpy(),
            "Infants Experience": ts_pop_inf["pop"].to_numpy(),
            "Elderly Experience": ts_pop_eld["pop"].to_numpy(),
        }
    )

    # Ensure Year is integer and set index
    df["Year"] = df["Year"].astype(int)
    df = df.set_index("Year")

    # Sanity check: Ensure all data is float (this will raise a clear error if not)
    df = df.astype(float)

    # 5. Plotting
    print("Generating Plots...")

    # Calculate 10-year rolling mean
    df_rolling = df.rolling(window=10, center=True).mean()

    # --- FIGURE 1: Exposure Comparison ---
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(
        data=df_rolling["Global Land Mean (Climate)"],
        ax=ax,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Global Land Mean (Climate Only)",
    )

    sns.lineplot(
        data=df_rolling["Infants Experience"],
        ax=ax,
        color="cornflowerblue",
        linewidth=2.5,
        label="Infants (Weighted Exposure)",
    )

    sns.lineplot(
        data=df_rolling["Elderly Experience"],
        ax=ax,
        color="coral",
        linewidth=2.5,
        label="Elderly (Weighted Exposure)",
    )

    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.axvspan(
        Vars.year_reference_start,
        Vars.year_reference_end,
        color="gray",
        alpha=0.1,
        label="Baseline Period",
    )

    ax.set_ylabel("Change in Heatwave Days (days/year)")
    ax.set_title(
        "Are people experiencing more heatwaves than the planet?\n(Population-Weighted vs. Area-Weighted Change)"
    )
    ax.legend()
    ax.set_xlim(Vars.year_min_analysis, Vars.year_max_analysis)

    # Annotate the gap (Demographic Penalty)
    try:
        valid_years = df_rolling.dropna().index
        if len(valid_years) > 0:
            last_year = valid_years[-1]
            gap_val = (
                df_rolling.loc[last_year, "Elderly Experience"]
                - df_rolling.loc[last_year, "Global Land Mean (Climate)"]
            )

            ax.annotate(
                f"Demographic Penalty:\n{gap_val:.1f} days",
                xy=(last_year, df_rolling.loc[last_year, "Elderly Experience"]),
                xytext=(
                    last_year - 5,
                    df_rolling.loc[last_year, "Elderly Experience"] + 2,
                ),
                arrowprops=dict(facecolor="black", shrink=0.05),
            )
    except Exception as e:
        print(f"Skipping annotation: {e}")

    plt.tight_layout()
    plt.savefig(Dirs.dir_figures / "heatwave_exposure_trends_comparison.pdf")
    plt.show()

    # --- FIGURE 2: Excess Exposure ---
    fig, ax = plt.subplots(figsize=(8, 5))
    excess_exposure = (
        df_rolling["Elderly Experience"] - df_rolling["Global Land Mean (Climate)"]
    )

    sns.lineplot(
        x=excess_exposure.index, y=excess_exposure.values, ax=ax, color="darkred"
    )
    ax.fill_between(
        excess_exposure.index, 0, excess_exposure.values, color="darkred", alpha=0.1
    )

    ax.set_ylabel("Excess Heatwave Days")
    ax.set_title(
        "The 'Urban/Populated' Heat Penalty\n(Difference between Person-Weighted and Area-Weighted)"
    )
    ax.axhline(0, color="black")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
