import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from my_config import DirsLocal
from python_code.shared_functions import (
    read_pop_data_processed,
    calculate_exposure_population,
)

# Set style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


def calculate_severity_exposure():
    print("Loading Population Data...")
    pop_inf, pop_eld, _, _ = read_pop_data_processed(get_pop_75=True)

    print("Loading Annual Severity Metrics...")
    input_dir = DirsLocal.dir_results / "ehf_severity_annual"
    files = sorted(input_dir.glob("severity_summary_*.nc"))

    if not files:
        raise FileNotFoundError(
            "No annual severity summary files found. Run Step 1 first."
        )

    ds_sev = xr.open_mfdataset(files, combine="by_coords", parallel=True)

    # We focus on Elderly (>65) for the main severity analysis
    # You can duplicate this for Infants if needed
    population = pop_eld
    pop_label = "Elderly (>65)"

    print(f"Calculating Exposure for {pop_label}...")

    # 1. Calculate Exposure for each severity tier
    # Result is Person-Days (Billions)
    exp_severe = calculate_exposure_population(population, ds_sev, metric="days_severe")
    exp_extreme = calculate_exposure_population(
        population, ds_sev, metric="days_extreme"
    )
    exp_low = calculate_exposure_population(population, ds_sev, metric="days_low")

    # 2. Sum Globally per Year
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(
        {
            "Year": exp_severe.year.values,
            "Low Intensity": exp_low.sum(dim=["latitude", "longitude"]).compute() / 1e9,
            "Severe": exp_severe.sum(dim=["latitude", "longitude"]).compute() / 1e9,
            "Extreme": exp_extreme.sum(dim=["latitude", "longitude"]).compute() / 1e9,
        }
    ).set_index("Year")

    return df, pop_label


def plot_severity_stack(df, pop_label):
    """
    Stacked Area Chart: Shows how the COMPOSITION of heatwaves is changing.
    Are we just seeing more heatwaves, or are they getting more dangerous?
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Colors: Yellow (Low) -> Orange (Severe) -> Dark Purple (Extreme)
    colors = ["#FFD700", "#FF4500", "#4B0082"]

    ax.stackplot(
        df.index,
        df["Low Intensity"],
        df["Severe"],
        df["Extreme"],
        labels=["Low Intensity", "Severe", "Extreme"],
        colors=colors,
        alpha=0.85,
    )

    ax.set_ylabel("Exposure (Billions of Person-Days)")
    ax.set_title(
        f"Global Exposure by Heatwave Severity: {pop_label}\n(BoM Classification)"
    )
    ax.legend(loc="upper left", title="Severity Tier")
    ax.set_xlim(df.index.min(), df.index.max())

    # Add a trend line for TOTAL exposure to show the overall growth
    total = df.sum(axis=1)
    ax.plot(df.index, total, color="black", linestyle="--", alpha=0.5, label="Total")

    plt.tight_layout()
    plt.savefig(DirsLocal.dir_figures / "heatwave_exposure_severity_stacked.pdf")
    plt.show()


def plot_extreme_rise(df, pop_label):
    """
    Focus plot: Zoom in on Severe & Extreme only.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot Severe
    sns.lineplot(data=df["Severe"], color="#FF4500", linewidth=2, label="Severe", ax=ax)

    # Plot Extreme (often much lower numbers, so maybe hard to see on same scale?
    # Usually Extreme is small but rising fast).
    sns.lineplot(
        data=df["Extreme"], color="#4B0082", linewidth=2.5, label="Extreme", ax=ax
    )

    ax.set_ylabel("Exposure (Billions of Person-Days)")
    ax.set_title(f"Rise of Dangerous Heat: {pop_label}\n(Severe & Extreme Only)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(DirsLocal.dir_figures / "heatwave_exposure_severe_extreme_trend.pdf")
    plt.show()


def main():
    df, pop_label = calculate_severity_exposure()

    # Plot 1: The Big Picture (Composition)
    #
    plot_severity_stack(df, pop_label)

    # Plot 2: The Dangerous Tail
    plot_extreme_rise(df, pop_label)


if __name__ == "__main__":
    main()
