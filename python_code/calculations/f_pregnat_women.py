"""
Calculates the estimated annual population of pregnant women (in Person-Years) by country.

METHODOLOGY & ASSUMPTIONS:
--------------------------
1. PROXY DATA:
   We lack direct data on the distribution of pregnant women. We use gridded/country-level
   Live Births (0-1 year olds) as the proxy.

2. TOTAL PREGNANCY ESTIMATION (INCIDENCE):
   Live Births are only a subset of total pregnancies. We adjust the count to include:
   - Stillbirths: Using UNICEF country-specific rates.
   - Pregnancy Loss (Miscarriage/Abortion): Since country-level data is unavailable,
     we apply a global correction factor (ratio of losses to live births).

3. TEMPORAL SHIFT (TIMING):
   A birth in Year T implies a pregnancy that spanned parts of Year T and Year T-1.
   Conversely, the pregnant population in Year T consists of:
   - Women giving birth in Year T (contribution ~60-75% of person-years).
   - Women giving birth in Year T+1 (contribution ~25-40% of person-years).

   Assumption: We estimate the "Effective Births" for Year T using a weighted sum:
   Effective_Births(T) = 0.75 * Births(T) + 0.25 * Births(T+1)

4. DURATION ADJUSTMENT (PREVALENCE):
   To estimate "Person-Years" (average population exposed to heat), we weight by duration:
   - Full-term pregnancies (Live + Stillbirths): Assumed 9 months (0.75 years).
   - Early-term losses (Miscarriages/Abortions): Assumed 3 months (0.25 years).

EQUATION:
---------
The estimated pregnant population (Pop) for year y is calculated as:

    Pop(y) = [ 0.75 * B(y) + 0.25 * B(y+1) ] * [ (1 + SBR) * 0.75 + (Loss_Factor * 0.25) ]

Where:
    - B(y): Live births in year y
    - SBR: Stillbirth Rate (Stillbirths / Live Births)
    - Loss_Factor: Estimated ratio of early pregnancy losses to live births (e.g., 0.35)
    - 0.75 / 0.25 (First bracket): Temporal weights for shifting births to pregnancy years.
    - 0.75 / 0.25 (Second bracket): Duration of pregnancy in years (9 months vs 3 months).

DATA SOURCES:
-------------
- Stillbirths: UNICEF (Lower/Median/Upper estimates). We use the Median.
  Missing country data is filled with the Global Median.
"""

import math
from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from my_config import FilesLocal


def import_stillbirth_rates():
    df = pd.read_excel(FilesLocal.unicef_still_births, skiprows=14)
    df = df[df["Uncertainty.Bounds*"] == "Median"]
    df.set_index("ISO.Code", inplace=True)
    data_cols = [col for col in df.columns if "20" in col]
    df = df[data_cols].stack().reset_index()
    df.rename(
        columns={"ISO.Code": "ISO", "level_1": "year", 0: "stillbirth-rate"},
        inplace=True,
    )
    df["year"] = [math.floor(float(x)) for x in df["year"]]
    return df


def plot_stillbirth_rates(df):
    f, ax = plt.subplots()
    for country in df["ISO"].unique():
        df_country = df[df["ISO"] == country]
        line_color = "lightgray"
        if country == "AZE":
            line_color = "blue"
        ax.plot(
            df_country["year"],
            df_country["stillbirth-rate"],
            label=country if country == "AZE" else "",
            color=line_color,
        )
    # Only show legend for specific countries to avoid clutter
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    ax.set_title("Stillbirth Rate by Country (per 1000 births)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Stillbirth Rate (per 1000 births)")
    plt.show()


def extrapolate_stillbirths(
    input_df,
    year_col="year",
    rate_col="sbr",
    start_year=1980,
    end_year=2025,
    floor=2.0,
    ceiling=100.0,
):
    """
    Extrapolates Stillbirth Rates (SBR) using Log-Linear Regression.
    Anchors the extrapolation to the first/last observed data points to avoid steps.
    """

    # 1. Prepare Data
    df_clean = input_df.dropna(subset=[year_col, rate_col]).sort_values(by=year_col)

    if len(df_clean) < 2:
        return pd.DataFrame()

    X = df_clean[year_col].values.reshape(-1, 1)
    y = df_clean[rate_col].values

    # Identify boundaries of real data
    min_real_year = df_clean[year_col].min()
    max_real_year = df_clean[year_col].max()
    val_at_min_year = df_clean.loc[
        df_clean[year_col] == min_real_year, rate_col
    ].values[0]
    val_at_max_year = df_clean.loc[
        df_clean[year_col] == max_real_year, rate_col
    ].values[0]

    # 2. Log-Linear Transformation
    y_safe = np.where(y <= 0, 0.001, y)
    y_log = np.log(y_safe)

    # 3. Fit the Model
    model = LinearRegression()
    model.fit(X, y_log)

    # 4. Generate Predictions for the FULL range
    years_future = np.arange(start_year, end_year + 1).reshape(-1, 1)
    y_log_pred = model.predict(years_future)
    y_pred_raw = np.exp(y_log_pred)

    # Create temporary DF for calculations
    df_pred = pd.DataFrame({"year": years_future.flatten(), "pred_raw": y_pred_raw})

    # --- 4b. CALCULATE AND APPLY DELTAS (The Fix) ---

    # Get the "predicted" value at the boundary years
    pred_at_min = df_pred.loc[df_pred["year"] == min_real_year, "pred_raw"].values[0]
    pred_at_max = df_pred.loc[df_pred["year"] == max_real_year, "pred_raw"].values[0]

    # Calculate Ratios (since we are working with exponential data, Ratio is safer than Difference)
    # If you prefer strict addition/subtraction, change this to (val - pred)
    # Using Ratio preserves the curve shape better.
    ratio_start = val_at_min_year / pred_at_min
    ratio_end = val_at_max_year / pred_at_max

    # Apply corrections
    # For years < min_real_year: shift the curve to hit the first real point
    mask_before = df_pred["year"] < min_real_year
    df_pred.loc[mask_before, "pred_raw"] = (
        df_pred.loc[mask_before, "pred_raw"] * ratio_start
    )

    # For years > max_real_year: shift the curve to hit the last real point
    mask_after = df_pred["year"] > max_real_year
    df_pred.loc[mask_after, "pred_raw"] = (
        df_pred.loc[mask_after, "pred_raw"] * ratio_end
    )

    # 5. Apply Floor & Ceiling
    df_pred["pred_raw"] = np.maximum(df_pred["pred_raw"], floor)
    df_pred["pred_raw"] = np.minimum(df_pred["pred_raw"], ceiling)

    # 6. Merge with Original Data
    # Rename for merging
    df_clean_renamed = df_clean[[year_col, rate_col]].rename(
        columns={rate_col: "stillbirth-rate"}
    )

    df_final = pd.merge(df_pred, df_clean_renamed, on="year", how="left")

    # Fill NaNs (estimates) with the corrected predictions
    # Note: Where real data exists, 'stillbirth-rate' is already present.
    # We only fill the missing years.
    df_final["stillbirth-rate"] = df_final["stillbirth-rate"].fillna(
        df_final["pred_raw"]
    )

    return df_final[["year", "stillbirth-rate"]]


df_still = import_stillbirth_rates()
plot_stillbirth_rates(df_still)

df_extrapolated = pd.DataFrame()
for country in df_still["ISO"].unique():
    # print(country)
    df_country = df_still[df_still["ISO"] == country]

    # Passing the specific country dataframe now
    df_country_extrapolated = extrapolate_stillbirths(
        df_country,
        year_col="year",
        rate_col="stillbirth-rate",
        floor=2.0,
        ceiling=100.0,  # Set your desired max value here
    )

    df_country_extrapolated["ISO"] = country
    # print(df_country_extrapolated.head())

    df_extrapolated = pd.concat(
        [df_extrapolated, df_country_extrapolated], ignore_index=True
    )

plot_stillbirth_rates(df_extrapolated)

f, ax = plt.subplots()
df_ext_aze = df_extrapolated[df_extrapolated["ISO"] == "AFG"]
df_still_aze = df_still[df_still["ISO"] == "AFG"]

# Plot the full extrapolated line (includes merged original data)
ax.plot(
    df_ext_aze["year"],
    df_ext_aze["stillbirth-rate"],
    label="Final Series (Observed + Extrapolated)",
    color="blue",
    linestyle="-",
)

# Plot the original points on top to verify they match
ax.scatter(
    df_still_aze["year"],
    df_still_aze["stillbirth-rate"],
    label="Original Observed Data",
    color="orange",
    zorder=5,
)

ax.legend()
ax.set_title("Stillbirth Rate for AZE: Merged Data")
ax.set_xlabel("Year")
ax.set_ylabel("Stillbirth Rate (per 1000 births)")
plt.show()


def simulate_pregnant_population():
    # 1. Setup the Simulation Data
    years = np.arange(2020, 2031)

    # Create a "Step" in births: 1000 until 2024, then 2000 from 2025 onwards
    births = [1000 if y < 2025 else 2000 for y in years]

    df = pd.DataFrame({"Year": years, "Births": births})

    # 2. Get "Next Year's Births" (Shift column up by -1)
    df["Births_Next_Year"] = df["Births"].shift(-1)

    # 3. Apply the Formula
    # Formula: Pop = Duration * (Contribution_Current + Contribution_Next)
    # Duration = 0.75 (9 months)
    # Split = 0.625 (Current Year) / 0.375 (Next Year)

    df["Pregnant_Pop"] = 0.75 * (
        (0.625 * df["Births"]) + (0.375 * df["Births_Next_Year"])
    )

    # 4. Compare with a "Naive" Calculation (No Time Shift)
    # Naive: Just assuming pregnant women = Births * 0.75 in the same year
    df["Naive_Pop"] = df["Births"] * 0.75

    # Drop the last row (NaN because we don't have next year's births)
    df_plot = df.dropna()

    # 5. Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Births (Bars)
    color = "tab:gray"
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Annual Live Births", color=color, fontweight="bold")
    bars = ax1.bar(
        df_plot["Year"], df_plot["Births"], color=color, alpha=0.3, label="Live Births"
    )
    ax1.tick_params(axis="y", labelcolor=color)

    # Create a second y-axis for the Population
    ax2 = ax1.twinx()
    color_est = "tab:blue"
    color_naive = "tab:red"

    ax2.set_ylabel("Avg Pregnant Population", color="black", fontweight="bold")

    # Plot Naive Approach (Red Dashed)
    (line1,) = ax2.plot(
        df_plot["Year"],
        df_plot["Naive_Pop"],
        color=color_naive,
        linestyle="--",
        marker="x",
        label="Naive Estimate (No Shift)",
    )

    # Plot Correct Approach (Blue Solid)
    (line2,) = ax2.plot(
        df_plot["Year"],
        df_plot["Pregnant_Pop"],
        color=color_est,
        linewidth=3,
        marker="o",
        label="Correct Time-Shifted Estimate",
    )

    ax2.tick_params(axis="y", labelcolor="black")

    # Title & Legend
    plt.title(
        "Simulation: Impact of Time-Shifting on Pregnant Population\n(Births jump from 1000 to 2000 in Year 2025)"
    )

    # Combine legends
    lines = [line2, line1, bars]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    plt.grid(True, alpha=0.3)
    plt.show()

    # Print the critical transition years to verify numbers
    print(df_plot[["Year", "Births", "Births_Next_Year", "Pregnant_Pop"]].iloc[3:7])


simulate_pregnant_population()
