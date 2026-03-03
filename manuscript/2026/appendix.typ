#set par(justify: true)
// Load the data
#let db = json("typst_variables.json")

#let max_year_analysis = db.methods.year_max_analysis

= Section 1: Health Hazards, Exposure, and Impact

== 1.1.1.2 Exposure of vulnerable populations to heatwaves

=== Indicator Authors
 
Dr Federico Tartarini, Prof Ollie Jay, Dr Mitchell Black

=== Methods

==== Heatwave Definition, Occurrence, and Duration

Heatwaves effects on human health is a growing concern worldwide, particularly for vulnerable populations such as Older adults and Infants.
However, there is no universally accepted definition of a heatwave, with various studies employing different temperature thresholds, durations, and metrics to characterize these events @xu2016impact.
For this analysis, we defined a heatwave as a period of three or more consecutive days in which both the daily minimum and maximum temperatures exceeded the 95th percentile of the local climatology.
This definition is based on the approach used by the World Meteorological Organization (WMO) in the "Heatwaves and Health: Guidance on Warning-System Development" @wmo2015heatwaves.
This dual-threshold definition captures both the direct heat stress caused by high daytime temperatures and the physiological strain associated with insufficient nighttime cooling @liu2024rising, @di2019verification.
Two climatological baselines were used:
- 1986-2005 reference period.
- 2007-2016 to align with the Paris Agreement.

To determine these events, we utilized daily 2-meter temperature data from the European Centre for Medium-Range Weather Forecasts (ECMWF) ERA5-Land reanalysis dataset @munoz2021era5, gridded at a 0.1° × 0.1° global resolution. 
For each grid cell and each year from 1980 to 2025, we calculated two primary metrics:

- Heatwave Duration: The total number of days per year spent during a heatwave.
- Heatwave Frequency: The total number of discrete heatwave events per year.

==== Heatwave Exposure Calculation

Exposure to heatwaves for each vulnerable group was calculated by combining heatwave occurrence data with gridded demographic datasets.

For each grid cell, the annual heatwave exposure (in Person-days) was computed as:
$ "Exposure" = "Heatwave Days" times "Population" $

Where:
- *Heatwave Days:* The total number of heatwave days in that grid cell for the year.
- *Population:* The number of individuals in the vulnerable group residing in that grid cell.

The total annual heatwave exposure for each vulnerable group was obtained by summing the Person-days across all grid cells globally.
We also present the Average heatwave days per person by dividing the total Person-days by the total population of the vulnerable group for that year.

#highlight[
Please remember to add in the methods for the counterfactual estimation from Andys analysis, which will be used to estimate how many heatwave days vulnerable populations would have experienced if climate change had not occurred, considering only demographic shifts.
]

==== Heatwave Severity (Excess Heat Factor)

#highlight[
To assess the changing intensity of heatwaves, we calculated the Excess Heat Factor (EHF), a metric that accounts for both the long-term climatological anomaly and short-term acclimatization @nairn2015excess.
]

/* The EHF for a given day ($t$) is calculated as:

$ "EHF"_t = "EHI"_("sig") times max(1, "EHI"_("accl")) $

Where:
- *Significance Index ($"EHI"_("sig")$):* The difference between the 3-day rolling average of the daily mean temperature ($T_("mean")$) and the 95th percentile of $T_("mean")$ for the 1986–2005 reference period.
- *Acclimatization Index ($"EHI"_("accl")$):* The difference between the 3-day rolling average of $T_("mean")$ and the average $T_("mean")$ of the preceding 30 days.

*/
#highlight[
We classified daily heatwave severity into three tiers—*Low-Intensity*, *Severe*, and *Extreme*—based on the methodology of the Australian Bureau of Meteorology @nairn2015excess.
]

/* The severity threshold ($"EHF"_(85)$) was defined as the 85th percentile of all positive EHF days recorded at that location during the 1986–2005 baseline.
Days were classified as:

- *Low-Intensity:* $0 < "EHF" <= "EHF"_(85)$
- *Severe:* $"EHF"_(85) < "EHF" <= 3 times "EHF"_(85)$
- *Extreme:* $"EHF" > 3 times "EHF"_(85)$
*/

==== Vulnerable Groups

We focused on two demographic groups particularly susceptible to heat-related health impacts:
- *Older adults (\≥65 years):* Age-related decrements in thermoregulation (e.g., reduced sweating) occur significantly by age 65 @kenney2003invited. Additionally, the risk of underlying chronic conditions such as cardiovascular, renal, and respiratory diseases—secondary aggravators of heat stress—increases with advanced age @ebi2021hot.
- *Infants (\<1 year):* Infants are highly vulnerable due to a high surface area-to-mass ratio (up to 4-fold greater than adults) and a limited behavioral ability to avoid heat @bin2023optimal.

==== Population Data Integration

To construct a continuous annual time series of global population distribution from 1980 to #max_year_analysis, we combined three distinct datasets:
- *1980–1999:* We utilized the *Lancet Countdown 2023 dataset* @romanello20232023, derived from the ISIMIP Histsoc dataset and NASA GPWv4 land area data. This data was available at a $0.25 degree times 0.25 degree$ resolution and was upscaled to match the ERA5-Land grid resolution of $0.1 degree times 0.1 degree$. We redistributed population counts from the coarser grid to the finer grid while preserving total population and masking out ocean cells.
- *2000–2014:* We used global gridded demographic data from the *WorldPop project* @worldpop2018global available at a $1 "km" times 1 "km"$ resolution based on the "top-down unconstrained approach." Age and sex groups were aggregated and then regridded to the ERA5-Land grid by summing values within each 0.1° cell.
- *2015–#max_year_analysis:* We utilized the *updated WorldPop dataset* @bondarenko2025spatial and aggregated age groups to the ERA5-Land grid by summing values within each cell.

For infant counts we aggregated the age band 0–1 from the respective datasets.
For Older adults (≥65 years), we summed the age bands 65–70, 70–75, 75–80, and 80+. 

==== Attribution of Changes in Heatwave Exposure: Decomposition Methods

To isolate the driving factors behind historical changes in human exposure to heatwaves, we decomposed the total change in exposure into three distinct components: the climate effect, the population effect, and their synergistic interaction. 

Exposure is defined as the number of person-days, calculated at the grid-cell level as the product of the vulnerable population count and the number of heatwave days. 
By isolating these variables, we can attribute exactly how much of the increased societal burden is driven by meteorological hazards versus demographic shifts.

Let $P_t$ represent the population of a specific vulnerable group (e.g., infants or older adults) in a target year $t$, and $C_t$ represent the number of heatwave days in that same year. 
Let $overline(P)_"base"$ and $overline(C)_"base"$ represent the mean population and mean heatwave days during the reference baseline period (e.g., 1986–2005 or 2007–2016).
The total change in exposure ($Delta E$) relative to the baseline is the difference between the target exposure and the baseline exposure:

$ Delta E = (P_t times C_t) - (overline(P)_"base" times overline(C)_"base") $

To attribute this total change to its specific drivers, we expand the equation into three discrete terms:

$ Delta E = Delta E_"climate" + Delta E_"population" + Delta E_"interaction" $

Each term in the decomposition isolates a specific real-world dynamic:

- *The Climate Effect* ($Delta E_"climate"$): This represents the increase in exposure driven purely by rising temperatures. It is calculated by multiplying the historically static baseline population by the change in heatwave days. It asks: _How much extra heat would the historical population have experienced?_
  $ Delta E_"climate" = overline(P)_"base" times (C_t - overline(C)_"base") $

- *The Population Effect* ($Delta E_"population"$): This represents the increase in exposure driven purely by demographic growth and aging. It is calculated by multiplying the newly added population by the historical baseline heatwave days. It asks: _How much extra exposure would have occurred simply because more vulnerable people now live in historically warm climates?_
  $ Delta E_"population" = (P_t - overline(P)_"base") times overline(C)_"base" $

- *The Interaction Effect* ($Delta E_"interaction"$): This represents the compounding, synergistic penalty of an expanding demographic facing a rapidly warming world. It accounts for the overlap where the *newly added* vulnerable populations are exposed to the *newly added* heatwave days. 
  $ Delta E_"interaction" = (P_t - overline(P)_"base") times (C_t - overline(C)_"base") $

Because these formulas are executed pixel-by-pixel, the methodology naturally accounts for the spatial heterogeneity of both climate change and human population dynamics.
The final global and regional totals for each effect were obtained by calculating the sum of the respective terms across all valid land grid cells. 

==== Code and resources to reproduce the results

The results were generated using Python, a copy of the code is available in this public repository https://github.com/FedericoTartarini/paper-lancet-countdown-global. 
// todo check if the Lancet Countdown has a public repository where we can upload the code, and if not, we can create one for this paper.
Users who want to reproduce the results will first need to download the datasets listed below. 
Then they can use the code to reproduce the results, please refer to the README file in the public repository which contains detailed instructions on how to run the Python code.

=== Updates Introduced for 2026

In this 2026 update, we have introduced: 
- the assessment of heatwave severity using the Excess Heat Factor (EHF) metric, allowing us to differentiate between low-intensity, severe, and extreme heatwaves.
- improved demographic data integration by utilizing the latest WorldPop datasets.
- removed the people aged 75+ since this group is already included in the Older adults (≥65 years) group.
- given that the population data now extends to 2025, we did not need to project population estimates beyond 2020 as done in previous years.
- we have included the analysis of heatwave exposure trends under the 2007–2016 baseline, to align with the Paris Agreement.
- #highlight[Add also the attribution/conterfactual analysis]

We are also proposing to include Dr Mitchell Black as a co-author for this indicator.

=== Data
 
- Climate Data: ECMWF ERA5-Land reanalysis dataset.
- Demographic Data (1980–2000): Hybrid gridded demographic dataset from the Lancet Countdown 2023 (0.25° resolution) @romanello20232023.
- Demographic Data (2000–2015): WorldPop Age and Sex Structure Unconstrained Global Mosaic @worldpop2018global.
- Demographic Data (2015–#max_year_analysis): WorldPop Age and Sex Structure Unconstrained Global Mosaic @bondarenko2025spatial. 
- #highlight[Add also the data used for the counterfactual/attribution analysis]

=== Caveats & Limitations

==== Climate Data

The ERA5-Land reanalysis dataset provides high-resolution temperature data suitable for heatwave analysis.
However, reanalysis datasets may have biases compared to in-situ observations.
These biases can affect the accuracy of heatwave detection and characterization.
Additionally, the spatial resolution of ERA5-Land (0.1° × 0.1°) may not capture microclimatic variations in urban areas, where heatwaves can be more intense due to the urban heat island effect.

==== Heatwave Definition

The chosen heatwave definition (3 consecutive days with both minimum and maximum temperatures above the 95th percentile) may not capture all relevant heatwave events, and does not account for humidity or other environmental factors that influence heat stress.

==== Demographic Data

To ensure consistency over time, data from multiple sources were integrated to capture both spatial and temporal demographic trends. 
However, validation of this integrated dataset is limited. 
In regions with sparse demographic data or shifting political boundaries, inconsistencies may arise in the spatial distribution of populations. 
For example, the division of Sudan is reflected in the dataset as missing or incomplete information for infant populations, illustrating the challenges of maintaining demographic continuity in dynamically changing regions.
WorldPop’s "top-down unconstrained" approach was used for population mapping. 
This method estimates population distribution without restricting allocation to residential areas, unlike the "constrained" approach, which relies on satellite imagery to identify inhabited locations. 
While this method ensures continuous coverage across all land areas, it may overestimate populations in low-density regions and underestimate them in high-density areas.

=== Future form of the indicator

Results will be updated each year using the latest available climate and population data. 
The definition of conditions that constitute a “heatwave” may be altered to align with emerging standardization from organizations such as the World Meteorological Society. 
The estimation of heat stress risk may also be expanded beyond heatwave days to include thermophysiological indices that account for dry-bulb air temperature, humidity, solar radiation, and wind speed, providing a more comprehensive assessment of heat-related health risks.

=== Additional Analyses and Figures

@change-heatwave-days illustrates the change in the number of heatwave days in #max_year_analysis compared to the baseline period, highlighting intense events across all continents.

#figure(
  image("/figures/map_hw_change_2025.png"),
  caption: [Map showing the change in heatwave days in 2025 compared to the 1986–2005 baseline. Eckert IV projection is used to preserve area, with a color scale indicating the increase (red) or decrease (blue) in heatwave days.],
) <change-heatwave-days>

While the total number of heatwave days decreased from the 2024 historical high (#db.heatwave_days_tot.over_65.at("2024") and #db.heatwave_days_tot.under_1.at("2024") billion Person-days for Older adults and Infants) the year #max_year_analysis was the second highest on record for heatwave exposure for both vulnerable groups combined.
Older adults were exposed to a #db.heatwave_days_tot.over_65.at("2025") billion Person-days of heatwaves, while Infants experienced #db.heatwave_days_tot.under_1.at("2025") billion Person-days, as illustrated in @hw-exposure-total.

#figure(
  image("/figures/hw_exposure_global_trends_combined.pdf"),
  caption: [Total number of Person-days experienced per year by Older adults and Infants.],
) <hw-exposure-total>

When normalized by population size, Older adults experienced on average #db.heatwave_days_avg.over_65.at("2025") Average heatwave days per person in #max_year_analysis, while Infants experienced #db.heatwave_days_avg.under_1.at("2025") Average heatwave days per person, as shown in @avg-hw-per-person.

#figure(
  image("/figures/hw_exposure_avg_days_per_person.pdf"),
  caption: [Average number of heatwave days experienced per person per year by Older adults and Infants.],
) <avg-hw-per-person>


==== Heatwave Severity - Excess Heat Factor (EHF)

To determine the contribution of heatwave severity to the overall exposure of vulnerable populations, we analyzed the distribution of heatwave days by severity category (low-intensity, severe, extreme) using the EHF metric.

#highlight[Here we need to finalize the numbers and the text for this section, which will be based on the analysis of the EHF data. 

This analysis will help to contextualize the increasing exposure to heatwaves in terms of not just frequency but also severity, which is crucial for understanding the potential health impacts on vulnerable groups.]

==== Heatwave Exposure by Regions and Countries

In this section, we present the geographic distribution of heatwave exposure for vulnerable populations, aggregated by country, Human Development Index (HDI) group, and World Health Organization (WHO) region.
These figures illustrate the disparities in heatwave exposure across different regions and the different distribution of vulnerable populations, which leads to significant variations in Person-days experienced across countries and regions.

When analyzed by country, as shown in @hw-days-by-country China is the country with the highest number of Person-days experienced by both vulnerable groups. 
India, the United States of America, Russian Federation also rank among the top countries with the highest heatwave exposure for both age groups.

#figure(
  image("/figures/heatwave_days_by_country.pdf"),
  caption: [Total Person-days experienced per year by vulnerable populations, aggregated by country.],
) <hw-days-by-country>

When data are grouped by HDI, countries classified as ‘Low’ HDI experienced the lowest number of Person-days for both age groups, while those in the ‘High’ and ‘Very High’ HDI categories experienced significantly higher exposure, as shown in @hw-days-by-hdi.
This pattern is likely driven by the higher concentration of vulnerable populations in more developed countries.

#figure(
  image("/figures/heatwave_days_by_hdi_group.pdf"),
  caption: [Total number of Person-days experienced per year by vulnerable populations, aggregated by Human Development Index (HDI) group.],
) <hw-days-by-hdi>

Countries in the Western Pacific experienced the highest number of Person-days for both age groups, as shown in @hw-days-by-who.
However, the African region experienced the fastest growth in Person-days after 2000 for Infants, while Europe and the Americas have a significant share of Person-days for the Older adult population, due to a rapid growth in the Older adult population coupled with a high number of heatwave days in these regions.

#figure(
  image("/figures/heatwave_days_by_who_region.pdf"),
  caption: [Total number of Person-days experienced per year by vulnerable populations, aggregated by World Health Organization (WHO) region.],
) <hw-days-by-who>

#figure(
  image("/figures/heatwave_days_by_lc_group.pdf"),
  caption: [Total Person-days experienced per year by vulnerable populations, by Lancet group.],
) <hw-days-by-lc>

==== Changes in Global Heatwave Exposure

===== Current Population Vulnerability

To quantify the additional heatwave burden faced by today's demographic, we estimated the change in exposure by multiplying the 2025 population of vulnerable groups by the change in heatwave days relative to historical climatological baselines.

In #max_year_analysis, compared with the 1986–2005 baseline, the population-weighted mean heatwave days per person increased by #db.hw_change.avg.at("1986-2005").under_1.at("2025") for Infants and #db.hw_change.avg.at("1986-2005").over_65.at("2025") for Older adults.
Relative to the more recent 2007–2016 baseline, the corresponding changes were #db.hw_change.avg.at("2007-2016").under_1.at("2025") for Infants and #db.hw_change.avg.at("2007-2016").over_65.at("2025") for Older adults, as shown in @hw-change-weighted-mean-by-baseline.

#figure(
  image("/figures/hw_change_weighted_mean_by_baseline.pdf"),
  caption: [Change in the average number of heatwave days per person per year for vulnerable populations between 1986–2005 and 2006–2024, under observed conditions with constant heatwave incidence at 1986–2005 levels.],
) <hw-change-weighted-mean-by-baseline>

Because of this increased frequency of extreme heat, the vulnerable populations alive in 2025 experienced a massive surge in total person-days of exposure to heatwaves, as illustrated in @hw-change-total-exposure-by-baseline.
Compared to the 1986–2005 average, this translated to an additional of #db.hw_change.total.at("1986-2005").under_1.at("2025") billion Person-days for Infants and #db.hw_change.total.at("1986-2005").over_65.at("2025") billion Person-days for Older adults.
Relative to the 2007–2016 baseline, the 2025 population experienced an additional #db.hw_change.total.at("2007-2016").under_1.at("2025") billion Person-days for Infants and #db.hw_change.total.at("2007-2016").over_65.at("2025") billion Person-days for Older adults.

#figure(
  image("/figures/hw_change_total_exposure_by_baseline.pdf"),
  caption: [Change in the total number of Person-days experienced by vulnerable populations between 1986–2005 and 2006–2024, under observed conditions with constant heatwave incidence at 1986–2005 levels.],
) <hw-change-total-exposure-by-baseline>

===== Attribution of Total Historical Change in Heatwave Exposure

While the previous metric isolates the climate-driven hazard applied to the modern population, a full historical decomposition reveals that demographic shifts have profoundly compounded the total societal burden. 
When accounting for the baseline heatwave exposure of the rapidly growing and aging global population, the true total increase in exposure is significantly higher. 
@f-climate-vs-pop-contributions illustrates the decomposition of the total change in heatwave exposure for vulnerable populations into the climate effect, population effect, and their interaction.

#figure(
  image("/figures/f_climate_vs_pop_contributions.pdf"),
  caption: [Decomposition of the total change in heatwave exposure (in billion Person-days) for vulnerable populations between 1986–2005 and 2006–2024, into the climate effect, population effect, and their interaction.],
) <f-climate-vs-pop-contributions>

#let r_b_1986_2005 = db.climate_vs_pop_summary.at("1986-2005")
#let r_b_2007_2016 = db.climate_vs_pop_summary.at("2007-2016")

Compared to the 1986–2005 baseline, the total societal exposure in #db.climate_vs_pop_summary.target for older adults increased by #r_b_1986_2005.over_65.combined billion person-days. 
By decomposing this total historical change, we attribute the drivers as follows:
- *The Climate Effect:* #r_b_1986_2005.over_65.climate_pct% (#r_b_1986_2005.over_65.climate billion person-days) of the increase was driven strictly by rising temperatures.
- *The Population Effect:* #r_b_1986_2005.over_65.population_pct% (#r_b_1986_2005.over_65.population billion person-days) was driven by the growing aging demographic.
- *The Interaction Effect:* The remaining #r_b_1986_2005.over_65.interaction_pct% (#r_b_1986_2005.over_65.interaction billion person-days) resulted from the synergistic interaction population growth and climate change.

For infants, the total societal exposure increased by #r_b_1986_2005.under_1.combined billion person-days relative to the 1986-2005 baseline. 
This was overwhelmingly driven by the climate effect (#r_b_1986_2005.under_1.climate billion person-days), which was slightly offset by negative population and interaction effects (-0.1 billion person-days), reflecting slowing birth rates or shifting infant demographics in certain highly exposed regions.

A similar compounding trend is observed when evaluating the 2007–2016 baseline. 
For older adults, the total societal exposure increased by #r_b_2007_2016.over_65.combined billion person-days, driven by a combination of the climate effect (#r_b_2007_2016.over_65.climate billion), the demographic population effect (#r_b_2007_2016.over_65.population billion), and their compounding interaction (#r_b_2007_2016.over_65.interaction billion).

#bibliography("references.bib")