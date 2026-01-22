#let max_year_analysis = 2025

= Section 1: Health Hazards, Exposure, and Impact

== 1.1 Health and heat 

Indicator 1.1.1: exposure of vulnerable populations to heatwaves 

=== Indicator Authors
 
Dr Federico Tartarini, Prof Ollie Jay

=== Methods

==== Heatwave Occurrence and Duration

// todo check if is a period of 2 or 3 days
We defined a heatwave as a period of three or more consecutive days in which both the daily minimum and maximum temperatures exceeded the 95th percentile of the local climatology (REF). 
The climatological baseline was defined as the 1986–2005 reference period.
This dual-threshold definition captures both the direct heat stress caused by high daytime temperatures and the physiological strain associated with insufficient nighttime cooling (REF).

To determine these events, we utilized daily 2-meter temperature data from the European Centre for Medium-Range Weather Forecasts (ECMWF) ERA5 reanalysis dataset (REF), gridded at a 0.25° × 0.25° global resolution. 
For each grid cell and each year from 1980 to 2025, we calculated two primary metrics:

- Heatwave Duration: The total number of days per year spent during a heatwave.
- Heatwave Frequency: The total number of discrete heatwave events per year.

==== Heatwave Severity (Excess Heat Factor)

To assess the changing intensity of heatwaves, we calculated the Excess Heat Factor (EHF), a metric that accounts for both the long-term climatological anomaly and short-term acclimatization (Nairn & Fawcett, 2015).
The EHF for a given day ($t$) is calculated as:

$ "EHF"_t = "EHI"_("sig") times max(1, "EHI"_("accl")) $

Where:
- *Significance Index ($"EHI"_("sig")$):* The difference between the 3-day rolling average of the daily mean temperature ($T_("mean")$) and the 95th percentile of $T_("mean")$ for the 1986–2005 reference period.
- *Acclimatization Index ($"EHI"_("accl")$):* The difference between the 3-day rolling average of $T_("mean")$ and the average $T_("mean")$ of the preceding 30 days.

We classified daily heatwave severity into three tiers—*Low-Intensity*, *Severe*, and *Extreme*—based on the methodology of the Australian Bureau of Meteorology (REF). 
The severity threshold ($"EHF"_(85)$) was defined as the 85th percentile of all positive EHF days recorded at that location during the 1986–2005 baseline. 
Days were classified as:

- *Low-Intensity:* $0 < "EHF" <= "EHF"_(85)$
- *Severe:* $"EHF"_(85) < "EHF" <= 3 times "EHF"_(85)$
- *Extreme:* $"EHF" > 3 times "EHF"_(85)$

==== Vulnerable Groups

We focused on three demographic groups particularly susceptible to heat-related health impacts:
- *Elderly ($>= 65$ years):* Age-related decrements in thermoregulation (e.g., reduced sweating) occur significantly by age 65 (REF). Additionally, the risk of underlying chronic conditions such as cardiovascular, renal, and respiratory diseases—secondary aggravators of heat stress—increases with advanced age (REF).
- *Infants ($<1$ year):* Infants are highly vulnerable due to a high surface area-to-mass ratio (up to 4-fold greater than adults) and a limited behavioral ability to avoid heat (REF).
- *Pregnant Women:* [New Addition] Pregnancy places significant physiological strain on the cardiovascular and thermoregulatory systems. Extreme heat exposure during pregnancy has been linked to adverse outcomes including preterm birth, low birth weight, and stillbirth (REF).

==== Demographic Data Sources

To construct a continuous annual time series of global population distribution from 1980 to #max_year_analysis, we combined three distinct datasets:
- *1980–1999:* We utilized the *Lancet Countdown 2023 dataset* (REF), derived from the ISIMIP Histsoc dataset. This data was resampled to a $0.25 degree times 0.25 degree$ resolution using 2D linear interpolation incorporating population densities and NASA GPWv4 land area data.
- *2000–2014:* We used global gridded demographic data from the *WorldPop project* (REF), available at a $1 "km" times 1 "km"$ resolution based on the "top-down unconstrained approach." Aggregated age/sex groups were downscaled to match the ERA5 grid by summing values within each cell.
- *2015–#max_year_analysis:* We utilized the *updated WorldPop dataset* (REF), providing high-resolution annual estimates that account for recent migration and urbanization trends.

For infants counts were derived by aggregating the age bands 0–1 from the respective datasets.
For the elderly ($>= 65$ years), we summed the age bands 65–70, 70–75, 75–80, and 80+. 

// todo explain the methodology to estimate the number of pregnant women

==== Code and resources to reproduce the results

The results were generated using Python, a copy of the code is available in this public repository https://github.com/FedericoTartarini/paper-lancet-countdown-global. Users who want to reproduce the results will first need to download the datasets listed below. Then they can use the code to reproduce the results, please refer to the README file in the public repository which contains detailed instructions on how to run the Python code. 

Data 
- Climate Data: ECMWF ERA5 reanalysis dataset.
- Demographic Data (1980–2000): Hybrid gridded demographic dataset from the Lancet Countdown 2023 (0.25° resolution).
- Demographic Data (2000–2015): WorldPop Age and Sex Structure Unconstrained Global Mosaic.
- Demographic Data (2015–#max_year_analysis): WorldPop Age and Sex Structure Unconstrained Global Mosaic (updated version). // todo check and better describe this dataset

To ensure consistency over time, data from multiple sources were integrated to capture both spatial and temporal demographic trends. 
However, validation of this integrated dataset is limited. 
In regions with sparse demographic data or shifting political boundaries, inconsistencies may arise in the spatial distribution of populations. 
For example, the division of Sudan is reflected in the dataset as missing or incomplete information for infant populations, illustrating the challenges of maintaining demographic continuity in dynamically changing regions.
WorldPop’s "top-down unconstrained" approach was used for population mapping. 
This method estimates population distribution without restricting allocation to residential areas, unlike the "constrained" approach, which relies on satellite imagery to identify inhabited locations. 
While this method ensures continuous coverage across all land areas, it may overestimate populations in low-density regions and underestimate them in high-density areas. 

==== Future form of the indicator

Results will be updated each year using the latest available climate and population data. 
The definition of conditions that constitute a “heatwave” may be altered to align with emerging standardization from organizations such as the World Meteorological Society. 
The estimation of heat stress risk may also be expanded beyond heatwave days to include thermophysiological indices that account for dry-bulb air temperature, humidity, solar radiation, and wind speed, providing a more comprehensive assessment of heat-related health risks.

=== Additional analysis

@change-heatwave-days illustrates the change in the number of heatwave days in #max_year_analysis compared to the baseline period, highlighting intense events across all continents, particularly in regions such as Africa, Asia. 

#figure(
  image("/figures/map_hw_change_2025.png"),
  caption: [Map showing the change in heatwave days in 2025 compared to the 1986–2005 baseline.],
) <change-heatwave-days>

While the total number of heatwave days decreased from last year, #max_year_analysis still ranks as the second highest on record.
// todo update numbers below
Older adults (65+ y) endured a record 17.7 billion person-days of heatwaves (49% increase), people aged 75 year experienced 6.4 billion person-days, while infants under one year experienced 2.9 billion person-days, as illustrated in Figure 2.

#figure(
  image("/figures/heatwaves_exposure_total.pdf"),
  caption: [Total number of heatwaves days experienced per year by older adults (over 65 and over 75)  and infants.],
) <hw-exposure-total>

#figure(
  image("/figures/global_hw_per_person.pdf"),
  caption: [],
) <avg-hw-per-person>

#figure(
  image("/figures/heatwave_exposure_severe_extreme_trend.pdf"),
  caption: [],
) <hw-exposure-severe-extreme-trend>

#figure(
  image("/figures/heatwave_exposure_severity_stacked.pdf"),
  caption: [],
) <hw-exposure-severe-extreme-stacked>

#figure(
  image("/figures/zonal_fingerprint_heat_vs_pop.pdf"),
  caption: [],
) <zonal-fingerprint-heat-vs-pop>

#figure(
  image("/figures/heatwave_exposure_global_trends_combined.pdf"),
  caption: [],
) <hw-exposure-global-trends>

#figure(
  image("/figures/heatwave_severity_ratio_combined.pdf"),
  caption: [],
) <hw-severity-ratio>

#figure(
  image("/figures/barplots_dominant_effect_change.pdf"),
  caption: [],
) <dominant-effect-change>

#figure(
  image("/figures/exposure_change_heatwave_days_population_weighted_mean.pdf"),
  caption: [],
) <exposure-change-heatwave-days>

#figure(
  image("/figures/heatwave_days_by_hdi.pdf"),
  caption: [],
) <hw-days-by-hdi>

#figure(
  image("/figures/heatwave_days_by_who.pdf"),
  caption: [],
) <hw-days-by-who>

#figure(
  image("/figures/hw_exposure_age_0_countries_1980-2025.pdf"),
  caption: [],
) <hw-exposure-age-0-countries>

#figure(
  image("/figures/hw_exposure_age_65_countries_1980-2025.pdf"),
  caption: [],
) <hw-exposure-age-65-countries>


 
/* 
Figure 3 shows that on average across the world heatwave exposure is the highest among individuals over 75 (21.1 heatwave days per person), followed by those aged 65+ y (20.8 heatwave days per person). Infants experienced on average 20.5 heatwaves days per person. 
 
Figure 3. Average number of heatwave days experienced by individuals over 65, over 75, and infants under one year old.
When analyzed by country, as shown in Figure 3 and Figure 4, China and India are the countries with the highest number of affected individuals in both age categories, primarily due to their large populations. In 2024, a significant number of people over 65 were also impacted in Japan, the United States of America, and Italy, while heatwave exposure among infants was particularly high in Indonesia, Nigeria, and the Democratic Republic of the Congo.
 
Figure 4. Total heatwave person-days experienced by infants under one year old, presented by year and by the most affected countries.
 
Figure 5. Total heatwave person-days experienced by individuals over 65, presented by year and by the most affected countries.
Before 2024, countries classified as ‘Low’ HDI, on average, exhibited lower heatwave exposure for both age groups, as shown in Figure 6. However, these countries experienced the fastest growth in 2024 rising from 7.5 to 21.0 days—a 181% increase. 
 
Figure 6. Average number of heatwave days experienced aggregated by HDI level.
Figure 7 presents data aggregated by WHO regions. The Wester Pacific region was the most affected for the infants (under 1) and the over-65 population.
 
Figure 7. Average number of heatwave days experienced aggregated by WHO region.
Additional analysis
While climate change drives the increase in heatwave days, population growth also contributes to the rising number of heatwave person-days. This section compares the periods 1986–2005 and 2006–2024 to estimate how many heatwave days vulnerable populations would have experienced if climate change had not occurred, considering only demographic shifts.  
For each geographic coordinate, the average annual heatwave days affecting both elderly and infant populations were calculated for 2006–2024. The same calculation was repeated while holding heatwave incidence constant to the 1986–2005 levels, isolating the impact of climate change. Comparing these scenarios reveals how many heatwave days vulnerable populations would have been exposed to purely due to demographic changes.
Under a constant heatwave incidence at baseline levels, vulnerable populations would have experienced an average of 5.4 heatwave days per person per year in 2006–2024—50% fewer than observed. Infants faced an average increase of 4.6 heatwave days per year, while individuals over 65, a rapidly growing group, experienced an additional 5.3 heatwave days annually. For infants a slight decrease in per-person heatwave exposure (from 4.8 to 4.6) would have been observed if heatwave incidence remained at 1986–2005 levels, reflecting shifts in the geographic distribution of vulnerable populations. No change would have been observed for adults ages 65 years or over.
