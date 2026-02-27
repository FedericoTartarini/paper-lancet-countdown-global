#let max_year_analysis = 2025
#set par(justify: true)

= Section 1: Health Hazards, Exposure, and Impact

== 1.1.1.2 Exposure of vulnerable populations to heatwaves

=== Indicator Authors
 
Dr Federico Tartarini, Prof Ollie Jay, Dr Mitchell Black

=== Methods

==== Heatwave Definition, Occurrence, and Duration

Heatwaves effects on human health is a growing concern worldwide, particularly for vulnerable populations such as the elderly and infants.
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

For each grid cell, the annual heatwave exposure (in person-days) was computed as:
$ "Exposure" = "Heatwave Days" times "Population" $

Where:
- *Heatwave Days:* The total number of heatwave days in that grid cell for the year.
- *Population:* The number of individuals in the vulnerable group residing in that grid cell.

The total annual exposure for each vulnerable group was obtained by summing the exposure across all grid cells globally.
We also present the average number of heatwave days experienced per person by dividing the total heatwave person-days by the total population of the vulnerable group for that year.

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
- *Elderly ($>= 65$ years):* Age-related decrements in thermoregulation (e.g., reduced sweating) occur significantly by age 65 @kenney2003invited. Additionally, the risk of underlying chronic conditions such as cardiovascular, renal, and respiratory diseases—secondary aggravators of heat stress—increases with advanced age @ebi2021hot.
- *Infants ($<1$ year):* Infants are highly vulnerable due to a high surface area-to-mass ratio (up to 4-fold greater than adults) and a limited behavioral ability to avoid heat @bin2023optimal.

==== Population Data Integration

To construct a continuous annual time series of global population distribution from 1980 to #max_year_analysis, we combined three distinct datasets:
- *1980–1999:* We utilized the *Lancet Countdown 2023 dataset* @romanello20232023, derived from the ISIMIP Histsoc dataset and NASA GPWv4 land area data. This data was available at a $0.25 degree times 0.25 degree$ resolution and was upscaled to match the ERA5-Land grid resolution of $0.1 degree times 0.1 degree$. We redistributed population counts from the coarser grid to the finer grid while preserving total population and masking out ocean cells.
- *2000–2014:* We used global gridded demographic data from the *WorldPop project* @worldpop2018global available at a $1 "km" times 1 "km"$ resolution based on the "top-down unconstrained approach." Age and sex groups were aggregated and then regridded to the ERA5-Land grid by summing values within each 0.1° cell.
- *2015–#max_year_analysis:* We utilized the *updated WorldPop dataset* @bondarenko2025spatial and aggregated age groups to the ERA5-Land grid by summing values within each cell.

For infant counts we aggregated the age band 0–1 from the respective datasets.
For the elderly ($>= 65$ years), we summed the age bands 65–70, 70–75, 75–80, and 80+.  

==== Code and resources to reproduce the results

The results were generated using Python, a copy of the code is available in this public repository https://github.com/FedericoTartarini/paper-lancet-countdown-global. 
// todo check if the Lancet Countdown has a public repository where we can upload the code, and if not, we can create one for this paper.
Users who want to reproduce the results will first need to download the datasets listed below. 
Then they can use the code to reproduce the results, please refer to the README file in the public repository which contains detailed instructions on how to run the Python code.

=== Updates Introduced for 2026

In this 2026 update, we have introduced: 
- the assessment of heatwave severity using the Excess Heat Factor (EHF) metric, allowing us to differentiate between low-intensity, severe, and extreme heatwaves.
- improved demographic data integration by utilizing the latest WorldPop datasets.
- removed the people aged 75+ since this group is already included in the 65+ age group.
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
  caption: [Map showing the change in heatwave days in 2025 compared to the 1986–2005 baseline.],
) <change-heatwave-days>

// todo check and finalise numbers in the following paragraph
While the total number of heatwave days decreased from last year, older adults (65+ y) were exposed to record 10 billion person-days of heatwaves (the second highest year on record), while infants under one year experienced 2.9 billion person-days, as illustrated in @hw-exposure-total.

#figure(
  image("/figures/hw_exposure_global_trends_combined.pdf"),
  caption: [Total number of heatwaves days experienced per year by older adults (over 65)  and infants.],
) <hw-exposure-total>

// todo check and finalise numbers in the following paragraph
When normalized by population size, individuals over 65 years experienced on average 12.4 heatwave days per person in #max_year_analysis, while infants experienced 9.5 heatwave days per person, as shown in @avg-hw-per-person.

#figure(
  image("/figures/hw_exposure_avg_days_per_person.pdf"),
  caption: [Average number of heatwave days experienced per person per year by older adults (over 65) and infants.],
) <avg-hw-per-person>


==== Heatwave Severity - Excess Heat Factor (EHF)

To determine the contribution of heatwave severity to the overall exposure of vulnerable populations, we analyzed the distribution of heatwave days by severity category (low-intensity, severe, extreme) using the EHF metric.

#highlight[Here we need to finalize the numbers and the text for this section, which will be based on the analysis of the EHF data. 

This analysis will help to contextualize the increasing exposure to heatwaves in terms of not just frequency but also severity, which is crucial for understanding the potential health impacts on vulnerable groups.]

==== Heatwave Exposure by Regions and Countries

In this section, we present the geographic distribution of heatwave exposure for vulnerable populations, aggregated by country, Human Development Index (HDI) group, and World Health Organization (WHO) region.
These figures illustrate the disparities in heatwave exposure across different regions and the different distribution of vulnerable populations, which leads to significant variations in heatwave person-days experienced across countries and regions.

When analyzed by country, as shown in @hw-days-by-country China is the country with the highest number of heatwave person-days experienced by both vulnerable groups. 
India, the United States of America, Russian Federation also rank among the top countries with the highest heatwave exposure for both age groups.

#figure(
  image("/figures/heatwave_days_by_country.pdf"),
  caption: [Total heatwave person-days experienced per year by vulnerable populations, aggregated by country.],
) <hw-days-by-country>

When data are grouped by HDI, countries classified as ‘Low’ HDI experienced the lowest number of heatwave person-days for both age groups, while those in the ‘High’ and ‘Very High’ HDI categories experienced significantly higher exposure, as shown in @hw-days-by-hdi.
This pattern is likely driven by the higher concentration of vulnerable populations in more developed countries.

#figure(
  image("/figures/heatwave_days_by_hdi_group.pdf"),
  caption: [Total number of person-days experienced per year by vulnerable populations, aggregated by Human Development Index (HDI) group.],
) <hw-days-by-hdi>

Countries in the Western Pacific experienced the highest number of heatwave person-days for both age groups, as shown in @hw-days-by-who.
However, the African region experienced the fastest growth in heatwave person-days after 2000 for infants, while Europe and the Americas have a significant share of heatwave person-days for the elderly population, due to a rapid growth in the elderly population coupled with a high number of heatwave days in these regions.

#figure(
  image("/figures/heatwave_days_by_who_region.pdf"),
  caption: [Total number of person-days experienced per year by vulnerable populations, aggregated by World Health Organization (WHO) region.],
) <hw-days-by-who>

#figure(
  image("/figures/heatwave_days_by_lc_group.pdf"),
  caption: [Total heatwave person-days experienced per year by vulnerable populations, by Lancet group.],
) <hw-days-by-lc>

==== Drivers of Change in Heatwave Exposure

Two factors are driving the increase in heatwave exposure for vulnerable populations: climate change and population growth.
@population-trend-absolute shows the global trend in total population of vulnerable groups from 1980 to #max_year_analysis, highlighting the significant growth in the elderly population over this period.
// todo check and finalise numbers in the following paragraph
The number of individuals over 65 has more than doubled from approximately 290 million in 1980 to over 800 million in #max_year_analysis, while the infant population has only seen a slight increase from around 100 million to 130 million.

// todo remove the red line from the plot
#figure(
  image("/figures/global_population_trend_millions.pdf"),
  caption: [],
) <population-trend-absolute>

@dominant-effect-change compares the periods 1986–2005 and 2006–2024 to estimate how many heatwave days vulnerable populations would have experienced if climate change had not occurred, considering only demographic shifts.  
Climate change is the primary driver of increased heatwave exposure for infants, accounting for all of the observed increase.
For the elderly population, both climate change and population growth contribute significantly, with populaton growth being the dominant factor in recent years.

#figure(
  image("/figures/barplots_dominant_effect_change.pdf"),
  caption: [],
) <dominant-effect-change>

==== Baseline Comparisons



#figure(
  image("/figures/exposure_change_heatwave_days_population_weighted_mean.pdf"),
  caption: [],
) <exposure-change-heatwave-days>
 
Before 2024, countries classified as ‘Low’ HDI, on average, exhibited lower heatwave exposure for both age groups, as shown in Figure 6. However, these countries experienced the fastest growth in 2024 rising from 7.5 to 21.0 days—a 181% increase. 
 
Figure 6. Average number of heatwave days experienced aggregated by HDI level.
Figure 7 presents data aggregated by WHO regions. The Wester Pacific region was the most affected for the infants (under 1) and the over-65 population.
 
Figure 7. Average number of heatwave days experienced aggregated by WHO region.
Additional analysis
While climate change drives the increase in heatwave days, population growth also contributes to the rising number of heatwave person-days. This section compares the periods 1986–2005 and 2006–2024 to estimate how many heatwave days vulnerable populations would have experienced if climate change had not occurred, considering only demographic shifts.  
For each geographic coordinate, the average annual heatwave days affecting both elderly and infant populations were calculated for 2006–2024. The same calculation was repeated while holding heatwave incidence constant to the 1986–2005 levels, isolating the impact of climate change. Comparing these scenarios reveals how many heatwave days vulnerable populations would have been exposed to purely due to demographic changes.
Under a constant heatwave incidence at baseline levels, vulnerable populations would have experienced an average of 5.4 heatwave days per person per year in 2006–2024—50% fewer than observed. Infants faced an average increase of 4.6 heatwave days per year, while individuals over 65, a rapidly growing group, experienced an additional 5.3 heatwave days annually. For infants a slight decrease in per-person heatwave exposure (from 4.8 to 4.6) would have been observed if heatwave incidence remained at 1986–2005 levels, reflecting shifts in the geographic distribution of vulnerable populations. No change would have been observed for adults ages 65 years or over.

#bibliography("references.bib")