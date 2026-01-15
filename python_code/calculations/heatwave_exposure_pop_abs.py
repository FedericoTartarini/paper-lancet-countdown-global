"""
# Total heatwave exposures

From the health system perspective, we'd like to know not just exposures to change (which in the climate perspective
is useful to demonstrate that HWs are really forming a trend relative to the null hypothesis of being normally
distributed around 0) - but the absolute values with the idea to know a) how big is this change from 'normal' and b)
how it compares to what we already cope with. More generally the idea is that if you measure millions more exposure
days but on a total value of billions, then even if you pick out a statistically significant trend you might not (
from the policy POV) care that much. On the other hand if you are talking 2x historical it's an issue.

The ideal is to show 'percentage change' rel. to a baseline. the problem is the population data doesn't exist and
even if it does, it doesn't make sense to average over 20years like we do for climatology.

The first step is to just calculate absolute values - these aren't too problematic since anyway the 'HW delta' is
kinda double-normalising since we 1x used 20y period for climatology then again for the baseline of the delta. Just
plotting then the time series gives a pretty good idea of where you stand relative to 'normal'

The next idea is to copy how GDP is presented as a percentage year-to-year. Since it doesn't make sense with pop to
normalise to a baseline period, and it's very arbitrary to pick one year of period, instead plot the percentage
change from previous year (e.g. https://fred.stlouisfed.org/graph/?g=eUmi)"""

import pandas as pd
import xarray as xr

from my_config import Dirs
from python_code.shared_functions import (
    read_pop_data_processed,
    calculate_exposure_population,
)


def main():
    pop_inf, pop_eld, _, pop_75 = read_pop_data_processed(get_pop_75=True)

    heatwave_metrics_files = sorted(Dirs.dir_results_heatwaves_days.value.glob("*.nc"))
    heatwave_metrics = xr.open_mfdataset(heatwave_metrics_files, combine="by_coords")

    exposures_over65 = calculate_exposure_population(
        data=pop_eld, heatwave_metrics=heatwave_metrics
    )
    exposures_infants = calculate_exposure_population(
        data=pop_inf, heatwave_metrics=heatwave_metrics
    )
    exposures_75 = calculate_exposure_population(
        data=pop_75, heatwave_metrics=heatwave_metrics
    )

    exposures = xr.concat(
        [exposures_infants, exposures_over65, exposures_75],
        dim=pd.Index([0, 65, 75], name="age_band_lower_bound"),
    )

    exposures_over65.to_netcdf(Dirs.dir_file_elderly_exposure_abs)

    exposures_infants.to_netcdf(Dirs.dir_file_infants_exposure_abs)

    Dirs.dir_file_all_exposure_abs.value.unlink(missing_ok=True)
    exposures.to_netcdf(Dirs.dir_file_all_exposure_abs.value)


if __name__ == "__main__":
    main()
    pass
