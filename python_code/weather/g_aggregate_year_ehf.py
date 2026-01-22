import xarray as xr
import numpy as np
from joblib import Parallel, delayed
from my_config import Vars, Dirs


def aggregate_year(year, input_dir, output_dir):
    """
    Reads daily severity categories (0,1,2,3) and sums them up into annual totals.
    """
    input_file = input_dir / f"ehf_severity_{year}.nc"
    output_file = output_dir / f"severity_summary_{year}.nc"

    if output_file.exists():
        return f"Skipped {year} (Exists)"

    if not input_file.exists():
        return f"Missing daily input for {year}"

    # Load Daily Data
    # 0=None, 1=Low, 2=Severe, 3=Extreme
    ds = xr.open_dataset(input_file)
    sev = ds["severity"]

    # Calculate Annual Counts
    # We count days where condition matches
    days_low = (sev == 1).sum(dim="time", dtype=np.int16)
    days_severe = (sev == 2).sum(dim="time", dtype=np.int16)
    days_extreme = (sev == 3).sum(dim="time", dtype=np.int16)

    # Wrap in Dataset
    ds_out = xr.Dataset(
        {
            "days_low": days_low,
            "days_severe": days_severe,
            "days_extreme": days_extreme,
            # Helper: "Severe or worse" (2 + 3)
            "days_severe_plus": days_severe + days_extreme,
        }
    )

    ds_out = ds_out.expand_dims(dim={"year": [year]})

    # Save
    encoding = {v: {"zlib": True, "complevel": 5} for v in ds_out.data_vars}
    ds_out.to_netcdf(output_file, encoding=encoding)

    return f"Aggregated {year}"


def main():
    input_dir = Dirs.dir_results / "ehf_severity"
    output_dir = Dirs.dir_results / "ehf_severity_annual"
    output_dir.mkdir(parents=True, exist_ok=True)

    years = Vars.get_analysis_years()

    results = Parallel(n_jobs=6, verbose=10)(
        delayed(aggregate_year)(year, input_dir, output_dir) for year in years
    )
    print("\n".join(results))


if __name__ == "__main__":
    main()
