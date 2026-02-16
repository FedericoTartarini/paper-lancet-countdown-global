"""
Create region rasters aligned to the ERA5-Land grid.

This script rasterizes World Bank Admin0 polygons to the exact grid of the
population dataset, producing integer masks for:
- country (ISO3)
- WHO region
- HDI group
- Lancet grouping

Outputs are saved in DirsLocal.region_rasters.

Run locally:
    python python_code/calculations/c_regions_rasters.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import regionmask
import xarray as xr
from matplotlib import pyplot as plt

from my_config import DirsLocal, FilesLocal, ensure_directories
from python_code.calculations.a_heatwave_exposure_pop_abs import standardize_grid

SHEET_NAME = "ISO3 - Name - LC - WHO - HDI"

SUBREGION_BOUNDS = {
    "name": "mediterranean",
    "lat_min": 20,
    "lat_max": 60,
    "lon_min": 0.0,
    "lon_max": 30,
}


def setup_logging() -> None:
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def resolve_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    """Resolve a column name from candidates in a DataFrame."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Could not find {label} column in {list(df.columns)}")


def load_groupings() -> pd.DataFrame:
    """Load ISO3 and grouping info from the Excel file."""
    if not FilesLocal.country_names_groupings.exists():
        raise FileNotFoundError(
            f"Grouping file not found: {FilesLocal.country_names_groupings}"
        )
    df = pd.read_excel(FilesLocal.country_names_groupings, sheet_name=SHEET_NAME)
    df = df.rename(
        columns={
            "WHO Region": "WHO Region",
            "HDI Group 2025": "HDI Group 2025",
            "LC Grouping": "LC Grouping",
            "Country Name to use": "Country Name to use",
            "ISO3": "ISO3",
        }
    )
    return df


def build_id_map(values: pd.Series) -> dict[str, int]:
    """Build a stable id mapping from sorted unique values."""
    unique_vals = sorted(v for v in values.dropna().unique())
    return {value: idx + 1 for idx, value in enumerate(unique_vals)}


def load_polygons() -> gpd.GeoDataFrame:
    """Load World Bank Admin0 polygons and keep only valid ISO3 rows."""
    if not FilesLocal.world_bank_shapefile.exists():
        raise FileNotFoundError(
            f"Shapefile not found: {FilesLocal.world_bank_shapefile}"
        )

    gdf = gpd.read_file(FilesLocal.world_bank_shapefile)
    iso_col = resolve_column(gdf, ["ISO_A3", "ISO3", "ISO_3_CODE"], "ISO3")

    gdf = gdf[gdf[iso_col].notna()].copy()
    gdf = gdf.rename(columns={iso_col: "ISO3"})
    gdf = gdf[gdf["ISO3"].str.len() == 3]

    return gdf


def load_grid_template() -> xr.DataArray:
    """Load a template grid from population data."""
    if not FilesLocal.pop_inf.exists():
        raise FileNotFoundError(f"Population file not found: {FilesLocal.pop_inf}")

    ds = xr.open_dataset(FilesLocal.pop_inf)
    da = ds["pop"] if "pop" in ds.data_vars else next(iter(ds.data_vars.values()))
    da = standardize_grid(da)
    return da


def check_overlaps(mask_3d: xr.DataArray, name: str) -> None:
    """Log if overlapping regions are detected in the raster mask."""
    overlap = (mask_3d.sum("region") > 1).sum().item()
    if overlap > 0:
        logging.warning("%s raster has %s overlapping pixels.", name, overlap)


def rasterize_regions(
    gdf: gpd.GeoDataFrame,
    lon: np.ndarray,
    lat: np.ndarray,
    id_col: str,
    name: str,
) -> xr.DataArray:
    """Rasterize polygons to a 2D integer mask using regionmask."""
    if id_col not in gdf.columns:
        raise ValueError(f"Missing id column '{id_col}' in GeoDataFrame.")

    missing = int(gdf[id_col].isna().sum())
    if missing:
        logging.warning(
            "Dropping %s rows with null '%s' for %s raster.", missing, id_col, name
        )
        gdf = gdf[gdf[id_col].notna()].copy()
    if gdf.empty:
        raise ValueError(f"No valid rows remain for id column '{id_col}'.")

    # Dissolve multipart countries/regions so each id is unique for regionmask.
    gdf = gdf.dissolve(by=id_col, as_index=False)

    mask_3d = regionmask.mask_3D_geopandas(gdf, lon, lat, numbers=id_col)
    check_overlaps(mask_3d, name)

    mask_2d = regionmask.mask_geopandas(gdf, lon, lat, numbers=id_col).astype("float32")
    mask_2d.name = name
    return mask_2d


def save_mask(mask: xr.DataArray, path: Path, name: str) -> None:
    """Save a mask DataArray to NetCDF with a simple dataset wrapper."""
    ds = mask.to_dataset(name=name)
    path.unlink(missing_ok=True)
    ds.to_netcdf(path)


def validate_mask(mask: xr.DataArray, name: str) -> None:
    """Log basic sanity checks for a mask."""
    total_cells = int(mask.count().item())
    assigned_cells = int(mask.notnull().sum().item())
    if assigned_cells == 0:
        raise ValueError(f"Mask {name} has zero assigned cells.")

    logging.info(
        "%s raster assigned %s/%s cells (%.2f%%).",
        name,
        assigned_cells,
        total_cells,
        100 * assigned_cells / total_cells,
    )


def plot_subregion_mask(
    polygons: gpd.GeoDataFrame,
    mask: xr.DataArray,
    name: str,
) -> None:
    """Plot polygons and raster mask for a subregion to validate alignment."""
    bounds = SUBREGION_BOUNDS
    ensure_directories([DirsLocal.region_rasters])

    sub_polygons = polygons.cx[
        bounds["lon_min"] : bounds["lon_max"], bounds["lat_min"] : bounds["lat_max"]
    ]
    if sub_polygons.empty:
        logging.warning("No polygons found in subregion for %s.", name)
        return

    sub_mask = mask.sel(
        lat=slice(bounds["lat_min"], bounds["lat_max"]),
        lon=slice(bounds["lon_min"], bounds["lon_max"]),
    )
    if sub_mask.size == 0:
        logging.warning("No raster cells found in subregion for %s.", name)
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    sub_polygons.boundary.plot(ax=ax, color="black", linewidth=0.6)
    ax.set_title(f"{name} polygons ({bounds['name']})")
    ax.set_xlim(bounds["lon_min"], bounds["lon_max"])
    ax.set_ylim(bounds["lat_min"], bounds["lat_max"])

    sub_mask.plot(ax=ax, cmap="tab20", add_colorbar=False)
    ax.set_title(f"{name} raster ({bounds['name']})")

    fig.tight_layout()
    fig.savefig(DirsLocal.region_rasters / f"{name}_{bounds['name']}_check.png")
    plt.show()


def main() -> None:
    """Create all region rasters aligned to the ERA5-Land grid."""
    setup_logging()
    ensure_directories([DirsLocal.region_rasters])

    logging.info("Loading groupings and polygons...")
    groupings = load_groupings()
    polygons = load_polygons()

    logging.info("Loading grid template...")
    template = load_grid_template()
    lat = template["latitude"].values
    lon = template["longitude"].values

    logging.info("Merging polygons with groupings...")
    merged = polygons.merge(groupings, on="ISO3", how="inner")
    if merged.empty:
        raise ValueError("No polygons matched the groupings table by ISO3.")

    logging.info("Creating country raster...")
    country_ids = build_id_map(merged["ISO3"])
    merged["country_id"] = merged["ISO3"].map(country_ids)
    country_mask = rasterize_regions(merged, lon, lat, "country_id", "country_id")
    validate_mask(country_mask, "country")
    save_mask(country_mask, FilesLocal.raster_country, "country_id")
    plot_subregion_mask(polygons=merged, mask=country_mask, name="country")

    logging.info("Creating WHO raster...")
    who_ids = build_id_map(merged["WHO Region"])
    merged["who_id"] = merged["WHO Region"].map(who_ids)
    who_mask = rasterize_regions(
        gdf=merged, lon=lon, lat=lat, id_col="who_id", name="who_id"
    )
    validate_mask(who_mask, "who")
    save_mask(who_mask, FilesLocal.raster_who, "who_id")
    plot_subregion_mask(merged, who_mask, "who")

    logging.info("Creating HDI raster...")
    hdi_ids = build_id_map(merged["HDI Group 2025"])
    merged["hdi_id"] = merged["HDI Group 2025"].map(hdi_ids)
    hdi_mask = rasterize_regions(merged, lon, lat, "hdi_id", "hdi_id")
    validate_mask(hdi_mask, "hdi")
    save_mask(hdi_mask, FilesLocal.raster_hdi, "hdi_id")
    plot_subregion_mask(merged, hdi_mask, "hdi")

    logging.info("Creating Lancet raster...")
    lc_ids = build_id_map(merged["LC Grouping"])
    merged["lc_id"] = merged["LC Grouping"].map(lc_ids)
    lc_mask = rasterize_regions(merged, lon, lat, "lc_id", "lc_id")
    validate_mask(lc_mask, "lancet")
    save_mask(lc_mask, FilesLocal.raster_lancet, "lc_id")
    plot_subregion_mask(merged, lc_mask, "lancet")

    logging.info("Region rasters saved to %s", DirsLocal.region_rasters)


def plot():
    """Plot the created rasters for visual validation."""
    for name, path in [
        ("country", FilesLocal.raster_country),
        ("who", FilesLocal.raster_who),
        ("hdi", FilesLocal.raster_hdi),
        ("lancet", FilesLocal.raster_lancet),
    ]:
        if not path.exists():
            logging.warning("Raster file not found for plotting: %s", path)
            continue
        ds = xr.open_dataset(path)
        da = next(iter(ds.data_vars.values()))
        plt.figure(figsize=(10, 6))
        da.plot()
        plt.title(f"{name.capitalize()} Raster")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.show()


if __name__ == "__main__":
    # main()
    plot()
