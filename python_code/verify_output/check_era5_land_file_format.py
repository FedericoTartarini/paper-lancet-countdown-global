"""Check file format for a given ERA5-Land monthly file.

Run:
    python python_code/verify_output/check_era5_land_file_format.py
"""

from __future__ import annotations

from pathlib import Path
from zipfile import BadZipFile, ZipFile


def detect_file_format(file_path: Path) -> str:
    """Return a short format label based on file signature bytes."""
    if not file_path.exists():
        return "missing"

    size = file_path.stat().st_size
    if size == 0:
        return "empty"

    with file_path.open("rb") as handle:
        header = handle.read(16)

    if header.startswith(b"PK"):
        return "zip"
    if header[:3] == b"CDF":
        return "netcdf-classic"
    if header[:8] == b"\x89HDF\r\n\x1a\n":
        return "netcdf4-hdf5"
    if header.startswith(b"<") or b"<!DOCTYPE" in header.upper():
        return "html"

    return "unknown"


def extract_zip_to_netcdf(zip_path: Path, output_path: Path) -> bool:
    """Extract the first .nc inside a ZIP and save it as output_path."""
    try:
        with ZipFile(zip_path, "r") as archive:
            members = [name for name in archive.namelist() if name.endswith(".nc")]
            if not members:
                return False
            with archive.open(members[0]) as src, output_path.open("wb") as dst:
                dst.write(src.read())
        return True
    except BadZipFile:
        return False


def main() -> None:
    target = Path(
        "/Users/ftar3919/Library/CloudStorage/OneDrive-TheUniversityofSydney(Staff)/"
        "data/lancet/countdown-global/era5-land/hourly/2t/2025/"
        "2t_era5-land_oper_sfc_20251101-20251130.nc"
    )
    fmt = detect_file_format(target)
    print(f"File: {target}")
    print(f"Size: {target.stat().st_size if target.exists() else 'n/a'}")
    print(f"Detected format: {fmt}")

    if fmt == "zip":
        extracted_path = target.with_suffix(".extracted.nc")
        ok = extract_zip_to_netcdf(target, extracted_path)
        print(f"Extracted: {extracted_path} ({'ok' if ok else 'failed'})")

    if fmt in {"html", "unknown"}:
        with target.open("rb") as handle:
            preview = handle.read(500).decode("utf-8", errors="replace")
        print("Preview (first 500 bytes):")
        print(preview)


if __name__ == "__main__":
    main()
