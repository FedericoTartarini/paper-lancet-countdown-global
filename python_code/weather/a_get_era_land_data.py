from my_config import DirsLocal, ensure_directories
from python_code.copy_files_hpc import sync_with_rsync
from pathlib import Path
import concurrent.futures

ensure_directories([DirsLocal.e5l_h])

# Get existing years that already have daily summaries
existing_years = set()
for file in DirsLocal.e5l_d.glob("*_daily_summaries.nc"):
    year_str = file.stem.split("_")[0]
    try:
        year = int(year_str)
        existing_years.add(year)
    except ValueError:
        pass  # Skip files that don't match the pattern

# Years to sync: from 1979 to current year (2026), excluding those with daily summaries
current_year = 2026
years_to_sync = [
    year for year in range(1979, current_year) if year not in existing_years
]


def sync_year(year):
    print(f"Syncing ERA Land data for year: {year}")
    source = Path(DirsLocal.dir_era_land, str(year))
    destination = Path(DirsLocal.e5l_h)
    sync_with_rsync(
        source=f"ft8695@gadi-dm.nci.org.au:{str(source)}",
        dest=destination,
    )
    return f"Completed syncing {year}"


# Parallelize with threads (since rsync is I/O bound)
if years_to_sync:
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(sync_year, year) for year in years_to_sync]
        for future in concurrent.futures.as_completed(futures):
            print(future.result())
else:
    print("All years already have daily summaries. No syncing needed.")
