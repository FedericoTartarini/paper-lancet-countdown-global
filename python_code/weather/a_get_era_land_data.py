from my_config import Dirs, ensure_directories
from python_code.copy_files_hpc import sync_with_rsync
from pathlib import Path
import concurrent.futures

ensure_directories([Dirs.dir_era_land_hourly_local])


def sync_year(year):
    print(f"Syncing ERA Land data for year: {year}")
    source = Path(Dirs.dir_era_land, str(year))
    destination = Path(Dirs.dir_era_land_hourly_local)
    sync_with_rsync(
        source=f"ft8695@gadi-dm.nci.org.au:{str(source)}",
        dest=destination,
    )
    return f"Completed syncing {year}"


# Parallelize with threads (since rsync is I/O bound)
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(sync_year, year) for year in range(2010, 2020)]
    for future in concurrent.futures.as_completed(futures):
        print(future.result())
