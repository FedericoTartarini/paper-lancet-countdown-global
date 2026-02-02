from my_config import Dirs, ensure_directories
from python_code.copy_files_hpc import sync_with_rsync
from pathlib import Path

ensure_directories([Dirs.dir_era_land_hourly_local])

for year in range(1980, 1990):
    print(f"Syncing ERA Land data for year: {year}")
    source = Path(Dirs.dir_era_land, str(year))
    destination = Path(Dirs.dir_era_land_hourly_local)
    sync_with_rsync(
        source=f"ft8695@gadi-dm.nci.org.au:{str(source)}",
        dest=destination,
    )
