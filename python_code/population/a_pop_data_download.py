import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from icecream import ic

from my_config import DirsLocal, VarsWorldPop

# this is needed to avoid the error in the download in MaxOS
os.environ["no_proxy"] = "*"


def download_file(url, filepath):
    filepath = Path(filepath)
    tmp_filepath = DirsLocal.dir_population_tmp / filepath.name

    ic(f"Downloading {filepath.name}...")

    if not filepath.is_file() and not tmp_filepath.is_file():
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            block_size = 8192

            # --- MODIFICATION: Removed internal tqdm to stop jumping ---
            with open(tmp_filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    f.write(chunk)
            # -----------------------------------------------------------

            shutil.move(tmp_filepath, filepath)
            ic(f"Downloaded {filepath}")

        except Exception as e:
            ic(f"Error downloading {filepath}: {e}")
    else:
        # ic(f"File {filepath} already exists")
        pass


def create_urls_sex_age_years(
    years=None,
    sexes=None,
    ages=None,
) -> list[tuple[str, str]]:
    if ages is None:
        ages = VarsWorldPop.worldpop_ages
    if sexes is None:
        sexes = VarsWorldPop.worldpop_sex
    if years is None:
        years = VarsWorldPop.get_years_range()
    _urls = []

    for year in years:
        for sex in sexes:
            for age in ages:
                download_url = VarsWorldPop.get_url_download(
                    year=year, sex=sex, age=age
                )
                filepath = (
                    DirsLocal.dir_population / f"global_{sex}_{age}_{year}_1km.tif"
                )
                tmp_filepath = DirsLocal.dir_population_tmp / filepath.name
                hd_filepath = DirsLocal.dir_pop_raw / filepath.name
                if (
                    not filepath.is_file()
                    and not tmp_filepath.is_file()
                    and not Path(hd_filepath).is_file()
                ):
                    _urls.append((download_url, filepath))
    return _urls


if __name__ == "__main__":
    urls = create_urls_sex_age_years(
        years=range(2025, 2031), sexes=["t"], ages=VarsWorldPop.worldpop_ages
    )
    ic("Total files to download:", len(urls))

    with ThreadPoolExecutor() as executor:
        executor.map(lambda p: download_file(*p), urls)
