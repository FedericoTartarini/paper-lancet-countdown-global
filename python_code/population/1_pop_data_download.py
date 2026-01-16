import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from icecream import ic
from tqdm import tqdm

from my_config import Dirs, VarsWorldPop

# this is needed to avoid the error in the download in MaxOS
os.environ["no_proxy"] = "*"


def download_file(url, filepath):
    filepath = Path(filepath)
    tmp_filepath = Dirs.dir_population_tmp / filepath.name
    ic(f"Downloading {filepath}")
    if not filepath.is_file() and not tmp_filepath.is_file():
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        block_size = 8192
        start_time = time.time()
        downloaded_size = 0

        with (
            open(tmp_filepath, "wb") as f,
            tqdm(
                desc=filepath.name,
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=block_size):
                f.write(chunk)
                downloaded_size += len(chunk)
                pbar.update(len(chunk))

                # Calculate download speed
                elapsed_time = time.time() - start_time
                if elapsed_time > 0:
                    speed = downloaded_size / elapsed_time  # bytes per second
                    pbar.set_postfix(speed=f"{speed / 1024:.2f} KB/s")

        shutil.move(tmp_filepath, filepath)
        ic(f"Downloaded {filepath}")
    else:
        ic(f"File {filepath} already exists")


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
                filepath = Dirs.dir_population / f"global_{sex}_{age}_{year}_1km.tif"
                tmp_filepath = Dirs.dir_population_tmp / filepath.name
                hd_filepath = Dirs.dir_pop_raw / filepath.name
                if (
                    not filepath.is_file()
                    and not tmp_filepath.is_file()
                    and not Path(hd_filepath).is_file()
                ):
                    _urls.append((download_url, filepath))
    return _urls


if __name__ == "__main__":
    urls = create_urls_sex_age_years(
        years=range(2021, 2026), sexes=["t"], ages=VarsWorldPop.worldpop_ages
    )
    ic(len(urls))

    # download multiple files at the same time
    with ThreadPoolExecutor() as executor:
        executor.map(lambda p: download_file(*p), urls)

    # # download one file at a time
    # for url, filepath in urls:
    #     print(url, filepath)
    #     download_file(url, filepath)
