import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from icecream import ic
from tqdm import tqdm

from my_config import dir_population_tmp, dir_pop_raw, dir_population

# this is needed to avoid the error in the download in MaxOS
os.environ["no_proxy"] = "*"


base_worldpop_url = "https://data.worldpop.org/GIS/AgeSex_structures/Global_2000_2020/"
years_range = np.arange(2000, 2021)


def download_file(url, filepath):
    filepath = Path(filepath)
    tmp_filepath = dir_population_tmp / filepath.name
    ic(f"Downloading {filepath}")
    if not filepath.is_file() and not tmp_filepath.is_file():
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        block_size = 8192
        start_time = time.time()
        downloaded_size = 0

        with open(tmp_filepath, "wb") as f, tqdm(
            desc=filepath.name,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
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


def create_urls_sex_age_years() -> list[tuple[str, str]]:
    _urls = []

    for year in years_range:
        for sex in ["m", "f"]:
            for age in [0, 65, 70, 75, 80]:
                download_url = f"{base_worldpop_url}{year}/0_Mosaicked/global_mosaic_1km/global_{sex}_{age}_{year}_1km.tif"
                filepath = dir_population / f"global_{sex}_{age}_{year}_1km.tif"
                tmp_filepath = dir_population_tmp / filepath.name
                hd_filepath = dir_pop_raw / filepath.name
                if (
                    not filepath.is_file()
                    and not tmp_filepath.is_file()
                    and not Path(hd_filepath).is_file()
                ):
                    _urls.append((download_url, filepath))
    return _urls


if __name__ == "__main__":
    urls = create_urls_sex_age_years()
    ic(len(urls))

    # download multiple files at the same time
    with ThreadPoolExecutor() as executor:
        executor.map(lambda p: download_file(*p), urls)

    # # download one file at a time
    # for url, filepath in urls:
    #     print(url, filepath)
    #     download_file(url, filepath)
