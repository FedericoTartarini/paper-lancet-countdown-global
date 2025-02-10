import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import shutil
from icecream import ic

from my_config import DATA_SRC

# this is needed to avoid the error in the download in MaxOS
os.environ["no_proxy"] = "*"

data_population_path = DATA_SRC / "population"
data_population_path.mkdir(parents=True, exist_ok=True)
tmp_path = data_population_path / "tmp"
tmp_path.mkdir(parents=True, exist_ok=True)
base_worldpop_url = "https://data.worldpop.org/GIS/AgeSex_structures/Global_2000_2020/"
years_range = np.arange(2012, 2013)

# headers = {
#     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
#     'X-Forwarded-For': '203.0.113.195',  # Spoofed IP address
#     'Referer': 'https://google.com',     # Spoofed referer
#     'Accept-Language': 'en-US,en;q=0.9', # Language preferences
#     'Accept-Encoding': 'gzip, deflate, br'
# }

def download_file(url, filepath):
    filepath = Path(filepath)
    tmp_filepath = tmp_path / filepath.name
    ic(f"Downloading {filepath}")
    if not filepath.is_file():
        response = requests.get(url, stream=True)
        # response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()
        with open(tmp_filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        shutil.move(tmp_filepath, filepath)
        ic(f"Downloaded {filepath}")
    else:
        ic(f"File {filepath} already exists")


def create_urls_sex_age_years() -> list[tuple[str, str]]:
    urls = []

    for year in years_range:
        for sex in ["f", "m"]:
            for age in [0, 65, 70, 75, 80]:
                download_url = f"{base_worldpop_url}{year}/0_Mosaicked/global_mosaic_1km/global_{sex}_{age}_{year}_1km.tif"
                filepath = DATA_SRC / f"population/global_{sex}_{age}_{year}_1km.tif"
                if not Path(filepath).is_file():
                    urls.append((download_url, filepath))
    return urls


def create_urls_aggregated_years() -> list[tuple[str, str]]:
    urls = []
    for year in years_range:
        download_url = (
            f"{base_worldpop_url}{year}/0_Mosaicked/ppp_{year}_1km_Aggregated.tif"
        )
        filepath = DATA_SRC / f"population/ppp_{year}_1km_Aggregated.tif"
        if not Path(filepath).is_file():
            urls.append((download_url, filepath))
    return urls


if __name__ == "__main__":
    urls = create_urls_sex_age_years()

    for url, filepath in urls:
        print(url, filepath)
        download_file(url, filepath)

    # with ThreadPoolExecutor() as executor:
    #     executor.map(lambda p: download_file(*p), urls)


    # urls = create_urls_aggregated_years()
    # with ThreadPoolExecutor() as executor:
    #     executor.map(lambda p: download_file(*p), urls)
