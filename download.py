import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

API = "https://api.figshare.com/v2"
CONTEXTS = [2, 3, 4, 9, 12]


def download(url, size, path):
    while not os.path.exists(path) or os.path.getsize(path) < size:
        done = os.path.getsize(path) if os.path.exists(path) else 0
        try:
            with requests.get(url, headers={"Range": f"bytes={done}-"}, stream=True, timeout=60) as response:
                if response.status_code != 206:
                    time.sleep(30)
                    continue
                with open(path, "ab") as f:
                    for chunk in response.iter_content(1 << 20):
                        f.write(chunk)
        except requests.RequestException:
            time.sleep(30)


def extract(folder, names):
    missing = [name for name in names if not os.path.exists(f"data/audio/{name}")]
    if missing:
        download(*files[f"{folder}.zip"], f"data/{folder}.zip")
        with zipfile.ZipFile(f"data/{folder}.zip") as archive:
            for name in missing:
                archive.extract(name, "data/audio")
    print(folder, len(missing), flush=True)


files = {}
for article in requests.get(f"{API}/collections/3666502/articles", params={"page_size": 100}).json():
    for file in requests.get(f"{API}/articles/{article['id']}/files").json():
        files[file["name"]] = (file["download_url"], file["size"])

os.makedirs("data/audio", exist_ok=True)
for name in ["Annotations.csv", "FileInfo.csv"]:
    download(*files[name], f"data/{name}")

annotations = pd.read_csv("data/Annotations.csv")
info = pd.read_csv("data/FileInfo.csv", usecols=range(6))
needed = info[info.FileID.isin(annotations[annotations.Context.isin(CONTEXTS)].FileID)]
groups = [(folder, list(group["File name"])) for folder, group in needed.groupby("File folder")]
with ThreadPoolExecutor(10) as pool:
    list(pool.map(lambda group: extract(*group), groups))
