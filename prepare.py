import os
from concurrent.futures import ProcessPoolExecutor

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch

CONTEXTS = {2: 0, 3: 1, 4: 2, 9: 3, 12: 4}
SAMPLES = 2 * 250000


def spectrogram(path, start, end):
    audio, _ = sf.read(path, start=start, stop=min(start + SAMPLES, end), dtype="float32")
    audio = np.pad(audio, (0, SAMPLES - len(audio)))
    spec = np.abs(librosa.stft(audio, n_fft=1024, hop_length=256))[:256]
    spec = librosa.amplitude_to_db(spec, ref=np.max, top_db=80)
    return torch.nn.functional.adaptive_avg_pool2d(torch.tensor(spec)[None], (128, 256))[0].numpy().astype(np.float16)


if __name__ == "__main__":
    annotations = pd.read_csv("data/Annotations.csv")
    info = pd.read_csv("data/FileInfo.csv", usecols=range(7))
    data = annotations[annotations.Context.isin(CONTEXTS)].merge(info, on="FileID")
    data["path"] = "data/audio/" + data["File name"]
    data = data[data.path.map(os.path.exists)].reset_index(drop=True)
    data["label"] = data.Context.map(CONTEXTS)
    data["day"] = data["Recording time"].str[:10]
    days = sorted(data.day.unique())
    data["split"] = "train"
    data.loc[data.day >= days[int(0.8 * len(days))], "split"] = "val"
    data.loc[data.day >= days[int(0.9 * len(days))], "split"] = "test"
    start = np.maximum(data["Start sample"], data["Voice start sample (1)"]).astype(int)

    with ProcessPoolExecutor() as pool:
        x = np.stack(list(pool.map(spectrogram, data.path, start, data["End sample"], chunksize=64)))
    np.save("data/x.npy", x)
    data[["label", "split"]].to_csv("data/y.csv", index=False)
    print(pd.crosstab(data.label, data.split))
