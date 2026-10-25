# Bat Call Context Classifier — Project Guide for Claude Code

## Who you are working with

- A university student team (3 people) doing a **Deep Learning class group project**.
- The person driving this repo is a **2nd-year computer engineering student**.
  - **Programming:** knows the basics of coding (variables, loops, functions, classes, files). Comfortable reading normal Python, but not an expert.
  - **ML / DL:** knows **almost nothing** yet. They are taking ML and DL classes this semester at the same time as this project. Assume no prior knowledge of neural networks, training, loss functions, PyTorch, NumPy tricks, or audio processing.
- **The deadline is close.** Prefer the simplest approach that works over the most impressive one.

### Target level: a solid MID-LEVEL student project — not a research project
- The goal is a project the student **fully understands** and can explain line by line in a presentation, not the highest possible accuracy.
- Use **standard, well-known tools and patterns** only: PyTorch, torchvision ResNet18, librosa, scikit-learn, pandas, matplotlib. No exotic libraries, custom architectures, or advanced tricks (no attention models, contrastive learning, hyperparameter search frameworks, mixed precision, distributed training, etc.).
- Keep the codebase **small**: a handful of short files, plain functions, a simple training loop written out by hand (no PyTorch Lightning or other frameworks that hide the loop).
- If a choice is between "slightly better results" and "easier to understand", **choose easier to understand**.
- If something must be advanced, isolate it in one clearly commented function and explain it.

### User decision (2026-10-25): minimal code
- The project was rewritten from scratch as **three short scripts** (`download.py`, `prepare.py`, `train.py`) plus `requirements.txt`.
- **No comments, no docstrings, no extra files or folders** in the code. Explanations belong in the chat, not in the code. This overrides "comments explaining why" below.
- Good results must come from good parameters and basic deep-learning methods (pretrained ResNet18, class weights, masking augmentation, fine-tuning), not from extra machinery.
- The previous, larger version is kept in `C:\Users\berka\Desktop\bat-context-classifier-old`.

### How to behave
- **Teach while building.** When you introduce a concept (spectrogram, tensor, train/test split, transfer learning, loss, optimizer, epoch, overfitting, class imbalance…), give a short plain-language explanation (1–3 sentences, an analogy is welcome) the first time it appears. Don't lecture — explain just enough to understand the current step.
- **Explain your code after writing it.** After creating or changing a file, summarize in a few bullet points what it does and why, in beginner-friendly language.
- **Check understanding at milestones.** At the end of each milestone, list 2–3 questions a teacher might ask about that part, with short answers, so the student can prepare for the presentation.
- The student must be able to **present and defend this project in class**. Everything you build should serve that.
- **Work in small, runnable steps.** After each step, make sure the code runs and tell the student what it produced (a plot, a number, a file).
- **Never fake results.** If accuracy is low, say so and explain why. Low but honest results are acceptable; leaked or invented results are not.
- **Ask before** downloading large files (> 1 GB), deleting data, or making big structural changes.
- Keep code readable for beginners: clear names, short functions, comments explaining *why*, not just *what*.

---

## Project goal

Build a deep learning model that listens to a recorded call of an **Egyptian fruit bat** and predicts the **behavioural context** of the call (e.g. fighting over food, over a sleeping spot, mating…).

Pitch: *"A first step toward a bat translator."*

### Scope (strict — deadline is close)

| Priority | Task | Status |
|---|---|---|
| **MUST** | Context classification from spectrograms (4–6 most common contexts) | core deliverable |
| **MUST** | Honest evaluation: accuracy, per-class F1, confusion matrix, comparison vs. random/majority baseline | core deliverable |
| SHOULD | A simple classic-ML baseline (MFCC features + logistic regression or random forest) to compare against the CNN | if time allows |
| COULD | Second output head: predict the **emitter** (which bat is calling) | stretch only |
| WON'T | Addressee prediction, full-day "transcript", Grad-CAM, building a CNN from scratch | mention as future work |

---

## The dataset

**Source:** Prat, Taub, Pratt & Yovel (2017), *"An annotated dataset of Egyptian fruit bat vocalizations across varying contexts and during vocal ontogeny"*, Scientific Data 4:170143. License: CC BY 4.0 (cite it in the report).

- Figshare collection: https://figshare.com/collections/An_annotated_dataset_of_Egyptian_fruit_bat_vocalizations_across_varying_contexts_and_during_vocal_ontogeny/3666502
- DOI: https://doi.org/10.6084/m9.figshare.c.3666502
- Paper: https://www.nature.com/articles/sdata2017143
- Smaller alternative (≈10k labelled bat clips, via Internet Archive): https://huggingface.co/datasets/cgeorgiaw/animal-sounds
- The BEANS benchmark also includes this bat data with ready splits: https://github.com/earthspecies/beans

### Known facts about the data
- ~293,238 WAV files, **250 kHz sampling rate**, 16-bit, a few seconds each. Compressed with **7-Zip**, split into archives of ≤ 10,000 files.
- The Figshare collection shows ~65 *items*: these are archives + metadata files, not 65 recordings.
- `Annotations.csv`: ~91,080 annotated interactions. Columns include **emitter** (bat ID), **addressee** (bat ID), **context**, and the start/end position of the annotated segment inside the file.
- `FileInfo.csv`: recording time, recording channel (microphone), and start/end of voiced segments for each file (in samples).
- `Metadata.pdf`: definitions of every code (context IDs, behaviours, bat IDs).
- **Negative bat IDs** mean the pair was known but who emitted/addressed was uncertain → **drop these rows**.
  - **Team decision (2026-10-06): for the context task we KEEP negative-ID rows.** The context label is still valid; only the caller is uncertain. Dropping them halves the data and removes the Fighting class (fights → caller often unclear). Drop them **only** for the emitter stretch goal. Explain this in the report.
- Context `0 = Unknown` → **drop**.
- Annotation accuracy reported by the authors: ~97.9 % for emitter/addressee, ~96 % for context.

### First thing to do with the data
**Do not assume column names or context codes.** Load `Annotations.csv` and `FileInfo.csv`, print `.head()`, `.columns`, value counts of the context column, and cross-check codes against `Metadata.pdf` before writing any pipeline code.

### Downloading
Use the Figshare public API (no login needed):
```python
import requests
API = "https://api.figshare.com/v2"
items = requests.get(f"{API}/collections/3666502/articles", params={"page_size": 100}).json()
for item in items:
    for f in requests.get(f"{API}/articles/{item['id']}/files").json():
        print(f["name"], round(f["size"]/1e9, 2), "GB", f["download_url"])
```
- Download the small files first (CSV, PDF).
- Then only **1–2 audio archives** (enough for a 5–10k-clip working subset). Ask the student before downloading more.
- Extract with `7z x <archive> -o data/raw/audio` (install `p7zip-full` on Linux/Colab, 7-Zip on Windows).
  - **Checked (2026-10-06):** the Figshare archives are plain `.zip` files (`files101.zip` … `files224.zip`) holding WAVs directly, so no 7-Zip is needed. `FileInfo.csv` column `File folder` says which archive each file is in. `python src/download.py files219.zip files220.zip` downloads and extracts **only annotated** recordings into `data/raw/audio/`.
  - **Since 2026-10-25 all 30 archives are used.** `download.py` streams each needed archive to `data/` with one resumable request (Figshare rate-limits many small requests, so don't fetch single files by range) and extracts only the needed WAVs to `data/audio/`.

---

## Technical plan

### Pipeline
```
WAV (250 kHz) ──► crop to annotated segment ──► fixed length (pad/trim)
             ──► spectrogram (image) ──► pretrained CNN (ResNet18) ──► context class
```

### Key decisions (keep unless there is a good reason)
1. **Audio → image.** Convert each call to a spectrogram and treat the task as image classification. This lets us reuse standard CNN code.
2. **Bat-specific spectrogram settings.** Bats are not humans:
   - Keep the native 250 kHz or downsample carefully; do **not** blindly resample to 16/22/44 kHz without checking which frequency band holds the energy (look at a few spectrograms first).
   - Try a **linear-frequency** spectrogram (log-amplitude). Mel scale is tuned to human hearing; compare mel vs. linear only if time allows.
   - Use `librosa` (or `torchaudio`). Save spectrograms to disk as `.npy` so training doesn't recompute them.
3. **Fixed input size.** Pad or crop every clip to the same duration (pick from the duration distribution, e.g. the ~90th percentile) and resize spectrograms to e.g. 128×128 or 224×224.
4. **Transfer learning.** Use `torchvision` ResNet18 pretrained on ImageNet. Replace the final layer with one output per context class. First train only the head, then optionally unfreeze the last block.
5. **Class imbalance.** Use class weights in `CrossEntropyLoss` (or a weighted sampler). Report per-class metrics, not just accuracy.
6. **Augmentation (audio-appropriate).** SpecAugment-style time masking and frequency masking. **No vertical flips or rotations** — they change pitch/meaning.

### Data splitting — critical
- **Never split randomly by clip.** Clips from the same recording session/day are correlated and would leak between train and test, inflating accuracy.
- Split **by recording date/time** (from `FileInfo.csv`), e.g. earliest ~70 % of days → train, next ~15 % → validation, last ~15 % → test.
- If a stretch goal is attempted, also test the context model on **bats it never heard in training** (to check it learns context, not voices).
- Fix random seeds and save the split to a file (`data/splits.csv`) so results are reproducible.

### Evaluation (must appear in the report)
- Baselines: majority-class accuracy and random-guess accuracy.
- Model: accuracy, macro-F1, per-class precision/recall, confusion matrix (saved as PNG).
- A short honest discussion of errors and limitations.

---

## Environment

- **Python 3.10+**, **PyTorch** + **torchvision**, **librosa**, **numpy**, **pandas**, **scikit-learn**, **matplotlib**, **tqdm**, **requests**.
- Pin versions in `requirements.txt`.
- GPU: if the student's machine has no CUDA GPU, keep the model small (ResNet18, 128×128 inputs, a few thousand clips) so CPU training is feasible, **or** prepare a Google Colab notebook for the training step. Detect the device automatically (`cuda` → `mps` → `cpu`).
- Large data stays out of git: add `data/` and `outputs/checkpoints/` to `.gitignore`.

---

## Repository structure (current, since 2026-10-25)

```
bat-context-classifier/
├── CLAUDE.md
├── requirements.txt
├── .gitignore
├── download.py             ← Figshare: CSVs + every archive with annotated calls, extracts needed WAVs
├── prepare.py              ← labels, 80/10/10 split by recording day, 2 s spectrograms → data/x.npy, data/y.csv
├── train.py                ← ResNet18 fine-tuning, best epoch on validation, test report, confusion_matrix.png
└── data/                   ← Annotations.csv, FileInfo.csv, audio/, archives, x.npy, y.csv (git-ignored)
```

---

## Milestones (do them in this order)

1. **Setup** — create structure, `requirements.txt`, `.gitignore`, verify imports.
2. **Download** — CSV/PDF files, then 1–2 audio archives; extract.
3. **Explore** — inspect labels, count contexts, plot durations, plot 3–4 spectrograms per context. Explain findings to the student.
4. **Clean labels** — drop unknown context and negative IDs; pick the 4–6 most frequent contexts with enough samples in the downloaded subset.
5. **Features** — generate and cache spectrograms.
6. **Split** — time-based split, saved to `data/splits.csv`; print class counts per split.
7. **First model** — ResNet18 head-only training; get a first test number, even if bad.
8. **Improve** — class weights, augmentation, unfreeze last block, tune input size/duration. Change one thing at a time and log results in `report/notes.md`.
9. **Evaluate** — baselines vs. model, confusion matrix, per-class metrics.
10. **(Stretch)** emitter head or MFCC baseline.
11. **Wrap-up** — README with how to run + results table; help draft the report/slide outline.

After each milestone, append a short entry to `report/notes.md`: what was done, key numbers, what was learned. This becomes the report.

---

## Current status (2026-10-25): minimal rewrite done

- **Data**: all 30 archives with annotated calls (files101–106, files201–224), June 2012 – Feb 2014. 50,998 recordings → 52,769 clips of the 5 classes. Archives are kept in `data/` (~64 GB, can be deleted after extraction).
- **Classes (5)**: Biting, Feeding, Fighting, Mating protest, Sleeping (codes 2, 3, 4, 9, 12). Negative IDs kept, context 0 dropped.
- **Clip**: 2 s from max(annotation start, first voiced sample of the file); STFT 1024/256, dB, 0–62.5 kHz, pooled to 128×256, float16 in `data/x.npy`.
- **Split**: by recording day, first 80% of days train, next 10% val (from 2013-04-12), last 10% test (2013-05-17 → 2014-02-19). Train 37,971 / val 8,229 / test 6,569.
- **Training**: full fine-tuning of ImageNet ResNet18, AdamW lr 1e-4 with cosine schedule, 8 epochs, batch 64, class weights = sqrt(inverse frequency), per-clip normalisation, one random time and frequency mask (≤16 px). Best epoch by validation macro-F1.
- **Result**: val macro-F1 0.484 (acc 0.765). Test macro-F1 0.432, acc 0.645 (majority 0.671 / 0.161). Test F1: Sleeping 0.76, Feeding 0.59, Mating 0.43, Fighting 0.36, Biting 0.02. The test period starts with the regrouped colonies (treatments 18–20), which is why it is harder than validation.
- Tried on validation and not adopted: constant learning rate, lr 3e-4, full class weights, 12 epochs, balanced sampling (5,000 clips per class per epoch with `WeightedRandomSampler`, no loss weights, 12 epochs: val 0.471, test 0.438 / acc 0.635 / balanced acc 0.491, Biting test F1 0.06). Balancing doesn't fix Biting: the imbalance is not the main bottleneck.
- The earlier 3-phase version (ensembles, class boosts, ~1,300 lines) is in `bat-context-classifier-old` with its own notes; its numbers used a different test set.

## Things to avoid

- Random clip-level train/test split (data leakage).
- Reporting only accuracy on imbalanced classes.
- Vertical flips/rotations as augmentation.
- Downloading the full dataset (hundreds of thousands of files) without asking.
- Building a CNN from scratch before a pretrained baseline works.
- Long, unexplained code dumps — the student needs to understand and present this.

## Citation for the report

Prat, Y., Taub, M., Pratt, E., & Yovel, Y. (2017). An annotated dataset of Egyptian fruit bat vocalizations across varying contexts and during vocal ontogeny. *Scientific Data*, 4, 170143. https://doi.org/10.1038/sdata.2017.143
