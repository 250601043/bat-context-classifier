import copy
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, TensorDataset

CLASSES = ["Biting", "Feeding", "Fighting", "Mating protest", "Sleeping"]
EPOCHS = 8
device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(0)
torch.manual_seed(0)

x = np.load("data/x.npy")
y = pd.read_csv("data/y.csv")


def loader(split):
    rows = (y.split == split).to_numpy()
    labels = torch.tensor(y.label[rows].to_numpy())
    return DataLoader(TensorDataset(torch.from_numpy(x[rows]), labels), batch_size=64, shuffle=split == "train")


def prepare(images, train):
    images = images.to(device).float()
    images = (images - images.mean((1, 2), keepdim=True)) / (images.std((1, 2), keepdim=True) + 1e-6)
    if train:
        for image in images:
            f, t = random.randint(0, 112), random.randint(0, 240)
            image[f:f + random.randint(0, 16)] = 0
            image[:, t:t + random.randint(0, 16)] = 0
    return images[:, None].repeat(1, 3, 1, 1)


def predict(data):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(prepare(images, False)).argmax(1).cpu().numpy() for images, _ in data])


train, val, test = loader("train"), loader("val"), loader("test")
counts = np.bincount(y.label[y.split == "train"])
weights = torch.tensor((counts.sum() / counts) ** 0.5, dtype=torch.float32, device=device)

model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(512, len(CLASSES))
model.to(device)
loss_function = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

best_f1 = 0
for epoch in range(EPOCHS):
    model.train()
    for images, labels in train:
        loss = loss_function(model(prepare(images, True)), labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    f1 = f1_score(y.label[y.split == "val"], predict(val), average="macro")
    print(f"epoch {epoch + 1}: validation macro-F1 {f1:.3f}")
    if f1 > best_f1:
        best_f1, best_weights = f1, copy.deepcopy(model.state_dict())

model.load_state_dict(best_weights)
torch.save(best_weights, "model.pt")
truth = y.label[y.split == "test"].to_numpy()
prediction = predict(test)
majority = np.full_like(truth, counts.argmax())
print(f"majority baseline: accuracy {accuracy_score(truth, majority):.3f}, macro-F1 {f1_score(truth, majority, average='macro'):.3f}")
print(f"random baseline: accuracy {1 / len(CLASSES):.3f}")
print(classification_report(truth, prediction, target_names=CLASSES, digits=3))
print(f"balanced accuracy: {balanced_accuracy_score(truth, prediction):.3f}")
ConfusionMatrixDisplay.from_predictions(truth, prediction, display_labels=CLASSES, normalize="true", values_format=".2f")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("confusion_matrix.png")
