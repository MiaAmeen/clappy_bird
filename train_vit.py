"""
Fine-tune google/vit-base-patch16-224 for binary car detection.

Labels: 1 = car present, 0 = no car
Data source: FiftyOne open-images-v6 dataset (built by data.py)
"""

import random
from pathlib import Path

import fiftyone as fo
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import ViTForImageClassification, ViTImageProcessor

from data import get_samples

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "google/vit-base-patch16-224"
DATASET_NAME = "open-images-v6"
CHECKPOINT_DIR = Path("./checkpoints")

EPOCHS = 5
BATCH_SIZE = 16
LR = 2e-5
VAL_SPLIT = 0.15
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class CarDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], processor: ViTImageProcessor):
        self.samples = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # (3, 224, 224)
        return pixel_values, torch.tensor(label, dtype=torch.long)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    val_samples, train_samples = get_samples(is_test=True), get_samples(is_test=False)

    car_count = sum(l for _, l in train_samples)
    car_count_val = sum(l for _, l in val_samples)
    print(f"Train: {len(train_samples)} ({car_count} car, {len(train_samples)-car_count} no-car)")
    print(f"Val  : {len(val_samples)} ({car_count_val} car, {len(val_samples)-car_count_val} no-car)")

    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    train_loader = DataLoader(CarDataset(train_samples, processor), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(CarDataset(val_samples,   processor), batch_size=BATCH_SIZE)

    # Model — replace head with binary classifier
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "no_car", 1: "car"},
        label2id={"no_car": 0, "car": 1},
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # --- Train ---
        model.train()
        train_loss, train_correct = 0.0, 0
        for pixel_values, labels in train_loader:
            pixel_values, labels = pixel_values.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(pixel_values=pixel_values).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
            train_correct += (logits.argmax(-1) == labels).sum().item()

        train_acc = train_correct / len(train_samples)

        # --- Val ---
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for pixel_values, labels in val_loader:
                pixel_values, labels = pixel_values.to(device), labels.to(device)
                logits = model(pixel_values=pixel_values).logits
                val_correct += (logits.argmax(-1) == labels).sum().item()

        val_acc = val_correct / len(val_samples)

        print(f"Epoch {epoch}/{EPOCHS}  "
              f"loss={train_loss/len(train_samples):.4f}  "
              f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(CHECKPOINT_DIR / "best")
            processor.save_pretrained(CHECKPOINT_DIR / "best")
            print(f"  Saved best checkpoint (val_acc={val_acc:.3f})")

    print(f"\nDone. Best val_acc={best_val_acc:.3f}  →  {CHECKPOINT_DIR/'best'}")


def confusion_matrix():
    """Run the best checkpoint on the test set and print a confusion matrix."""
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")

    processor = ViTImageProcessor.from_pretrained(str(CHECKPOINT_DIR / "best"))
    model = ViTForImageClassification.from_pretrained(str(CHECKPOINT_DIR / "best"))
    model.to(device)
    model.eval()

    test_samples = get_samples(is_test=True)
    loader = DataLoader(CarDataset(test_samples, processor), batch_size=BATCH_SIZE)

    # [row=actual][col=predicted]  0=no_car  1=car
    matrix = [[0, 0], [0, 0]]
    with torch.no_grad():
        for pixel_values, labels in loader:
            preds = model(pixel_values=pixel_values.to(device)).logits.argmax(-1).cpu()
            for actual, pred in zip(labels.tolist(), preds.tolist()):
                matrix[actual][pred] += 1

    tn, fp, fn, tp = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    total = tn + fp + fn + tp
    print(f"\nConfusion Matrix (n={total})")
    print(f"                Pred: no_car   Pred: car")
    print(f"Actual: no_car     {tn:>6}      {fp:>6}")
    print(f"Actual: car        {fn:>6}      {tp:>6}")
    print(f"\nAccuracy:  {(tn+tp)/total:.3f}")
    print(f"Precision: {tp/(tp+fp):.3f}" if (tp+fp) else "Precision: N/A")
    print(f"Recall:    {tp/(tp+fn):.3f}" if (tp+fn) else "Recall:    N/A")


if __name__ == "__main__":
    train()
    confusion_matrix()