"""
Fine-tune google/vit-base-patch16-224 for binary car detection.

Labels: 1 = car present, 0 = no car
Data source: FiftyOne open-images-v6 dataset (built by data.py)
"""

import random
from pathlib import Path

import fiftyone as fo
import matplotlib.pyplot as plt
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

EPOCHS = 10
BATCH_SIZE = 128
LR = 1e-4
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

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
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for pixel_values, labels in val_loader:
                pixel_values, labels = pixel_values.to(device), labels.to(device)
                logits = model(pixel_values=pixel_values).logits
                val_loss += loss_fn(logits, labels).item() * len(labels)
                val_correct += (logits.argmax(-1) == labels).sum().item()

        val_loss /= len(val_samples)
        val_acc = val_correct / len(val_samples)
        scheduler.step(val_loss)

        print(f"Epoch {epoch}/{EPOCHS}  "
              f"loss={train_loss/len(train_samples):.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

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


def eval_occlusion(occlusion_levels: list[int] = [20, 40, 60, 80]):
    """Evaluate the best checkpoint on each occluded-car dataset.

    Every image in occluded_cars_dataset_X/ is a car image (label=1).
    Top-1 accuracy = fraction predicted as 'car'.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")

    processor = ViTImageProcessor.from_pretrained(str(CHECKPOINT_DIR / "best"))
    model = ViTForImageClassification.from_pretrained(str(CHECKPOINT_DIR / "best"))
    model.to(device)
    model.eval()

    print(f"\n{'Occlusion %':<14} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-" * 44)

    results = {}
    for pct in occlusion_levels:
        folder = Path(f"./occluded_cars_dataset_{pct}")
        paths = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
        dataset = CarDataset([(p, 1) for p in paths], processor)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE)

        correct = 0
        with torch.no_grad():
            for pixel_values, labels in loader:
                preds = model(pixel_values=pixel_values.to(device)).logits.argmax(-1).cpu()
                correct += (preds == labels).sum().item()

        acc = correct / len(paths)
        results[pct] = acc
        print(f"{pct:<14} {correct:<10} {len(paths):<10} {acc:.3f}")

    return results


def plot_occlusion_curve(
    vit_results: dict[int, float],
    resnet_results: dict[int, float],
    save_path: Path = Path("occlusion_curve.png"),
):
    levels      = sorted(l for l in vit_results if l > 0)
    vit_accs    = [vit_results[k]    for k in levels]
    resnet_accs = [resnet_results[k] for k in levels]

    fig, ax = plt.subplots(figsize=(7, 4))
    vit_line,    = ax.plot(levels, vit_accs,    marker="o", label="ViT")
    resnet_line, = ax.plot(levels, resnet_accs, marker="s", label="ResNet")

    if 0 in vit_results:
        ax.axhline(vit_results[0],    color=vit_line.get_color(),
                   linestyle=":", label="ViT baseline")
    if 0 in resnet_results:
        ax.axhline(resnet_results[0], color=resnet_line.get_color(),
                   linestyle=":", label="ResNet baseline")

    ax.set_xlabel("Occlusion (%)")
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_title("Accuracy Degradation Under Occlusion")
    ax.set_xticks(levels)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    print(f"Saved → {save_path}")
    plt.show()


if __name__ == "__main__":
    # train()
    # confusion_matrix()
    # vit_results    = eval_occlusion()

    vit_results = {0: 0.98, 20: 0.973, 40: 0.973, 60: 0.979, 80: 0.955}
    resnet_results = {0: 0.94, 20: 0.967, 40: 0.940, 60: 0.929, 80: 0.955}
    plot_occlusion_curve(vit_results, resnet_results)