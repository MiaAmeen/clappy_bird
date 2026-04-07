"""
Fine-tune ResNet-18 for binary car/no-car classification
using Open Images v6 via FiftyOne Zoo.

Requires data.py to be in the same directory.
"""

import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import precision_score, recall_score, confusion_matrix

from data import get_samples

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
VAL_SPLIT   = 0.20
BATCH_SIZE  = 16
NUM_EPOCHS  = 10
LR          = 1e-4

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CarDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def build_dataloaders():
    val_samples, train_samples = get_samples(is_test=True), get_samples(is_test=False)

    train_dataset = CarDataset(train_samples, transform=train_transforms)
    val_dataset   = CarDataset(val_samples,   transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(freeze_backbone: bool = False) -> nn.Module:
    model = models.resnet18(weights="IMAGENET1K_V1")
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(device)


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    acc = correct / len(loader.dataset)
    return total_loss / len(loader), acc, all_preds, all_labels

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict(image_path: str, model: nn.Module, threshold: float = 0.5) -> dict:
    img = Image.open(image_path).convert("RGB")
    tensor = val_transforms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
    car_prob = probs[0][1].item()  # index 1 = car, per get_samples()
    return {"car": car_prob > threshold, "confidence": car_prob}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    train_loader, val_loader = build_dataloaders()

    model     = build_model(freeze_backbone=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_val_acc  = 0.0
    best_epoch_info = {}
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc              = train_epoch(model, train_loader, optimizer, criterion)
        val_loss,   val_acc, preds, labels = eval_epoch(model, val_loader, criterion)
        scheduler.step()

        print(
            f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "resnet18_car_detector_best.pth")
            print(f"  ✓ New best model saved (val_acc={val_acc:.3f})")
            best_epoch_info = {
                "epoch":  epoch + 1,
                "acc":    val_acc,
                "preds":  preds,
                "labels": labels,
            }

    # Best epoch summary
    preds  = best_epoch_info["preds"]
    labels = best_epoch_info["labels"]
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    print(f"\n========== Best Epoch Summary (Epoch {best_epoch_info['epoch']}) ==========")
    print(f"  Accuracy  : {best_epoch_info['acc']:.3f}")
    print(f"  Precision : {precision_score(labels, preds):.3f}")
    print(f"  Recall    : {recall_score(labels, preds):.3f}")
    print(f"  --- Cars    ---  Correct: {tp} | Incorrect: {fn}")
    print(f"  --- No Cars ---  Correct: {tn} | Incorrect: {fp}")
    print(f"=================================================================")
    # Save final model
    torch.save(model.state_dict(), "resnet18_car_detector_final.pth")
    print("Training complete. Final model saved.")


if __name__ == "__main__":
    main()