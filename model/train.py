import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

TRAIN_DIR = "data/train"
VAL_DIR   = "data/val"
TEST_DIR  = "data/test"
MODEL_DIR = "model"

BATCH_SIZE    = 32
MAX_EPOCHS    = 50
PATIENCE      = 7
FREEZE_EPOCHS = 5  # freeze backbone for first 5 epochs, then unfreeze layer3+4
LR_HEAD       = 1e-3
LR_BACKBONE   = 1e-5  # much lower lr for backbone after unfreezing
LR_HEAD_2     = 3e-4
DROPOUT       = 0.3


class FrequencyMask:
    # masks a random horizontal band - simulates missing freq data
    def __init__(self, max_width=20):
        self.max_width = max_width

    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            return img
        _, h, w = img.shape
        mw = random.randint(1, min(self.max_width, h // 4))
        start = random.randint(0, h - mw)
        img[:, start:start + mw, :] = 0
        return img


class TimeMask:
    # masks a random vertical band - simulates dropout in time
    def __init__(self, max_width=20):
        self.max_width = max_width

    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            return img
        _, h, w = img.shape
        mw = random.randint(1, min(self.max_width, w // 4))
        start = random.randint(0, w - mw)
        img[:, :, start:start + mw] = 0
        return img


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    FrequencyMask(max_width=20),
    TimeMask(max_width=20),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class BioSonarNet(nn.Module):
    # resnet34 with dropout before the final layer
    # dropout helps bc some species dont have that many training samples
    def __init__(self, num_classes, dropout=DROPOUT):
        super().__init__()
        backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.dropout  = nn.Dropout(dropout)
        self.fc       = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.fc(x)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_class_weights(dataset):
    # beaked whale has way fewer samples so we weight it higher
    # so the model doesnt just ignore it
    targets = [s[1] for s in dataset.samples]
    counts  = np.bincount(targets)
    weights = 1.0 / (counts + 1e-6)
    return torch.tensor(weights / weights.sum() * len(counts), dtype=torch.float32)


def freeze_backbone(model):
    for param in model.features.parameters():
        param.requires_grad = False


def unfreeze_late_layers(model):
    # unfreeze layer3 (idx 6) and layer4 (idx 7)
    # keeping early layers frozen bc they just detect basic edges/textures
    # which are already useful from imagenet pretraining
    for i, child in enumerate(model.features.children()):
        if i >= 6:
            for param in child.parameters():
                param.requires_grad = True


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    correct    = 0
    total      = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

    return total_loss / total, correct / total


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    device = get_device()
    print(f"training on: {device}\n")

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=val_transform)
    test_ds  = datasets.ImageFolder(TEST_DIR,  transform=val_transform)

    print(f"classes: {train_ds.classes}\n")

    # pin_memory only works on cuda, not mps
    pin = device.type == "cuda"

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)

    class_weights = compute_class_weights(train_ds).to(device)
    model         = BioSonarNet(num_classes=len(train_ds.classes)).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    freeze_backbone(model)
    optimizer = optim.AdamW(
        list(model.dropout.parameters()) + list(model.fc.parameters()),
        lr=LR_HEAD, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    history           = []
    best_val_acc      = 0.0
    epochs_no_improve = 0

    for epoch in range(MAX_EPOCHS):
        if epoch == FREEZE_EPOCHS:
            print(f"\nunfreezing layer3 + layer4 at epoch {epoch+1}\n")
            unfreeze_late_layers(model)
            optimizer = optim.AdamW([
                {"params": [p for i, c in enumerate(model.features.children())
                             if i >= 6 for p in c.parameters()], "lr": LR_BACKBONE},
                {"params": list(model.dropout.parameters()) +
                            list(model.fc.parameters()),          "lr": LR_HEAD_2},
            ], weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
            epochs_no_improve = 0

        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, device)
        scheduler.step()

        lrs = [f"{pg['lr']:.1e}" for pg in optimizer.param_groups]
        print(f"epoch {epoch+1:3d}/{MAX_EPOCHS} | "
              f"train {train_loss:.4f}/{train_acc:.3f} | "
              f"val {val_loss:.4f}/{val_acc:.3f} | "
              f"lr {','.join(lrs)}")

        history.append({
            "epoch": epoch+1, "train_loss": round(train_loss,4),
            "train_acc": round(train_acc,4), "val_loss": round(val_loss,4),
            "val_acc": round(val_acc,4)
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "biosonar.pth"))
            print(f"  saved (val_acc={val_acc:.3f})")
        else:
            epochs_no_improve += 1

        if epoch >= FREEZE_EPOCHS and epochs_no_improve >= PATIENCE:
            print(f"\nearly stopping at epoch {epoch+1}")
            break

    pd.DataFrame(history).to_csv(os.path.join(MODEL_DIR, "training_log.csv"), index=False)
    print(f"\nbest val acc: {best_val_acc:.3f}")

    # test set
    print("\nrunning on test set...")
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "biosonar.pth"), map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            out = model(imgs.to(device))
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}
    results = pd.DataFrame({
        "true":      [idx_to_class[i] for i in all_labels],
        "predicted": [idx_to_class[i] for i in all_preds],
    })
    results.to_csv(os.path.join(MODEL_DIR, "test_results.csv"), index=False)

    test_acc = (results["true"] == results["predicted"]).mean()
    print(f"test accuracy: {test_acc:.3f}")


if __name__ == "__main__":
    train()
