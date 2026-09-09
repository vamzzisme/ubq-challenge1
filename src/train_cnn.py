#!/usr/bin/env python3
"""Train a compact 1D-CNN for activity recognition on raw accelerometer windows.

Architecture: 3 conv blocks (Conv1d → BN → ReLU → MaxPool) + GAP + FC head.
Input: 500 × 3 raw accelerometer (x, y, z) at 25 Hz.
Same user-level train/val/test split as the RF baselines.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from signal_features import load_axes


# ── Dataset ──────────────────────────────────────────────────────────────────

def preload_data(rows: list[dict[str, str]], class_to_idx: dict[str, int], desc: str = "data") -> tuple[torch.Tensor, torch.Tensor]:
    """Load all CSV files into RAM at once, returning (N, 3, 500) and (N,) tensors."""
    arrays: list[np.ndarray] = []
    labels: list[int] = []
    skipped = 0
    for i, row in enumerate(rows):
        try:
            acc_path = Path(row["sensor_csv_path"])
            gyro_path = Path(str(acc_path).replace("raw_acc", "proc_gyro"))
            if not gyro_path.exists():
                skipped += 1
                continue
                
            acc_axes = load_axes(acc_path)    # (500, 3)
            gyro_axes = load_axes(gyro_path)  # (500, 3)
            
            # Per-window normalisation
            acc_mean = acc_axes.mean(axis=0, keepdims=True)
            acc_std = acc_axes.std(axis=0, keepdims=True) + 1e-8
            acc_axes = (acc_axes - acc_mean) / acc_std
            
            gyro_mean = gyro_axes.mean(axis=0, keepdims=True)
            gyro_std = gyro_axes.std(axis=0, keepdims=True) + 1e-8
            gyro_axes = (gyro_axes - gyro_mean) / gyro_std
            
            combined = np.hstack((acc_axes, gyro_axes)) # (500, 6)
            arrays.append(combined.T)  # (6, 500)
            labels.append(class_to_idx[row["activity"]])
        except Exception:
            skipped += 1
        if (i + 1) % 5000 == 0:
            print(f"    Pre-loaded {i + 1}/{len(rows)} {desc} recordings...")
    if skipped:
        print(f"    Skipped {skipped} unreadable recordings in {desc}")
    data = torch.from_numpy(np.stack(arrays)).float()  # (N, 6, 500)
    targets = torch.tensor(labels, dtype=torch.long)   # (N,)
    print(f"    {desc}: {data.shape[0]} samples, {data.shape[1]}×{data.shape[2]}, {data.element_size() * data.nelement() / 1e6:.0f} MB")
    return data, targets


def augment_window(x: torch.Tensor) -> torch.Tensor:
    """Apply random augmentations to a (6, 500) sensor window."""
    # Jitter: add small Gaussian noise
    if torch.rand(1).item() < 0.8:
        x = x + torch.randn_like(x) * 0.05
    # Scaling: random per-channel scale
    if torch.rand(1).item() < 0.5:
        scale = 0.8 + 0.4 * torch.rand(6, 1)  # [0.8, 1.2]
        x = x * scale
    # Time shift: roll along time axis
    if torch.rand(1).item() < 0.5:
        shift = torch.randint(-50, 51, (1,)).item()
        x = torch.roll(x, shifts=int(shift), dims=1)
    # Random time-warp via resampling a segment (Sensor Dropout)
    if torch.rand(1).item() < 0.5:
        seg_len = torch.randint(20, 100, (1,)).item()
        start = torch.randint(0, 500 - seg_len, (1,)).item()
        x[:, start:start + seg_len] = 0.0
    # Axis Permutation (simulates different phone orientations)
    if torch.rand(1).item() < 0.3:
        perm = torch.randperm(3)
        x[:3, :] = x[perm, :]
        x[3:, :] = x[3 + perm, :]
    return x


class AccelWindowDataset(Dataset):
    """Pre-cached dataset: all sensor windows live in RAM as tensors."""

    def __init__(self, data: torch.Tensor, targets: torch.Tensor, augment: bool = False) -> None:
        self.data = data        # (N, 3, 500)
        self.targets = targets  # (N,)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        x = self.data[index]
        if self.augment:
            x = augment_window(x.clone())
        return x, int(self.targets[index])


# ── Model ────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: bool = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_ch)
        
        self.pool = nn.MaxPool1d(2) if pool else nn.Identity()
        
        if in_ch != out_ch or pool:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1),
                self.pool
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.pool(out)
        out += identity
        out = self.relu(out)
        return out


class AccelCNN(nn.Module):
    """Robust 1D-ResNet + BiLSTM for complex activity patterns."""

    def __init__(self, num_classes: int = 6, in_channels: int = 6) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )
        
        self.layer1 = ResBlock(32, 64, pool=True)
        self.layer2 = ResBlock(64, 128, pool=True)
        self.layer3 = ResBlock(128, 128, pool=True)
        
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, bidirectional=True)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = out.permute(0, 2, 1) 
        out, _ = self.lstm(out)
        
        out = out.mean(dim=1) 
        
        return self.classifier(out)


# ── Training ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("data/processed/raw_acc_training_index.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/cnn"))
    parser.add_argument("--validation-user", default="2C32C23E-E30C-498A-8DD2-0EFB9150A02E")
    parser.add_argument("--test-user", default="0A986513-7828-4D53-AA1F-E02D6DF9561B")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 = main thread).")
    return parser.parse_args()


def read_index(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, class_names: list[str]) -> dict[str, Any]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(y.cpu())
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, labels=list(range(len(class_names))), average="macro", zero_division=0)
    report = classification_report(
        labels,
        preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    return {"accuracy": acc, "macro_f1": f1, "report": report, "confusion_matrix": cm, "predictions": preds, "labels": labels}


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    rows = read_index(args.index)
    train_rows = [r for r in rows if r["user_id"] not in {args.validation_user, args.test_user}]
    val_rows = [r for r in rows if r["user_id"] == args.validation_user]
    test_rows = [r for r in rows if r["user_id"] == args.test_user]

    all_activities = sorted(set(r["activity"] for r in rows))
    class_to_idx = {a: i for i, a in enumerate(all_activities)}
    num_classes = len(all_activities)
    print(f"Classes ({num_classes}): {all_activities}")
    print(f"Train: {len(train_rows)}, Val: {len(val_rows)}, Test: {len(test_rows)}")

    # Pre-load ALL data into RAM (avoids per-sample CSV I/O during training)
    print("Pre-loading all sensor data into memory...")
    train_data, train_targets = preload_data(train_rows, class_to_idx, "train")
    val_data, val_targets = preload_data(val_rows, class_to_idx, "val")
    test_data, test_targets = preload_data(test_rows, class_to_idx, "test")

    train_ds = AccelWindowDataset(train_data, train_targets, augment=True)
    val_ds = AccelWindowDataset(val_data, val_targets, augment=False)
    test_ds = AccelWindowDataset(test_data, test_targets, augment=False)

    # Weighted sampler for class imbalance
    train_labels = train_targets.tolist()
    class_counts = Counter(train_labels)
    weights = [1.0 / class_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    use_pin = device.type not in ("mps",)  # pin_memory unsupported on MPS
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=use_pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=use_pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=use_pin)

    # Class-weighted loss
    total = sum(class_counts.values())
    class_weights = torch.tensor([total / (num_classes * class_counts[i]) for i in range(num_classes)], dtype=torch.float32).to(device)

    model = AccelCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    best_val_f1 = 0.0
    best_epoch = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        t0 = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            # MixUp implementation
            if torch.rand(1).item() < 0.3:
                lam = np.random.beta(0.2, 0.2)
                index = torch.randperm(x.size(0)).to(device)
                x = lam * x + (1 - lam) * x[index, :]
                y_a, y_b = y, y[index]
                logits = model(x)
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            else:
                logits = model(x)
                loss = criterion(logits, y)
                
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
            epoch_correct += (logits.argmax(1) == y).sum().item()
            epoch_total += x.size(0)

        scheduler.step()
        train_acc = epoch_correct / epoch_total
        train_loss = epoch_loss / epoch_total

        val_result = evaluate(model, val_loader, device, all_activities)
        elapsed = time.time() - t0
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "val_acc": val_result["accuracy"], "val_f1": val_result["macro_f1"]})

        print(f"Epoch {epoch:3d}/{args.epochs} | loss={train_loss:.4f} train_acc={train_acc:.3f} "
              f"val_acc={val_result['accuracy']:.3f} val_f1={val_result['macro_f1']:.3f} [{elapsed:.1f}s]")

        if val_result["macro_f1"] > best_val_f1:
            best_val_f1 = val_result["macro_f1"]
            best_epoch = epoch
            args.output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.output_dir / "cnn_model.pt")

    print(f"\nBest val F1={best_val_f1:.3f} at epoch {best_epoch}")
    print("Loading best model for test evaluation...")
    model.load_state_dict(torch.load(args.output_dir / "cnn_model.pt", weights_only=True))
    test_result = evaluate(model, test_loader, device, all_activities)

    print(f"Test accuracy={test_result['accuracy']:.3f}, macro-F1={test_result['macro_f1']:.3f}")

    # Save confusion matrix CSV (same format as RF baselines)
    cm = test_result["confusion_matrix"]
    with (args.output_dir / "test_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true_activity", *all_activities])
        for label, row_vals in zip(all_activities, cm):
            writer.writerow([label, *row_vals])

    # Save metrics JSON (same schema as RF baselines)
    val_final = evaluate(model, val_loader, device, all_activities)
    summary: dict[str, Any] = {
        "split": {
            "train_users": sorted(set(r["user_id"] for r in train_rows)),
            "validation_user": args.validation_user,
            "test_user": args.test_user,
        },
        "recordings": {"train": len(train_rows), "validation": len(val_rows), "test": len(test_rows)},
        "train_class_counts": dict(Counter(r["activity"] for r in train_rows)),
        "model_config": {
            "model_type": "AccelCNN",
            "parameters": param_count,
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
        },
        "validation": {k: v for k, v in val_final.items() if k not in {"confusion_matrix", "predictions", "labels"}},
        "test": {k: v for k, v in test_result.items() if k not in {"confusion_matrix", "predictions", "labels"}},
        "training_history": history,
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save model info for inference compatibility
    torch.save({
        "state_dict": model.state_dict(),
        "class_names": all_activities,
        "class_to_idx": class_to_idx,
        "num_classes": num_classes,
        "sample_rate_hz": 25.0,
        "model_config": summary["model_config"],
    }, args.output_dir / "cnn_checkpoint.pt")

    print(f"Wrote model and metrics to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
