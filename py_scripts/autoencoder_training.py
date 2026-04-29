#!/usr/bin/env python3
"""Autoencoder Training: Train a deep convolutional autoencoder on CodecFake audio data."""

import argparse
import os
import re
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class CodecFakeMultiClassDataset(Dataset):
    """
    PyTorch Dataset for CodecFake deepfake audio samples.
    Extracts class labels from filename patterns (F01–F06).
    """
    def __init__(self, root_dir, seed=None, target_len=80000, samples=None):
        self.samples = []
        self.target_len = target_len
        root = Path(root_dir)

        for file_path in sorted(root.glob("*.pt")):
            match = re.search(r"F(\d+)_", file_path.name)
            if match:
                label = int(match.group(1))
                self.samples.append((file_path, label))

        class_counts = {}
        for _, label in self.samples:
            class_counts[label] = class_counts.get(label, 0) + 1
        for label, count in sorted(class_counts.items()):
            print(f"Class {label}: {count} samples")
        print(f"Total samples: {len(self.samples)}")

        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(self.samples)
        else:
            random.shuffle(self.samples)

        if not self.samples:
            raise ValueError("No valid .pt files found or unable to extract labels.")

        if samples is None:
            samples = len(self.samples)
        
        # Uniform selection
        grouped = {}
        for fp, lbl in self.samples:
            grouped.setdefault(lbl, []).append((fp, lbl))

        selected = []
        # Split total samples evenly across classes
        n_classes = len(grouped)
        per_class = samples // n_classes
        for lbl, files in grouped.items():
            selected.extend(files[:per_class])

        self.samples = selected
        print(f"Uniformly selected {len(self.samples)} samples ({per_class} per class).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data_loaded = torch.load(path)
        tensor = data_loaded["waveform"].float() if isinstance(data_loaded, dict) else data_loaded.float()
        label = label - 1

        length = tensor.shape[1]
        if length < self.target_len:
            tensor = F.pad(tensor, (0, self.target_len - length))
        else:
            tensor = tensor[:, :self.target_len]
        return tensor, label


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────

class DeepAutoencoder(nn.Module):
    """Deep Convolutional Autoencoder for audio waveform feature learning."""
    def __init__(self):
        super(DeepAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.ConvTranspose1d(32, 1, kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train_with_checkpoints(model, train_loader, val_loader, lr, epochs, save_path,
                           checkpoint_dir="checkpoints", resume_path=None, save_freq=5):
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss(beta=0.0001)

    start_epoch = 0
    best_val_loss = float("inf")

    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_loss"]
        print(f"Resumed from epoch {start_epoch}")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, epochs):
        # Training
        model.train()
        train_loss = 0
        for x, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x = x.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                val_loss += criterion(model(x), x).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch+1:02d} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

        # Save best model
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved to {save_path}")

        # Save checkpoint
        if (epoch + 1) % save_freq == 0 or (epoch + 1) == epochs:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_loss": best_val_loss,
            }, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Deep Autoencoder on CodecFake data")
    parser.add_argument("--data-dir", type=str, default="/ocean/projects/cis260121p/shared/pt_files/codecfake_pt",
                        help="Path to CodecFake .pt files")
    parser.add_argument("--save-path", type=str, default="/ocean/projects/cis260121p/shared/models/autoencoder.pt",
                        help="Path to save the best model")
    parser.add_argument("--checkpoint-dir", type=str, default="/ocean/projects/cis260121p/shared/checkpoints/autoencoder",
                        help="Directory for training checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-freq", type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--max-samples", type=int, default=500,
                    help="Max samples to use (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = CodecFakeMultiClassDataset(root_dir=args.data_dir, seed=args.seed)
    if args.max_samples and len(dataset) > args.max_samples:
        dataset, _ = data.random_split(dataset, [args.max_samples, len(dataset) - args.max_samples],
                                    generator=torch.Generator().manual_seed(args.seed))

    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    val_len = total_len - train_len
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = data.random_split(dataset, [train_len, val_len], generator=generator)

    print(f"Total: {total_len} | Train: {len(train_set)} | Val: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    model = DeepAutoencoder().to(device)

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    print(f"Model will be saved to {args.save_path}")

    train_with_checkpoints(
        model, train_loader, val_loader,
        lr=args.lr,
        epochs=args.epochs,
        save_path=args.save_path,
        checkpoint_dir=args.checkpoint_dir,
        resume_path=args.resume,
        save_freq=args.save_freq,
    )


if __name__ == "__main__":
    main()
