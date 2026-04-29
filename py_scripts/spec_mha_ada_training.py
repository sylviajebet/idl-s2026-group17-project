#!/usr/bin/env python3
"""SpecAugment + MHA-ADA Training (NON-DESTRUCTIVE VERSION)"""

import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)
from torch.utils.data import DataLoader, Dataset

from autoencoder_training import DeepAutoencoder

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ──────────────────────────────────────────────
# 🔹 SpecAugment
# ──────────────────────────────────────────────

def time_mask(spec, max_width=32):
    T = spec.shape[-1]
    width = random.randint(0, min(max_width, T))
    start = random.randint(0, max(0, T - width))
    spec[..., start:start + width] = 0
    return spec

def freq_mask(spec, max_width=16):
    Freq = spec.shape[-2]
    width = random.randint(0, min(max_width, Freq))
    start = random.randint(0, max(0, Freq - width))
    spec[..., start:start + width, :] = 0
    return spec

def apply_spec_augment(waveform, n_fft=512, hop_length=160):
    if waveform.ndim == 2:
        waveform = waveform.squeeze(0)

    window = torch.hann_window(n_fft, device=waveform.device)

    spec = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True
    )

    mag = torch.abs(spec)
    phase = torch.angle(spec)

    # Apply masks
    mag = time_mask(mag, int(mag.shape[-1] * 0.1))
    mag = freq_mask(mag, int(mag.shape[-2] * 0.1))

    spec_aug = mag * torch.exp(1j * phase)

    waveform = torch.istft(
        spec_aug,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        length=waveform.shape[-1]
    )

    return waveform.unsqueeze(0)


# ──────────────────────────────────────────────
# Dataset (WITH AUGMENT OPTION)
# ──────────────────────────────────────────────

class PathLabelDataset(Dataset):
    def __init__(self, csv_file, target_len=80000, augment=False):
        df = pd.read_csv(csv_file)
        self.samples = [(Path(p), l) for p, l in zip(df["path"], df["label"])]
        self.target_len = target_len
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        data = torch.load(path)
        tensor = data["waveform"].float() if isinstance(data, dict) else data.float()

        # pad / trim
        length = tensor.shape[1]
        if length < self.target_len:
            tensor = F.pad(tensor, (0, self.target_len - length))
        else:
            tensor = tensor[:, :self.target_len]

        # 🔥 ONLY APPLY ON TRAINING
        if self.augment:
            tensor = apply_spec_augment(tensor)

        return tensor, label


# ──────────────────────────────────────────────
# Model (same as yours)
# ──────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, z):
        z = z.permute(0, 2, 1)
        z_norm = self.norm(z)
        attn_out, _ = self.attn(z_norm, z_norm, z_norm)
        z = z + attn_out
        return z.permute(0, 2, 1)


class AudioDeepfakeAttributionModel(nn.Module):
    def __init__(self, autoencoder, num_classes=3):
        super().__init__()
        self.encoder = autoencoder.encoder

        # freeze encoder (same as original)
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.attention = MultiHeadSelfAttention()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        z = self.encoder(x)
        z = self.attention(z)
        return self.classifier(z)


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train_model(model, train_loader, val_loader, device, epochs=20, lr=1e-4,
                save_path="models/spec_MHA_ADA_model.pt", checkpoint_dir="checkpoints/spec_MHA_ADA",
                resume_from=None, save_freq=5):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_loss", float("inf"))
        train_losses = checkpoint.get("train_losses", [])
        val_losses = checkpoint.get("val_losses", [])
        print(f"Resumed from epoch {start_epoch} (best val loss: {best_val_loss:.4f})")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train = total_loss / len(train_loader)
        train_losses.append(avg_train)
        print(f"Train Loss: {avg_train:.4f}")

        model.eval()
        val_loss = 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                val_loss += criterion(output, y).item()
                y_true.extend(y.cpu().numpy())
                y_pred.extend(output.argmax(dim=1).cpu().numpy())

        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)
        print(classification_report(y_true, y_pred, digits=4))

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

        if (epoch + 1) % save_freq == 0 or (epoch + 1) == epochs:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_loss": best_val_loss,
                "train_losses": train_losses,
                "val_losses": val_losses,
            }, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────

def evaluate_and_plot(model, loader, device, report_path, plot_dir):
    """Full evaluation with confusion matrix, ROC, PR, and t-SNE plots."""
    model.eval()
    y_true, y_pred, logits_list, latents = [], [], [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            z = model.encoder(x)
            pooled = F.adaptive_avg_pool1d(z, 1).squeeze(-1)
            latents.append(pooled.cpu())
            out = model(x)
            y_pred.extend(out.argmax(dim=1).cpu().numpy())
            y_true.extend(y.cpu().numpy())
            logits_list.append(out.cpu())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    probs = torch.softmax(torch.cat(logits_list), dim=1).numpy()
    latents = torch.cat(latents, dim=0).numpy()

    report = classification_report(y_true, y_pred, output_dict=True, digits=4)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    pd.DataFrame(report).transpose().to_csv(report_path)

    os.makedirs(plot_dir, exist_ok=True)

    # Confusion matrix
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(plot_dir, "confusion_matrix.png"))
    plt.close()

    # ROC Curves
    y_bin = label_binarize(y_true, classes=list(range(3)))
    plt.figure(figsize=(8, 6))
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        plt.plot(fpr, tpr, label=f"Class {i} AUC={auc(fpr, tpr):.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title("ROC Curves")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "roc_curves.png"))
    plt.close()

    # PR Curves
    plt.figure(figsize=(8, 6))
    for i in range(3):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], probs[:, i])
        ap = average_precision_score(y_bin[:, i], probs[:, i])
        plt.plot(rec, prec, label=f"Class {i} AP={ap:.2f}")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "pr_curves.png"))
    plt.close()

    # t-SNE 2D
    tsne_2d = TSNE(n_components=2, random_state=SEED, perplexity=min(30, len(latents) - 1))
    emb_2d = tsne_2d.fit_transform(latents)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=emb_2d[:, 0], y=emb_2d[:, 1], hue=y_true, palette="tab10", legend="full")
    plt.title("t-SNE (2D) of Latent Space")
    plt.savefig(os.path.join(plot_dir, "tsne_2d.png"))
    plt.close()

    # t-SNE 3D
    tsne_3d = TSNE(n_components=3, random_state=SEED, perplexity=min(30, len(latents) - 1))
    emb_3d = tsne_3d.fit_transform(latents)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(emb_3d[:, 0], emb_3d[:, 1], emb_3d[:, 2], c=y_true, cmap="tab10")
    legend = ax.legend(*scatter.legend_elements(), title="Class")
    ax.add_artist(legend)
    ax.set_title("t-SNE (3D) of Latent Space")
    plt.savefig(os.path.join(plot_dir, "tsne_3d.png"))
    plt.close()

    print(f"Report saved to {report_path}")
    print(f"Plots saved to {plot_dir}")


# ──────────────────────────────────────────────
# Confidence Thresholding
# ──────────────────────────────────────────────

def compute_confidence_scores(model, loader, device, output_csv):
    model.eval()
    scores, true_labels, pred_labels = [], [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Computing confidence scores"):
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            max_conf, pred = probs.max(dim=1)
            scores.extend(max_conf.cpu().numpy())
            pred_labels.extend(pred.cpu().numpy())
            true_labels.extend(y.numpy())

    df = pd.DataFrame({"confidence": scores, "true_label": true_labels, "pred_label": pred_labels})
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Confidence scores saved to {output_csv}")


def find_confidence_thresholds(csv_path):
    df = pd.read_csv(csv_path)
    confidences = df["confidence"].values
    true_labels = df["true_label"].values
    pred_labels = df["pred_label"].values

    thresholds = np.linspace(0.0, 1.0, 500)
    best_acc, best_thresh = 0, 0
    for t in thresholds:
        mask = confidences >= t
        if np.sum(mask) == 0:
            continue
        accuracy = np.mean(pred_labels[mask] == true_labels[mask])
        coverage = np.sum(mask) / len(true_labels)
        if coverage >= 0.80 and accuracy > best_acc:
            best_acc = accuracy
            best_thresh = t

    print(f"Coverage >=80% Threshold: {best_thresh:.4f} | Accuracy: {best_acc:.4f}")
    return best_thresh


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train MHA-ADA model for Audio Deepfake Attribution")
    parser.add_argument("--working-dir", type=str, default="/ocean/projects/cis260121p/shared")
    parser.add_argument("--autoencoder-path", type=str, default="/ocean/projects/cis260121p/shared/models/spec_autoencoder.pt")
    parser.add_argument("--save-path", type=str, default="/ocean/projects/cis260121p/shared/models/spec_MHA_ADA_model.pt")
    parser.add_argument("--csv-dir", type=str, default="/ocean/projects/cis260121p/shared/csv/MHA_ADA_split") # Remain as is, reusing same split as MHA_ADA
    parser.add_argument("--checkpoint-dir", type=str, default="/ocean/projects/cis260121p/shared/checkpoints/spec_MHA_ADA")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--skip-split", action="store_true",
                        help="Skip CSV split preparation (use if already done)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # datasets
    train_set = PathLabelDataset(os.path.join(args.csv_dir, "train.csv"), augment=True)
    val_set = PathLabelDataset(os.path.join(args.csv_dir, "val.csv"), augment=False)
    test_set = PathLabelDataset(os.path.join(args.csv_dir, "test.csv"), augment=False)

    # dataloaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    # model
    autoencoder = DeepAutoencoder().to(device)
    autoencoder.load_state_dict(torch.load(args.autoencoder_path, map_location=device))
    model = AudioDeepfakeAttributionModel(autoencoder).to(device)

    # train
    train_model(model, train_loader, val_loader, device,
                epochs=args.epochs, lr=args.lr, save_path=args.save_path,
                checkpoint_dir=args.checkpoint_dir, resume_from=args.resume)

    # evaluate
    model.load_state_dict(torch.load(args.save_path, map_location=device))

    conf_csv = os.path.join(args.working_dir, "csv/spec_MHA_ADA/confidence_scores.csv")
    compute_confidence_scores(model, train_loader, device, output_csv=conf_csv)
    find_confidence_thresholds(conf_csv)

    evaluate_and_plot(
        model=model,
        loader=test_loader,
        device=device,
        report_path=os.path.join(args.working_dir, "reports/spec_MHA_ADA_test_report.csv"),
        plot_dir=os.path.join(args.working_dir, "plots/spec_MHA_ADA/"),
    )


if __name__ == "__main__":
    main()
