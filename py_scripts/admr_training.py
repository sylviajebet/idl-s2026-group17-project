#!/usr/bin/env python3
"""ADMR Training: Audio Deepfake Model Recognition — classify by generation model (6 classes)."""

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

# Import shared components
from autoencoder_training import DeepAutoencoder, CodecFakeMultiClassDataset
from ada_training import PathLabelDataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ──────────────────────────────────────────────
# Data Split Preparation
# ──────────────────────────────────────────────

def prepare_ADMR_splits(data_dir, output_csv_dir, samples):
    """Create stratified train/val/test CSV splits for ADMR (6-class)."""
    os.makedirs(output_csv_dir, exist_ok=True)
    dataset = CodecFakeMultiClassDataset(root_dir=data_dir, seed=SEED, samples=samples)
    all_samples = dataset.samples

    train_val, test = train_test_split(
        all_samples, test_size=0.2,
        stratify=[lbl for _, lbl in all_samples], random_state=SEED
    )
    train, val = train_test_split(
        train_val, test_size=0.25,
        stratify=[lbl for _, lbl in train_val], random_state=SEED
    )

    for name, split in {"train": train, "val": val, "test": test}.items():
        df = pd.DataFrame([(str(p), l - 1) for p, l in split], columns=["path", "label"])
        save_path = os.path.join(output_csv_dir, f"{name}.csv")
        df.to_csv(save_path, index=False)
        print(f"Saved {name} split with {len(split)} samples.")


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────

class ADMR_model(nn.Module):
    """ADMR model: pretrained encoder + attention + 6-class classifier."""
    def __init__(self, pretrained_autoencoder):
        super(ADMR_model, self).__init__()
        self.encoder = pretrained_autoencoder.encoder
        for name, param in self.encoder.named_parameters():
            if name not in ["10.weight", "10.bias"]:
                param.requires_grad = False

        self.attention = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6),
        )

    def forward(self, x):
        z = self.encoder(x)
        a = self.attention(z)
        z = z * a
        return self.classifier(z)


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train_ADMR_model(model, train_loader, val_loader, device, epochs=10, lr=1e-4,
                     save_path="models/ADMR_model.pt", checkpoint_dir="checkpoints/ADMR",
                     resume_from=None, save_freq=5):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    best_val_loss = float("inf")

    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item()
                y_true.extend(y.cpu().numpy())
                y_pred.extend(logits.argmax(dim=1).cpu().numpy())

        avg_val = val_loss / len(val_loader)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

        print(f"\nEpoch {epoch+1} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        print(classification_report(y_true, y_pred, digits=4))

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
# Evaluation
# ──────────────────────────────────────────────

def evaluate_model(model, test_loader, device, report_path, plot_path):
    """Full evaluation with confusion matrix, ROC, PR, and t-SNE plots."""
    model.eval()
    y_true, y_pred, logits_list, latent_features = [], [], [], []

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Evaluating on test set"):
            x, y = x.to(device), y.to(device)
            z = model.encoder(x)
            pooled = F.adaptive_avg_pool1d(z, 1).squeeze(-1)
            latent_features.append(pooled.cpu())
            logits = model(x)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy())
            logits_list.append(logits.cpu())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    probs = torch.softmax(torch.cat(logits_list, dim=0), dim=1).numpy()
    latents = torch.cat(latent_features, dim=0).numpy()

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, digits=4)
    print(f"\nTest Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, digits=4))

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    pd.DataFrame(report).transpose().to_csv(report_path)
    os.makedirs(plot_path, exist_ok=True)

    # Confusion matrix
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=[f"C{i+1}" for i in range(6)],
                yticklabels=[f"C{i+1}" for i in range(6)])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(plot_path, "confusion_matrix.png"))
    plt.close()

    # t-SNE 2D
    perp = min(30, len(latents) - 1)
    tsne_2d = TSNE(n_components=2, random_state=SEED, perplexity=perp)
    emb_2d = tsne_2d.fit_transform(latents)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=emb_2d[:, 0], y=emb_2d[:, 1], hue=y_true, palette="tab10", legend="full")
    plt.title("t-SNE (2D) of Test Latent Representations")
    plt.savefig(os.path.join(plot_path, "tsne_2d.png"))
    plt.close()

    # t-SNE 3D
    tsne_3d = TSNE(n_components=3, random_state=SEED, perplexity=perp)
    emb_3d = tsne_3d.fit_transform(latents)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(emb_3d[:, 0], emb_3d[:, 1], emb_3d[:, 2], c=y_true, cmap="tab10")
    legend = ax.legend(*scatter.legend_elements(), title="Class")
    ax.add_artist(legend)
    ax.set_title("t-SNE (3D) of Test Latent Representations")
    plt.savefig(os.path.join(plot_path, "tsne_3d.png"))
    plt.close()

    # ROC & PR Curves
    y_bin = label_binarize(y_true, classes=list(range(6)))
    plt.figure(figsize=(8, 6))
    for i in range(6):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        plt.plot(fpr, tpr, label=f"C{i+1} (AUC={auc(fpr, tpr):.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.savefig(os.path.join(plot_path, "roc_curves.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    for i in range(6):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], probs[:, i])
        ap = average_precision_score(y_bin[:, i], probs[:, i])
        plt.plot(rec, prec, label=f"C{i+1} (AP={ap:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.savefig(os.path.join(plot_path, "pr_curves.png"))
    plt.close()

    print(f"Report saved to {report_path}")
    print(f"Plots saved to {plot_path}")


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
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train ADMR model for Deepfake Model Recognition")
    parser.add_argument("--working-dir", type=str, default="/ocean/projects/cis260121p/shared")
    parser.add_argument("--data-dir", type=str, default="/ocean/projects/cis260121p/shared/codecfake_pt",
                        help="Path to CodecFake .pt files")
    parser.add_argument("--autoencoder-path", type=str, default="/ocean/projects/cis260121p/shared/models/autoencoder.pt")
    parser.add_argument("--save-path", type=str, default="/ocean/projects/cis260121p/shared/models/ADMR_model.pt")
    parser.add_argument("--csv-dir", type=str, default="/ocean/projects/cis260121p/shared/csv/ADMR_split")
    parser.add_argument("--checkpoint-dir", type=str, default="/ocean/projects/cis260121p/shared/checkpoints/ADMR")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--samples", type=int, default=25000)
    parser.add_argument("--skip-split", action="store_true",
                        help="Skip CSV split preparation (use if already done)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Step 1: Prepare splits
    if not args.skip_split:
        print("Preparing ADMR splits...")
        prepare_ADMR_splits(args.data_dir, args.csv_dir, args.samples)

    # Step 2: Load model
    autoencoder = DeepAutoencoder().to(device)
    autoencoder.load_state_dict(torch.load(args.autoencoder_path, map_location=device))
    model = ADMR_model(pretrained_autoencoder=autoencoder).to(device)

    # Step 3: Dataloaders
    train_set = PathLabelDataset(os.path.join(args.csv_dir, "train.csv"))
    val_set = PathLabelDataset(os.path.join(args.csv_dir, "val.csv"))
    test_set = PathLabelDataset(os.path.join(args.csv_dir, "test.csv"))

    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    # Step 4: Train
    train_ADMR_model(model, train_loader, val_loader, device,
                     epochs=args.epochs, lr=args.lr, save_path=args.save_path,
                     checkpoint_dir=args.checkpoint_dir, resume_from=args.resume)

    # Step 5: Evaluate
    model.load_state_dict(torch.load(args.save_path, map_location=device))
    evaluate_model(model, test_loader, device,
                   report_path=os.path.join(args.working_dir, "reports/ADMR/test_report.csv"),
                   plot_path=os.path.join(args.working_dir, "plots/ADMR/"))

    # Step 6: Confidence analysis
    conf_csv = os.path.join(args.working_dir, "csv/ADMR/confidence_scores.csv")
    compute_confidence_scores(model, train_loader, device, output_csv=conf_csv)
    find_confidence_thresholds(conf_csv)


if __name__ == "__main__":
    main()
