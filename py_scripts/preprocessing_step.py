#!/usr/bin/env python3
"""Preprocessing Step: Convert audio files (.wav, .flac, .mp3) to .pt tensors.
   Skips files that have already been converted."""

import argparse
import random
from pathlib import Path
import torchaudio

import torch
import torchaudio
import soundfile as sf


def load_audio(src):
    """Load audio file using soundfile, return (waveform, sample_rate) matching torchaudio format."""        
    try:
        waveform, sample_rate = torchaudio.load(str(src))
        return waveform, sample_rate
    except RuntimeError as e:
            print(f"Failed to load {src}: {e}")
            return None, None


# ──────────────────────────────────────────────
# Generic converter (used by all datasets)
# ──────────────────────────────────────────────

def convert_first_n(base_dir, out_dir, n, extensions, save_dict=True):
    """Convert first n audio files in base_dir to .pt, skipping already-converted files.

    Args:
        n: Number of files to convert. 0 = all.
        save_dict: If True, saves {"waveform": ..., "sample_rate": ...}.
                   If False, saves raw waveform tensor only.
    """
    base_dir, out_dir = Path(base_dir), Path(out_dir)
    converted = []
    skipped = 0
    failed = []

    # Collect all source audio files
    all_files = []
    for ext in extensions:
        all_files.extend(sorted(base_dir.rglob(f"*{ext}")))

    if not all_files:
        print(f"No audio files found in {base_dir}")
        return []

    selected = all_files[:n] if n else all_files
    print(f"Found {len(all_files)} file(s). Processing {len(selected)}...")

    for i, f in enumerate(selected):
        # Check if .pt already exists — skip if so
        rel = f.relative_to(base_dir)
        dest = (out_dir / rel).with_suffix(".pt")
        if dest.exists():
            skipped += 1
            continue

        try:
            waveform, sample_rate = load_audio(f)
            dest.parent.mkdir(parents=True, exist_ok=True)

            if save_dict:
                torch.save({"waveform": waveform, "sample_rate": sample_rate}, dest)
            else:
                torch.save(waveform, dest)

            converted.append(dest)
        except Exception as e:
            print(f"Skipping {f.name}: {e}")
            failed.append(f)

        if (len(converted)) % 1000 == 0 and len(converted) > 0:
            print(f"Newly converted: {len(converted)} files so far...")

    if failed:
        print(f"\n{len(failed)} file(s) failed and were skipped:")
        for f in failed[:10]:
            print(f"  {f}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

    print(f"Done! Already existed: {skipped} | Newly converted: {len(converted)} | Failed: {len(failed)}")
    return converted


# ──────────────────────────────────────────────
# Verification
# ──────────────────────────────────────────────

def check_counts(working_dir):
    for part in ["codecfake_pt", "for_original_fake_pt", "asvspoof_pt"]:
        pt_dir = Path(f"{working_dir}/{part}")
        if pt_dir.exists():
            pt_files = list(pt_dir.rglob("*.pt"))
            print(f"{part}: {len(pt_files)} .pt files")
        else:
            print(f"{part}: directory not found")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preprocess audio datasets to .pt tensors")
    parser.add_argument("--working-dir", type=str, default="/ocean/projects/cis260121p/shared",
                        help="Working directory for outputs")
    parser.add_argument("--codecfake-dir", type=str,
                        default="/ocean/projects/cis260121p/shared/datasets/codecfake/dev",
                        help="Path to CodecFake .wav files")
    parser.add_argument("--asvspoof-dir", type=str,
                        default="/ocean/projects/cis260121p/shared/datasets/avsspoof_2021/ASVspoof2021_LA_eval/ASVspoof2021_LA_eval/flac",
                        help="Path to ASVspoof2021 .flac files")
    parser.add_argument("--for-dir", type=str,
                        default="/ocean/projects/cis260121p/shared/datasets/for/for-original/for-original/training/fake",
                        help="Path to FakeOrReal audio files")
    parser.add_argument("--codecfake-n", type=int, default=0,
                        help="Number of CodecFake files to convert (0 = all)")
    parser.add_argument("--asvspoof-n", type=int, default=30000,
                        help="Number of ASVspoof files to convert (0 = all)")
    parser.add_argument("--for-n", type=int, default=30000,
                        help="Number of FakeOrReal files to convert (0 = all)")
    args = parser.parse_args()

    # Step 1: CodecFake
    # print("=" * 60)
    # print("Step 1: Converting CodecFake .wav → .pt")
    # print("=" * 60)
    # convert_first_n(
    #     base_dir=args.codecfake_dir,
    #     out_dir=f"{args.working_dir}/codecfake_pt",
    #     n=args.codecfake_n,
    #     extensions={".wav"},
    #     save_dict=True,
    # )

    # Step 2: ASVspoof2021
    # print("\n" + "=" * 60)
    # print("Step 2: Converting ASVspoof2021 .flac → .pt")
    # print("=" * 60)
    # convert_first_n(
    #     base_dir=args.asvspoof_dir,
    #     out_dir=f"{args.working_dir}/asvspoof_pt",
    #     n=args.asvspoof_n,
    #     extensions={".wav", ".flac"},
    #     save_dict=True,
    # )

    # Step 3: FakeOrReal
    # print("\n" + "=" * 60)
    # print("Step 3: Converting FakeOrReal → .pt")
    # print("=" * 60)
    # convert_first_n(
    #     base_dir=args.for_dir,
    #     out_dir=f"{args.working_dir}/for_original_fake_pt",
    #     n=args.for_n,
    #     extensions={".wav", ".flac", ".mp3"},
    #     save_dict=False,
    # )

    # Step 4: Verify
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)
    check_counts(args.working_dir)


if __name__ == "__main__":
    main()