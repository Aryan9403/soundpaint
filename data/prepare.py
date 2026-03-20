"""
Tokenize audio files using DAC (Descript Audio Codec).

Extracts codebook 0 tokens only → 1D LongTensor of length ~2580 (30s at 86 tok/s).

Usage:
    python data/prepare.py --audio_dir dataset/ --output_dir data/tokens_small --device cuda
    python data/prepare.py --audio_dir dataset/ --output_dir data/tokens_small --device cpu
"""

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize audio with DAC")
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory of .mp3 files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save .pt token files")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--duration", type=float, default=30.0, help="Duration in seconds to extract per file")
    return parser.parse_args()


def load_audio(path: Path, sample_rate: int, duration: float):
    """Load audio file, resample to mono at target sample_rate, truncate/pad to duration."""
    import audiotools
    signal = audiotools.AudioSignal(str(path))
    signal = signal.resample(sample_rate)
    signal = signal.to_mono()

    target_samples = int(sample_rate * duration)
    audio = signal.audio_data  # (1, 1, T) or (1, C, T)

    # Flatten to (1, T)
    if audio.ndim == 3:
        audio = audio[:, 0, :]  # take channel 0 → (1, T)

    # Truncate or pad
    T = audio.shape[-1]
    if T >= target_samples:
        audio = audio[..., :target_samples]
    else:
        pad = torch.zeros(1, target_samples - T)
        audio = torch.cat([audio, pad], dim=-1)

    return audio  # (1, T)


def tokenize_file(path: Path, dac_model, sample_rate: int, duration: float, device: str) -> torch.Tensor:
    """Encode one audio file → codebook 0 tokens, shape (L,)."""
    audio = load_audio(path, sample_rate, duration)  # (1, T)
    audio = audio.unsqueeze(0).to(device)  # (1, 1, T) → batch of 1

    with torch.no_grad():
        audio = dac_model.preprocess(audio, sample_rate)  # normalize
        encoded = dac_model.encode(audio)  # returns DACFile or tuple

    # DAC encode returns (z, codes, latents, commitment_loss, codebook_loss)
    # codes shape: (B, n_codebooks, L)
    if isinstance(encoded, (tuple, list)):
        codes = encoded[1]  # (B, n_codebooks, L)
    else:
        codes = encoded.codes  # DACFile object

    # Codebook 0 only → (L,)
    tokens = codes[0, 0, :]  # (L,)
    return tokens.long().cpu()


def main():
    args = parse_args()

    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mp3_files = sorted(audio_dir.glob("**/*.mp3"))
    if not mp3_files:
        print(f"No .mp3 files found in {audio_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(mp3_files)} .mp3 files")
    print(f"Loading DAC 44kHz model...")

    import dac
    dac_model = dac.DAC.load(dac.utils.download(model_type="44khz"))
    dac_model = dac_model.to(args.device)
    dac_model.eval()

    sample_rate = 44100
    skipped = []
    processed = 0

    for path in tqdm(mp3_files, desc="Tokenizing"):
        out_path = output_dir / (path.stem + ".pt")
        if out_path.exists():
            processed += 1
            continue

        try:
            tokens = tokenize_file(path, dac_model, sample_rate, args.duration, args.device)
            torch.save(tokens, out_path)
            processed += 1
        except Exception as e:
            print(f"\nSkipping {path.name}: {e}", file=sys.stderr)
            skipped.append(str(path))

    print(f"\nDone. Processed: {processed}, Skipped: {len(skipped)}")
    if skipped:
        print("Skipped files:")
        for f in skipped:
            print(f"  {f}")

    # Print token shape info from first successful file
    pt_files = sorted(output_dir.glob("*.pt"))
    if pt_files:
        sample = torch.load(pt_files[0], weights_only=True)
        print(f"Token shape example: {sample.shape} | dtype: {sample.dtype}")
        print(f"Total .pt files: {len(pt_files)}")


if __name__ == "__main__":
    main()
