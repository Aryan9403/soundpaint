"""
Smoke test: load one .mp3, encode with DAC, decode back, save as .wav.

Usage:
    python scripts/test_dac.py
    python scripts/test_dac.py --audio_dir dataset/ --output test_roundtrip.wav
"""

import argparse
import sys
from pathlib import Path

import torch
import soundfile as sf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=str, default="dataset/")
    parser.add_argument("--output", type=str, default="test_roundtrip.wav")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    # Find one mp3
    audio_dir = Path(args.audio_dir)
    mp3_files = sorted(audio_dir.glob("**/*.mp3"))
    if not mp3_files:
        print(f"No .mp3 files found in {audio_dir}", file=sys.stderr)
        sys.exit(1)

    mp3_path = mp3_files[0]
    print(f"File: {mp3_path}")

    # Load DAC
    print("Loading DAC 44kHz model...")
    import dac
    dac_model = dac.DAC.load(dac.utils.download(model_type="44khz"))
    dac_model = dac_model.to(args.device)
    dac_model.eval()
    print(f"DAC loaded. n_codebooks={dac_model.quantizer.n_codebooks}")

    # Load audio
    print("Loading audio with audiotools...")
    import audiotools
    signal = audiotools.AudioSignal(str(mp3_path))
    signal = signal.resample(44100).to_mono()

    # Truncate to 10s for speed
    target = 10 * 44100
    audio = signal.audio_data  # (1, C, T) or (1, T)
    if audio.ndim == 3:
        audio = audio[:, 0, :]   # (1, T)
    audio = audio[..., :target]
    audio = audio.unsqueeze(0).to(args.device)   # (1, 1, T)

    print(f"Audio shape: {audio.shape} | range: [{audio.min():.3f}, {audio.max():.3f}]")

    # Encode
    print("Encoding...")
    with torch.no_grad():
        audio = dac_model.preprocess(audio, 44100)
        encoded = dac_model.encode(audio)

    if isinstance(encoded, (tuple, list)):
        z, codes, *_ = encoded
    else:
        z = encoded.z
        codes = encoded.codes

    print(f"codes shape: {codes.shape}")           # (1, n_codebooks, L)
    print(f"Codebook 0 tokens: {codes[0, 0, :10].tolist()}...")
    print(f"Token length: {codes.shape[-1]} tokens for {audio.shape[-1]/44100:.1f}s "
          f"= {codes.shape[-1]/(audio.shape[-1]/44100):.1f} tok/s")

    # Decode
    print("Decoding...")
    with torch.no_grad():
        z_q = dac_model.quantizer.from_codes(codes)[0]
        recon = dac_model.decode(z_q)  # (1, 1, T)

    print(f"Reconstructed shape: {recon.shape}")

    # Save
    audio_out = recon.squeeze().cpu().numpy()
    sf.write(args.output, audio_out, 44100)
    print(f"Saved: {args.output}")
    print("DAC roundtrip test PASSED")


if __name__ == "__main__":
    main()
