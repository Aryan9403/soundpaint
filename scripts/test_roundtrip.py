"""
Encode/decode 5 files and report SNR / basic stats to verify codec quality.

Usage:
    python scripts/test_roundtrip.py --audio_dir dataset/ --device cpu
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import soundfile as sf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=str, default="dataset/")
    parser.add_argument("--n_files", type=int, default=5)
    parser.add_argument("--duration", type=float, default=5.0, help="Seconds to test per file")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def snr_db(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Signal-to-noise ratio in dB."""
    noise = original - reconstructed
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-12:
        return float("inf")
    return 10 * np.log10(signal_power / (noise_power + 1e-12))


def main():
    args = parse_args()

    audio_dir = Path(args.audio_dir)
    mp3_files = sorted(audio_dir.glob("**/*.mp3"))[: args.n_files]
    if not mp3_files:
        print(f"No .mp3 files found in {audio_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading DAC 44kHz model...")
    import dac
    dac_model = dac.DAC.load(dac.utils.download(model_type="44khz"))
    dac_model = dac_model.to(args.device)
    dac_model.eval()

    import audiotools

    results = []
    target_samples = int(args.duration * 44100)

    for path in mp3_files:
        print(f"\nFile: {path.name}")
        try:
            signal = audiotools.AudioSignal(str(path))
            signal = signal.resample(44100).to_mono()

            audio = signal.audio_data
            if audio.ndim == 3:
                audio = audio[:, 0, :]
            audio = audio[..., :target_samples]

            # Pad if shorter than target
            if audio.shape[-1] < target_samples:
                pad = torch.zeros(1, target_samples - audio.shape[-1])
                audio = torch.cat([audio, pad], dim=-1)

            audio_np = audio.squeeze().numpy()
            audio_tensor = audio.unsqueeze(0).to(args.device)  # (1, 1, T)

            with torch.no_grad():
                encoded = dac_model.encode(audio_tensor)
                if isinstance(encoded, (tuple, list)):
                    z, codes, *_ = encoded
                else:
                    z = encoded.z
                    codes = encoded.codes

                z_q = dac_model.quantizer.from_codes(codes)[0]
                recon = dac_model.decode(z_q)

            recon_np = recon.squeeze().cpu().numpy()

            # Align lengths
            min_len = min(len(audio_np), len(recon_np))
            snr = snr_db(audio_np[:min_len], recon_np[:min_len])

            tok_len = codes.shape[-1]
            print(f"  tokens: {tok_len} | {tok_len/args.duration:.1f} tok/s | SNR: {snr:.1f} dB")
            print(f"  original  range: [{audio_np.min():.3f}, {audio_np.max():.3f}]")
            print(f"  recon     range: [{recon_np.min():.3f}, {recon_np.max():.3f}]")

            results.append({"file": path.name, "snr_db": snr, "tokens": tok_len})

        except Exception as e:
            print(f"  ERROR: {e}")

    if results:
        snrs = [r["snr_db"] for r in results]
        print(f"\n{'='*50}")
        print(f"Results across {len(results)} files:")
        print(f"  Mean SNR: {np.mean(snrs):.1f} dB")
        print(f"  Min  SNR: {np.min(snrs):.1f} dB")
        print(f"  Max  SNR: {np.max(snrs):.1f} dB")
        print(f"  Tokens/sec (avg): {np.mean([r['tokens']/args.duration for r in results]):.1f}")
        print(f"\nNote: DAC is a lossy codec. >10dB SNR with music is typical.")


if __name__ == "__main__":
    main()
