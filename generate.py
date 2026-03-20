"""
Generate audio from a trained MusicMambaLM checkpoint.

Usage:
    python generate.py --checkpoint checkpoints/tiny/best.pt --duration 10 --output output.wav
    python generate.py --checkpoint checkpoints/tiny/best.pt --duration 30 --temperature 1.0 --top_k 500
"""

import argparse
from pathlib import Path

import torch
import soundfile as sf

from model.mamba_lm import MusicMambaLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration in seconds")
    parser.add_argument("--output", type=str, default="output.wav", help="Output .wav path")
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=250)
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu/cuda)")
    return parser.parse_args()


def decode_tokens(tokens: torch.Tensor, dac_model, device: str) -> torch.Tensor:
    """
    Decode codebook-0 tokens to audio waveform.

    Args:
        tokens: (L,) LongTensor of codebook-0 token indices
        dac_model: loaded DAC model
        device: 'cpu' or 'cuda'

    Returns:
        audio: (T,) float32 numpy array
    """
    n_codebooks = dac_model.quantizer.n_codebooks
    L = tokens.shape[0]

    # Pad missing codebooks with zeros (silence)
    codes = torch.zeros(1, n_codebooks, L, dtype=torch.long, device=device)
    codes[0, 0, :] = tokens.to(device)

    with torch.no_grad():
        z = dac_model.quantizer.from_codes(codes)[0]  # (1, d_model, L)
        audio = dac_model.decode(z)                    # (1, 1, T_audio)

    return audio.squeeze().cpu()  # (T_audio,)


def main():
    args = parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    print(f"Config: {config['name']} | d_model={config['d_model']} | n_layers={config['n_layers']}")

    # Build model and load weights
    model = MusicMambaLM(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded from step {ckpt.get('step', '?')}")

    # Generate tokens
    tokens_per_sec = 86  # DAC 44kHz
    max_new = int(args.duration * tokens_per_sec)
    bos = config["bos_token_id"]
    eos = config["eos_token_id"]

    prompt = torch.tensor([[bos]], dtype=torch.long, device=device)
    print(f"Generating {args.duration}s ({max_new} tokens)...")

    generated = model.generate(
        prompt,
        max_new_tokens=max_new,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=eos,
    )
    tokens = generated[0, 1:]  # strip BOS
    print(f"Generated {tokens.shape[0]} tokens")

    # Load DAC and decode
    print("Loading DAC 44kHz decoder...")
    import dac
    dac_model = dac.DAC.load(dac.utils.download(model_type="44khz"))
    dac_model = dac_model.to(device)
    dac_model.eval()

    audio = decode_tokens(tokens, dac_model, device)  # (T,)
    audio_np = audio.numpy()

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio_np, 44100)
    print(f"Saved: {out_path} ({len(audio_np)/44100:.1f}s, {len(audio_np)} samples)")


if __name__ == "__main__":
    main()
