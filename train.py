"""
Train MusicMambaLM on tokenized audio.

Usage:
    python train.py --config configs/tiny.yaml
    python train.py --config configs/tiny.yaml --resume checkpoints/tiny/step_500.pt
"""

import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from data.dataloader import get_dataloaders
from model.lm import MusicLM
from model.utils import count_parameters, load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def get_scheduler(optimizer, config, num_training_steps: int):
    """Linear warmup + cosine decay."""
    warmup = config.get("warmup_steps", 100)

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, num_training_steps - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, val_loader, device, pad_token_id, use_fp16):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for input_ids, target_ids in val_loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        with autocast(enabled=use_fp16):
            logits = model(input_ids)  # (B, T, vocab_size)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            ignore_index=pad_token_id,
            reduction="sum",
        )
        mask = target_ids != pad_token_id
        total_loss += loss.item()
        total_tokens += mask.sum().item()

    model.train()
    return total_loss / max(1, total_tokens)


def generate_sample(model, config, device, duration_sec: float) -> torch.Tensor:
    """Generate audio tokens for ~duration_sec seconds."""
    tokens_per_sec = 86  # DAC 44kHz
    max_new = int(duration_sec * tokens_per_sec)
    bos = config["bos_token_id"]
    eos = config["eos_token_id"]

    prompt = torch.tensor([[bos]], dtype=torch.long, device=device)
    generated = model.generate(
        prompt,
        max_new_tokens=max_new,
        temperature=0.95,
        top_k=250,
        eos_token_id=eos,
    )
    return generated[0]  # (T,)


def decode_and_save(tokens: torch.Tensor, out_path: str, device: str):
    """Decode codebook-0 tokens → waveform → .wav file."""
    import dac
    import soundfile as sf
    import numpy as np

    dac_model = dac.DAC.load(dac.utils.download(model_type="44khz"))
    dac_model = dac_model.to(device)
    dac_model.eval()

    n_codebooks = dac_model.quantizer.n_codebooks
    L = tokens.shape[0]

    # Build full codebook tensor: codebook 0 = generated, rest = 0 (silence)
    codes = torch.zeros(1, n_codebooks, L, dtype=torch.long, device=device)
    codes[0, 0, :] = tokens.to(device)

    with torch.no_grad():
        z = dac_model.quantizer.from_codes(codes)[0]   # (1, d_model, L)
        audio = dac_model.decode(z)                     # (1, 1, T)

    audio_np = audio.squeeze().cpu().numpy()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, audio_np, 44100)
    print(f"  Saved sample: {out_path}")


def save_checkpoint(path, model, optimizer, step, config):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }, path)


def main():
    args = parse_args()
    config = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = config.get("fp16", True) and device == "cuda"
    print(f"Device: {device} | FP16: {use_fp16}")

    # Data
    train_loader, val_loader = get_dataloaders(config)

    # Model
    model = MusicLM(config).to(device)
    n_params = count_parameters(model)
    print(f"Model params: {n_params:,} ({n_params/1e6:.1f}M)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.1),
        betas=(0.9, 0.95),
    )

    scaler = GradScaler(enabled=use_fp16)
    scheduler = get_scheduler(optimizer, config, config["max_steps"])

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"]
        print(f"Resumed from step {start_step}")

    pad_token_id = config["pad_token_id"]
    eval_every = config.get("eval_every", 200)
    save_every = config.get("save_every", 500)
    max_steps = config["max_steps"]
    grad_clip = config.get("grad_clip", 1.0)
    ckpt_dir = config.get("checkpoint_dir", "checkpoints/run")
    gen_dir = config.get("generate_dir", "samples/run")
    gen_duration = config.get("generate_duration", 10)

    model.train()
    step = start_step
    train_iter = iter(train_loader)

    pbar = tqdm(total=max_steps, initial=start_step, desc="Training")

    while step < max_steps:
        # Get batch (cycle through loader)
        try:
            input_ids, target_ids = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            input_ids, target_ids = next(train_iter)

        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        optimizer.zero_grad()

        with autocast(enabled=use_fp16):
            logits = model(input_ids)  # (B, T, vocab_size)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                ignore_index=pad_token_id,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        step += 1
        pbar.update(1)
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        if step % eval_every == 0:
            val_loss = evaluate(model, val_loader, device, pad_token_id, use_fp16)
            print(f"\nStep {step} | val_loss: {val_loss:.4f}")

            # Generate sample
            tokens = generate_sample(model, config, device, gen_duration)
            out_wav = os.path.join(gen_dir, f"step_{step:06d}.wav")
            decode_and_save(tokens, out_wav, device)

        if step % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"step_{step:06d}.pt")
            save_checkpoint(ckpt_path, model, optimizer, step, config)
            print(f"  Checkpoint saved: {ckpt_path}")

    # Final checkpoint
    final_path = os.path.join(ckpt_dir, "best.pt")
    save_checkpoint(final_path, model, optimizer, step, config)
    print(f"\nTraining complete. Final checkpoint: {final_path}")
    pbar.close()


if __name__ == "__main__":
    main()
