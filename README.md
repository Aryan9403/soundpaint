# SoundPaint

Music generation using a Mamba language model trained on DAC audio tokens.

**Pipeline**: MP3s → DAC 44kHz codec (codebook 0) → Mamba LM → generated tokens → DAC decode → .wav

---

## Setup

```bash
pip install -r requirements.txt
```

> `mamba_ssm` requires CUDA. On a CPU-only machine it will fail to install — use Colab instead.

---

## Dataset

Not included in this repo. Put your `.mp3` files in `dataset/`.

---

## Usage

### 1. Tokenize

Encode MP3s into DAC tokens. Run on GPU for speed.

```bash
python data/prepare.py \
    --audio_dir dataset/ \
    --output_dir data/tokens_small \
    --device cuda \
    --duration 30
```

Produces one `.pt` file per MP3 in `data/tokens_small/`. Each file is a 1D `LongTensor` of ~2580 tokens (30s at 86 tok/sec).

### 2. Train

```bash
python train.py --config configs/tiny.yaml
```

- Checkpoints saved to `checkpoints/tiny/` every 500 steps
- Audio samples generated to `samples/tiny/` every 200 steps
- Resume with `--resume checkpoints/tiny/step_001000.pt`

### 3. Generate

```bash
python generate.py \
    --checkpoint checkpoints/tiny/best.pt \
    --duration 10 \
    --output output.wav
```

---

## Colab (Free T4)

Open `colab.ipynb`. Change runtime to **T4 GPU**, then run cells top to bottom:

1. Clone repo + install deps
2. Upload your dataset as a zip
3. Tokenize
4. Train (~30-60 min for 2000 steps)
5. Generate
6. Listen + download

---

## Configs

| Config | d_model | Layers | Batch | Steps | Target |
|---|---|---|---|---|---|
| `configs/tiny.yaml` | 256 | 4 | 4 | 2000 | Colab T4 |
| `configs/default.yaml` | 1024 | 24 | 32 | 100k | A100 |

---

## Smoke Tests

```bash
# Verify DAC loads and encodes one file
python scripts/test_dac.py

# Encode/decode 5 files, print SNR stats
python scripts/test_roundtrip.py
```

---

## Architecture

- **Codec**: [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec) 44kHz, codebook 0 only
- **Model**: Mamba SSM blocks (`mamba_ssm`), no attention
- **Vocab**: 1028 tokens (1024 DAC + PAD/BOS/EOS/UNK)
- **Sequence length**: 2580 tokens = 30 seconds
