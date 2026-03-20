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

> **Runtime**: Runtime → Change runtime type → **T4 GPU**

### Step 1 — Clone & install

```python
!git clone https://github.com/Aryan9403/soundpaint.git
%cd soundpaint
!pip install -q torch descript-audio-codec librosa soundfile numpy pyyaml tqdm
!pip install -q git+https://github.com/descriptinc/audiotools
!pip install -q mamba_ssm causal-conv1d
```

### Step 2 — Upload dataset

Zip your `dataset/` folder locally, then upload it:

```python
from google.colab import files
import zipfile

uploaded = files.upload()          # select dataset.zip
with zipfile.ZipFile('dataset.zip', 'r') as z:
    z.extractall('.')
```

Or use the notebook's Cell 2 which does this automatically.

### Step 3 — Tokenize

```python
!python data/prepare.py \
    --audio_dir dataset/ \
    --output_dir data/tokens_small \
    --device cuda \
    --duration 30
```

Takes ~2-5 minutes for 240 files on T4.

### Step 4 — Train

```python
!python train.py --config configs/tiny.yaml
```

- ~30-60 min for 2000 steps on T4
- Val loss printed every 200 steps
- A 10s audio sample is auto-generated and saved to `samples/tiny/` every 200 steps — no need to run generate separately during training
- Checkpoints saved to `checkpoints/tiny/` every 500 steps
- To resume after a Colab disconnect: `!python train.py --config configs/tiny.yaml --resume checkpoints/tiny/step_001000.pt`

### Step 5 — Generate audio

After training, generate a clip from scratch:

```python
!python generate.py \
    --checkpoint checkpoints/tiny/best.pt \
    --duration 10 \
    --output output.wav \
    --temperature 0.95 \
    --top_k 250
```

Options:
- `--duration` — length of clip in seconds
- `--temperature` — higher = more random (try 0.8–1.2)
- `--top_k` — sampling pool size (try 100–500)

### Step 6 — Listen & download

```python
from IPython.display import Audio, display
from google.colab import files

display(Audio('output.wav'))
files.download('output.wav')
```

You can also listen to the mid-training samples:

```python
import glob
samples = sorted(glob.glob('samples/tiny/*.wav'))
display(Audio(samples[-1]))   # most recent sample
```

### Saving checkpoints before Colab disconnects

Download your checkpoint so you don't lose progress:

```python
from google.colab import files
files.download('checkpoints/tiny/best.pt')
```

To resume from a downloaded checkpoint next session, re-upload it:

```python
from google.colab import files
uploaded = files.upload()   # select best.pt
!mkdir -p checkpoints/tiny
!mv best.pt checkpoints/tiny/best.pt
!python train.py --config configs/tiny.yaml --resume checkpoints/tiny/best.pt
```

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
