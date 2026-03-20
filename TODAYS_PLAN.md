# Today's Plan: Get It Working on Colab (Free T4)

## Goal
End of today: a working end-to-end pipeline that tokenizes audio, trains a tiny model, and generates audio. All on free Colab. Zero dollars spent.

## Setup
- **Local machine:** Write all code, push to GitHub
- **Colab:** Clone repo, upload small dataset, run training
- **Dataset:** ~100-200 tracks from FMA Small (you pick, keep it tiny for speed)
- **Model:** Tiny config (d_model=256, n_layers=4, ~5M params) — just to prove the pipeline works

---

## Step 1: Project Scaffold (Local)
Write these files locally, push to GitHub:

```
music-gen/
├── configs/
│   ├── default.yaml        ← full FMA Medium config (for later)
│   └── tiny.yaml           ← tiny config for Colab debugging TODAY
├── data/
│   ├── prepare.py          ← tokenize audio → .pt files
│   ├── dataset.py          ← PyTorch Dataset
│   └── dataloader.py       ← train/val DataLoader factory
├── model/
│   ├── __init__.py
│   ├── mamba_lm.py         ← Mamba language model
│   ├── embedding.py        ← token + position embeddings
│   └── utils.py            ← param counter, helpers
├── train.py                ← training loop
├── generate.py             ← generate audio from trained model
├── scripts/
│   ├── test_dac.py         ← DAC sanity check
│   └── test_roundtrip.py   ← codebook quality ladder
├── requirements.txt
├── colab.ipynb             ← ONE notebook that runs everything
└── README.md
```

## Step 2: Tiny Config for Today

```yaml
# configs/tiny.yaml — FOR DEBUGGING ONLY
audio_dir: "data/fma_small_subset"
token_dir: "data/tokens_small"
sample_rate: 44100
track_duration_sec: 30
dac_model_type: "44khz"

n_codebooks: 9
codebook_size: 1024
tokens_per_second: 86

# Tiny model — trains fast, proves pipeline works
vocab_size: 1028
d_model: 256
n_layers: 4
seq_len: 2580

pad_token_id: 1024
bos_token_id: 1025
eos_token_id: 1026
sep_token_id: 1027

batch_size: 4
learning_rate: 3.0e-4
weight_decay: 0.01
warmup_steps: 100
max_steps: 2000             # short run, ~30 min on T4
grad_clip: 1.0
mixed_precision: true       # fp16 on T4
eval_every: 500
save_every: 500
log_every: 10
num_workers: 2

temperature: 0.95
top_k: 250
top_p: 0.0
max_gen_tokens: 860         # generate 10s for quick test
```

## Step 3: Order of Work (write code in this order)

### 3a. data/prepare.py
- Takes a directory of .mp3 files
- Loads each file with librosa/audiotools, resamples to 44100Hz mono
- Encodes with pretrained DAC → extracts first codebook only
- Saves each track as a .pt file (1D LongTensor of ~2580 ints)
- Handles corrupted files gracefully (skip + log)

### 3b. data/dataset.py
- PyTorch Dataset class
- Loads .pt files
- Returns (input_ids, target_ids) shifted by 1 for next-token prediction

### 3c. data/dataloader.py
- Creates train/val split (95/5)
- Returns DataLoaders with shuffle, pin_memory, num_workers

### 3d. model/embedding.py
- Token embedding: nn.Embedding(vocab_size, d_model)
- Positional embedding: nn.Embedding(max_seq_len, d_model)
- Forward: token_emb + pos_emb

### 3e. model/mamba_lm.py
- MusicMambaLM class
- Stack of Mamba blocks from mamba_ssm
- Forward: input_ids → logits
- generate(): autoregressive sampling

### 3f. model/utils.py
- count_parameters()
- load_config()

### 3g. train.py
- Load config
- Build model, optimizer (AdamW), scheduler (warmup + cosine)
- Training loop: forward → loss → backward → step
- Validation every N steps
- Checkpointing
- Generate sample audio every eval_every steps

### 3h. generate.py
- Load checkpoint
- Load DAC decoder
- Generate tokens → decode to audio → save .wav

## Step 4: Colab Notebook (colab.ipynb)

The notebook should be dead simple — just clone, install, and run:

```python
# Cell 1: Setup
!git clone https://github.com/YOUR_USERNAME/music-gen.git
%cd music-gen
!pip install -q -r requirements.txt

# Cell 2: Upload your small FMA subset
# (drag and drop a zip of ~100-200 .mp3 files, or wget from gdrive)
!mkdir -p data/fma_small_subset
!unzip /content/fma_subset.zip -d data/fma_small_subset/

# Cell 3: Tokenize
!python data/prepare.py \
    --audio_dir data/fma_small_subset \
    --output_dir data/tokens_small \
    --device cuda

# Cell 4: Train (tiny model, ~30 min)
!python train.py --config configs/tiny.yaml

# Cell 5: Generate
!python generate.py \
    --checkpoint checkpoints/tiny/best.pt \
    --duration 10 \
    --output output.wav

# Cell 6: Listen
from IPython.display import Audio
Audio("output.wav")
```

## Step 5: What "Working" Means Today

At the end of today, ALL of these must be true:
- [ ] prepare.py tokenizes audio files into .pt tokens
- [ ] Dataset/DataLoader loads tokens and yields correct shapes
- [ ] Model forward pass works (input → logits, correct shapes)
- [ ] Loss decreases over 2000 steps (model is learning SOMETHING)
- [ ] generate.py produces a .wav file you can play
- [ ] The .wav sounds like noise/garbage (THAT'S OK — tiny model, tiny data)
- [ ] The full pipeline runs without crashes on Colab T4

## What We're NOT Doing Today
- No worrying about audio quality (that needs real data + real model size)
- No FMA Medium (save for GCP)
- No hyperparameter tuning
- No evaluation metrics
- No second-stage model

## After Today Works → GCP Run
1. Push working code to GitHub
2. SSH into GCP A100
3. Clone repo
4. Download FMA Medium
5. Tokenize with prepare.py (2 hrs)
6. Switch to configs/default.yaml (full 300M model)
7. Train for real (30-35 hrs)
8. Generate and evaluate
