"""
Stage 2 — extract_features.py
Loads wav2vec2, extracts hidden states for each word token in words.csv.
Saves float32 representations to data/features_float32.npz

We extract once in float32. All other precision levels are derived
from this in the next stage (convert_precision.py).
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoProcessor, AutoModel
from scipy.io import wavfile

# ── load parameters ────────────────────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

MODEL_NAME   = params["model"]["name"]
LAYER_IDX    = params["model"]["layer"]
WORDS_CSV    = "data/words.csv"
OUTPUT_PATH  = "data/features_float32.npz"

# ── load model ─────────────────────────────────────────────────────
print(f"Loading model: {MODEL_NAME}")
print("This may take a minute on first run (downloading weights)...")

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True)
model.eval()  # disable dropout
print("Model loaded.\n")

# ── load word tokens ───────────────────────────────────────────────
df = pd.read_csv(WORDS_CSV)
print(f"Processing {len(df)} word tokens...")
print(f"Extracting hidden states from layer {LAYER_IDX}\n")

# ── helper: load and slice a wav segment ──────────────────────────
def load_wav_segment(wav_path, onset, offset):
    """
    Loads a WAV file and returns the segment between onset and offset (seconds).
    Returns a float32 numpy array at 16kHz (wav2vec2 requirement).
    """
    sample_rate, audio = wavfile.read(wav_path)

    # convert to float32 in [-1, 1]
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0

    # stereo → mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # slice the segment
    start = int(onset * sample_rate)
    end   = int(offset * sample_rate)
    segment = audio[start:end]

    # resample to 16kHz if needed
    if sample_rate != 16000:
        # simple resampling using numpy
        target_len = int(len(segment) * 16000 / sample_rate)
        segment = np.interp(
            np.linspace(0, len(segment), target_len),
            np.arange(len(segment)),
            segment
        )

    return segment.astype(np.float32)


# ── extract representations ────────────────────────────────────────
features  = []   # one vector per token
token_ids = []   # row index in words.csv

start_time = time.time()

for idx, row in df.iterrows():
    wav_path = row["wav_file"]

    if not os.path.exists(wav_path):
        print(f"  WARNING: file not found: {wav_path}")
        features.append(None)
        token_ids.append(idx)
        continue

    # load audio segment
    segment = load_wav_segment(wav_path, row["onset"], row["offset"])

    # tokenize for wav2vec2
    inputs = processor(
        segment,
        sampling_rate=16000,
        return_tensors="pt"
    )

    # forward pass (no gradient needed)
    with torch.no_grad():
        outputs = model(**inputs)

    # get hidden states from chosen layer → shape: (1, time_steps, hidden_dim)
    hidden = outputs.hidden_states[LAYER_IDX]  # (1, T, D)

    # mean pool over time → shape: (hidden_dim,)
    vector = hidden.squeeze(0).mean(dim=0).numpy()  # (D,)

    features.append(vector)
    token_ids.append(idx)

    # progress every 50 tokens
    if (idx + 1) % 50 == 0:
        elapsed = time.time() - start_time
        remaining = elapsed / (idx + 1) * (len(df) - idx - 1)
        print(f"  {idx+1}/{len(df)} tokens — "
              f"elapsed: {elapsed:.0f}s — "
              f"remaining: ~{remaining:.0f}s")

# ── handle missing files ───────────────────────────────────────────
valid_mask = [f is not None for f in features]
valid_features = [f for f in features if f is not None]
valid_ids = [i for i, v in zip(token_ids, valid_mask) if v]

print(f"\nExtracted {len(valid_features)}/{len(df)} tokens successfully.")

# ── save as float32 npz ────────────────────────────────────────────
X = np.stack(valid_features).astype(np.float32)  # (N, D)

np.savez(
    OUTPUT_PATH,
    features=X,
    token_ids=np.array(valid_ids),
)

elapsed = time.time() - start_time
print(f"Saved to {OUTPUT_PATH}  shape: {X.shape}")
print(f"Total time: {elapsed:.0f}s")