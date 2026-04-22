"""
Stage 4 — compute_distances.py
For each precision level, computes:
  - intra-speaker distance: same speaker, same word, different repetitions
  - inter-speaker distance: different speakers, same word

Uses cosine distance throughout.
float64 is the reference — we check whether lower precisions
change the relative ordering and magnitude of these distances.
"""

import os
import numpy as np
import pandas as pd
import yaml
from itertools import combinations

# ── load parameters ────────────────────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

WORDS_CSV  = "data/words.csv"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── cosine distance ────────────────────────────────────────────────
def cosine_distance(a, b):
    """
    Cosine distance = 1 - cosine_similarity.
    Works for any float dtype — we cast to float64 for the computation
    only when we want the reference; otherwise we keep the native dtype.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return np.nan
    return 1.0 - np.dot(a, b) / (norm_a * norm_b)


# ── load word tokens metadata ──────────────────────────────────────
df = pd.read_csv(WORDS_CSV)
df = df.reset_index(drop=True)

# ── process each precision level ───────────────────────────────────
precision_files = {
    "float64": "data/features_float64.npz",
    "float32": "data/features_float32.npz",
    "float16": "data/features_float16.npz",
    "int8":    "data/features_int8.npz",
}

all_results = []   # one row per (precision, word, speaker_pair_type)

for precision, path in precision_files.items():
    print(f"\nProcessing {precision}...")

    # load features
    data = np.load(path)
    X    = data["features"]

    # reconstruct int8 to float for distance computation
    if precision == "int8":
        scales = data["scales"]
        X = X.astype(np.float64) * scales[:, np.newaxis]

    # attach features to dataframe
    df["vec"] = list(X)

    # ── intra-speaker distances ────────────────────────────────────
    # same speaker, same word, different repetitions
    intra_distances = []

    for (spk, word), group in df.groupby(["speaker_id", "word"]):
        vecs = list(group["vec"])
        if len(vecs) < 2:
            continue
        # all pairs of repetitions
        for v1, v2 in combinations(vecs, 2):
            d = cosine_distance(v1, v2)
            if not np.isnan(d):
                intra_distances.append({
                    "precision":  precision,
                    "word":       word,
                    "speaker_id": spk,
                    "distance":   d,
                    "type":       "intra",
                })

    # ── inter-speaker distances ────────────────────────────────────
    # different speakers, same word
    # use per-speaker mean vector to avoid explosion of pairs
    inter_distances = []

    for word, group in df.groupby("word"):
        # compute mean vector per speaker for this word
        spk_means = {}
        for spk, spk_group in group.groupby("speaker_id"):
            vecs = np.stack(list(spk_group["vec"]))
            spk_means[spk] = vecs.mean(axis=0)

        # all pairs of speakers
        speakers = list(spk_means.keys())
        for spk1, spk2 in combinations(speakers, 2):
            d = cosine_distance(spk_means[spk1], spk_means[spk2])
            if not np.isnan(d):
                inter_distances.append({
                    "precision": precision,
                    "word":      word,
                    "spk1":      spk1,
                    "spk2":      spk2,
                    "distance":  d,
                    "type":      "inter",
                })

    all_results.extend(intra_distances)
    all_results.extend(inter_distances)

    # ── summary for this precision ─────────────────────────────────
    intra_vals = [r["distance"] for r in intra_distances]
    inter_vals = [r["distance"] for r in inter_distances]
    ratio      = np.mean(inter_vals) / np.mean(intra_vals)

    print(f"  intra-speaker: mean={np.mean(intra_vals):.4f}  "
          f"std={np.std(intra_vals):.4f}")
    print(f"  inter-speaker: mean={np.mean(inter_vals):.4f}  "
          f"std={np.std(inter_vals):.4f}")
    print(f"  ratio (inter/intra): {ratio:.4f}")

# ── save all distances ─────────────────────────────────────────────
results_df = pd.DataFrame(all_results)
results_df.to_csv(f"{OUTPUT_DIR}/distances.csv", index=False)
print(f"\nSaved {len(results_df)} distance values to {OUTPUT_DIR}/distances.csv")

# ── summary table across precisions ───────────────────────────────
print("\n── Summary table ──────────────────────────────────────────")
print(f"{'Precision':<10} {'Intra mean':>12} {'Inter mean':>12} "
      f"{'Ratio':>8} {'Ordering OK':>12}")

# reference values (float64)
ref = results_df[results_df["precision"] == "float64"]
ref_intra = ref[ref["type"] == "intra"]["distance"].mean()
ref_inter = ref[ref["type"] == "inter"]["distance"].mean()

for precision in ["float64", "float32", "float16", "int8"]:
    sub      = results_df[results_df["precision"] == precision]
    intra    = sub[sub["type"] == "intra"]["distance"].mean()
    inter    = sub[sub["type"] == "inter"]["distance"].mean()
    ratio    = inter / intra
    # ordering is preserved if inter > intra (as in reference)
    ordering = "✓" if inter > intra else "✗"
    print(f"{precision:<10} {intra:>12.4f} {inter:>12.4f} "
          f"{ratio:>8.4f} {ordering:>12}")