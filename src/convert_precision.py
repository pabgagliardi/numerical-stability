"""
Stage 3 — convert_precision.py
Takes float64 representations (reference) and converts downward to:
  - float32
  - float16
  - int8 (with per-vector linear quantisation)

float64 is the reference — all precision loss is measured relative to it.
"""

import os
import numpy as np
import yaml

# ── load parameters ────────────────────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

INPUT_PATH = "data/features_float64.npz"

# ── load float64 reference ─────────────────────────────────────────
data      = np.load(INPUT_PATH)
X_ref     = data["features"]      # shape: (N, 768), dtype float64
token_ids = data["token_ids"]

print(f"Reference features: shape={X_ref.shape}, dtype={X_ref.dtype}")
print()

# ── int8 quantisation ──────────────────────────────────────────────
def to_int8(X):
    """
    Linear quantisation to int8, per vector.
    Each vector is independently scaled so its range maps to [-127, 127].

    scale_i = max(|x_i|) / 127
    x_int8  = round(x_i / scale_i)

    We store scale factors separately so distances can be computed
    after approximate reconstruction: x_reconstructed = x_int8 * scale
    """
    X_int8 = np.zeros_like(X, dtype=np.int8)
    scales = np.zeros(X.shape[0], dtype=np.float64)

    for i in range(X.shape[0]):
        max_val = np.max(np.abs(X[i]))
        scales[i] = max_val / 127.0 if max_val > 0 else 1.0
        X_int8[i] = np.clip(
            np.round(X[i] / scales[i]), -127, 127
        ).astype(np.int8)

    return X_int8, scales


# ── convert and save ───────────────────────────────────────────────
os.makedirs("data", exist_ok=True)

# float32
X_f32    = X_ref.astype(np.float32)
path_f32 = "data/features_float32.npz"
np.savez(path_f32, features=X_f32, token_ids=token_ids)

# float16
X_f16    = X_ref.astype(np.float16)
path_f16 = "data/features_float16.npz"
np.savez(path_f16, features=X_f16, token_ids=token_ids)

# int8
X_i8, scales = to_int8(X_ref)
path_i8      = "data/features_int8.npz"
np.savez(path_i8, features=X_i8, scales=scales, token_ids=token_ids)

# ── report sizes ───────────────────────────────────────────────────
print("File sizes (memory trade-off):")
for label, path in [
    ("float64 (reference)", INPUT_PATH),
    ("float32",             path_f32),
    ("float16",             path_f16),
    ("int8",                path_i8),
]:
    size_mb = os.path.getsize(path) / 1e6
    print(f"  {label:22s} {size_mb:.1f} MB")

# ── sanity check: how much precision is lost? ──────────────────────
print()
print("Precision loss relative to float64:")

for label, X_conv in [
    ("float32", X_f32.astype(np.float64)),
    ("float16", X_f16.astype(np.float64)),
    ("int8",    (X_i8.astype(np.float64) * scales[:, np.newaxis])),
]:
    max_err  = np.max(np.abs(X_conv - X_ref))
    mean_err = np.mean(np.abs(X_conv - X_ref))
    print(f"  {label:8s}  max_error={max_err:.2e}  mean_error={mean_err:.2e}")

print()
print("Done.")