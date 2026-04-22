# Numerical Stability in wav2vec Representations

M1 Computational Linguistics — Professional Skills in CL  
Université Paris — 2025–2026

## Project
This project investigates how reducing numerical precision (float64 → float32 → float16 → int8) affects the geometry of wav2vec2 speech representations, using the Russian–French Interference Corpus.

## Pipeline (DVC)
| Stage | Script | Output |
|-------|--------|--------|
| 1. Parse corpus | `src/parse_corpus.py` | `data/words.csv` |
| 2. Extract features | `src/extract_features.py` | `data/features_float64.npz` |
| 3. Convert precision | `src/convert_precision.py` | `data/features_*.npz` |
| 4. Compute distances | `src/compute_distances.py` | `results/distances.csv` |
| 5. Visualise | `src/visualise.py` | `results/plots` |

## Reproduce
```bash
pip install -r requirements.txt
dvc repro
```

## Corpus
Russian–French Interference Corpus, ORTOLANG:
https://www.ortolang.fr/market/corpora/ru-fr_interference