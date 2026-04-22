"""
Stage 1 — parse_corpus.py
Reads all TextGrid files, metadata_RUFR.csv, and RUFRcorr.csv.
Outputs data/words.csv with one row per target word token.

Structure:
- 19 speakers, each reads 78 sentences (FRcorp1 to FRcorp78)
- 12 target words, each appears in 6 sentences (repetitions)
- 1 distractor word (ignored)
"""

import os
import re
import pandas as pd
import yaml

# ── load parameters ────────────────────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

CORPUS_PATH   = params["corpus"]["path"]
METADATA_PATH = os.path.join(CORPUS_PATH, "metadata_RUFR.csv")
RUFRCORR_PATH = os.path.join(CORPUS_PATH, "RUFRcorr.csv")
TEXTGRIDS_PATH = os.path.join(
    CORPUS_PATH, "wav_et_textgrids", "FRcorp_textgrids_only"
)
OUTPUT_PATH = "data/words.csv"

# ── load speaker metadata ──────────────────────────────────────────
metadata = pd.read_csv(METADATA_PATH, sep=";")
metadata.columns = metadata.columns.str.strip()
metadata["spk"] = metadata["spk"].str.strip().str.upper()
spk_info = metadata.set_index("spk").to_dict(orient="index")

# ── load word/sentence mapping ─────────────────────────────────────
# RUFRcorr.csv tells us which FRcorp numbers contain each target word
rufrcorr = pd.read_csv(RUFRCORR_PATH, sep="\t")
rufrcorr = rufrcorr.dropna(subset=["Word"])  # drop distractor row

# build a dict: frcorp_number -> (word, repetition_index)
# e.g. {13: ("tsarine", 1), 15: ("tsarine", 2), ...}
frcorp_to_word = {}
for _, row in rufrcorr.iterrows():
    word = row["Word"]
    for rep_idx, col in enumerate(["occ.1","occ.2","occ.3",
                                    "occ.4","occ.5","occ.6"], start=1):
        frcorp_num = int(row[col])
        frcorp_to_word[frcorp_num] = {
            "word":       word,
            "repetition": rep_idx
        }

# ── parse word boundaries from a TextGrid ─────────────────────────
def parse_words_tier(path):
    """
    Returns a list of dicts, one per non-empty word interval.
    Each dict has: word, onset, offset, duration_ms
    """
    with open(path, encoding="utf-8") as f:
        content = f.read()

    # extract the words tier block
    words_match = re.search(
        r'name = "words".*?intervals: size = \d+(.*?)(?=item \[\d+\]:|$)',
        content,
        re.DOTALL,
    )
    if not words_match:
        return []

    intervals = re.findall(
        r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "(.*?)"',
        words_match.group(1),
    )

    tokens = []
    for xmin, xmax, text in intervals:
        text = text.strip()
        if text == "":
            continue
        xmin, xmax = float(xmin), float(xmax)
        tokens.append({
            "word":        text,
            "onset":       xmin,
            "offset":      xmax,
            "duration_ms": round((xmax - xmin) * 1000, 2),
        })
    return tokens


# ── walk all speaker folders ───────────────────────────────────────
rows = []
missing_metadata = []

for spk_folder in sorted(os.listdir(TEXTGRIDS_PATH)):
    spk_id   = spk_folder.upper()
    spk_path = os.path.join(TEXTGRIDS_PATH, spk_folder)
    if not os.path.isdir(spk_path):
        continue

    info = spk_info.get(spk_id, None)
    if info is None:
        missing_metadata.append(spk_id)
        continue

    l1_status = "L1" if info["L1"] == "fr" else "L2"
    gender    = info["Gender"].strip()

    for filename in sorted(os.listdir(spk_path)):
        if not filename.endswith(".TextGrid"):
            continue

        # parse filename to get FRcorp number
        match = re.match(
            r"\w+_(fr|rus|fra)_list\d+_FRcorp(\d+)\.TextGrid",
            filename,
            re.IGNORECASE,
        )
        if not match:
            print(f"  WARNING: unexpected filename: {filename}")
            continue

        frcorp_num = int(match.group(2))

        # skip distractor sentences
        if frcorp_num not in frcorp_to_word:
            continue

        word_info = frcorp_to_word[frcorp_num]
        tg_path   = os.path.join(spk_path, filename)
        word_tokens = parse_words_tier(tg_path)

        # find the target word in this sentence
        target = word_info["word"]
        for tok in word_tokens:
            # match loosely (the TextGrid text may differ slightly)
            if tok["word"].lower() == target.lower():
                wav_file = os.path.join(
                    TEXTGRIDS_PATH, spk_folder,
                    filename.replace(".TextGrid", ".wav")
                )
                rows.append({
                    "speaker_id":  spk_id,
                    "l1_status":   l1_status,
                    "gender":      gender,
                    "word":        target,
                    "repetition":  word_info["repetition"],
                    "frcorp_num":  frcorp_num,
                    "onset":       tok["onset"],
                    "offset":      tok["offset"],
                    "duration_ms": tok["duration_ms"],
                    "wav_file":    wav_file,
                })

# ── save output ────────────────────────────────────────────────────
df = pd.DataFrame(rows)
os.makedirs("data", exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"Done. {len(df)} word tokens saved to {OUTPUT_PATH}")
print(f"Speakers:    {df['speaker_id'].nunique()}")
print(f"Words:       {df['word'].nunique()} — {sorted(df['word'].unique())}")
print(f"Repetitions: {df['repetition'].nunique()} per word per speaker")
if missing_metadata:
    print(f"WARNING: no metadata for speakers: {missing_metadata}")
print()
print(df.groupby(['l1_status','gender'])['speaker_id'].nunique())