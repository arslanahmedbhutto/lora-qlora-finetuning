"""Emotions dataset (6 classes) loading and stratified subsampling."""

from pathlib import Path

import pandas as pd

from peft_study import DATA_DIR, SEED

LABELS = ["anger", "fear", "joy", "love", "sadness", "surprise"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}


def load_split(split: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    texts, labels = [], []
    with open(Path(data_dir) / f"{split}.txt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if ";" in line:
                text, label = line.rsplit(";", 1)
                texts.append(text.strip())
                labels.append(LABEL2ID[label.strip()])
    return pd.DataFrame({"text": texts, "label": labels})


def stratified_sample(df: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    """Keep the class proportions of `df` in a sample of about `n` rows."""
    if n >= len(df):
        return df.reset_index(drop=True)
    frac = n / len(df)
    out = df.groupby("label", group_keys=False).sample(frac=frac, random_state=seed)
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)
