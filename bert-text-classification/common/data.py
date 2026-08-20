"""
Data loading for the "Coronavirus tweets NLP - Text Classification" dataset
(https://www.kaggle.com/code/nayansakhiya/text-classification-using-bert/input)

Expected files (as provided by the Kaggle dataset):
    Corona_NLP_train.csv
    Corona_NLP_test.csv

Columns: UserName, ScreenName, Location, TweetAt, OriginalTweet, Sentiment
Sentiment has 5 classes:
    Extremely Negative, Negative, Neutral, Positive, Extremely Positive
"""
import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

LABELS_5CLASS = [
    "Extremely Negative",
    "Negative",
    "Neutral",
    "Positive",
    "Extremely Positive",
]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS_5CLASS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


def clean_tweet(text: str) -> str:
    """Light cleaning only. BERT's subword tokenizer handles most noise fine,
    so we deliberately avoid aggressive preprocessing (no stopword removal,
    no lowercasing/stemming) which tends to hurt transformer models."""
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", "<URL>", text)
    text = re.sub(r"@\w+", "<USER>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _read_csv(path: str) -> pd.DataFrame:
    # the original Kaggle CSVs are not UTF-8
    return pd.read_csv(path, encoding="latin-1")


def _resolve_csv(path: str) -> str:
    """ENV override so the same yaml works on Kaggle and locally:
    export TWEETS_TRAIN_CSV=... TWEETS_TEST_CSV=... (resolved once at load)."""
    env_name = "TWEETS_TRAIN_CSV" if "train" in os.path.basename(path).lower() else "TWEETS_TEST_CSV"
    override = os.environ.get(env_name)
    return override if override else path


def load_dataframes(train_csv: str, test_csv: str, val_size: float = 0.1, seed: int = 42):
    train_df = _read_csv(_resolve_csv(train_csv))
    test_df = _read_csv(_resolve_csv(test_csv))

    for df in (train_df, test_df):
        df["text"] = df["OriginalTweet"].apply(clean_tweet)
        df["label"] = df["Sentiment"].map(LABEL2ID)
        df.dropna(subset=["label"], inplace=True)
        df["label"] = df["label"].astype(int)

    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size,
        random_state=seed,
        stratify=train_df["label"],
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


class TweetDataset(Dataset):
    """Tokenizes on the fly; pair with transformers.DataCollatorWithPadding
    for dynamic padding (cheaper than padding everything to max_length)."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
        )
        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        item["idx"] = idx  # keep original row index for error analysis
        return item


def make_collate_fn(tokenizer):
    """DataCollatorWithPadding chokes on non-tokenizer fields (labels/idx),
    so we pad the tokenizer fields ourselves and stack the rest."""

    def collate(batch):
        idx = torch.tensor([b.pop("idx") for b in batch], dtype=torch.long)
        labels = torch.stack([b.pop("labels") for b in batch])
        padded = tokenizer.pad(batch, return_tensors="pt")
        padded["labels"] = labels
        padded["idx"] = idx
        return padded

    return collate
