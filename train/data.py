"""Load the unified training table produced by stage 07."""
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset

REQUIRED = {"id", "study_id", "task", "label", "text", "origin"}


def load_instances(path, section_classes):
    df = pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}. Run python run_dataset.py first.")

    unknown_tasks = sorted(set(df["task"]) - set(section_classes))
    if unknown_tasks:
        raise ValueError(f"unknown tasks: {unknown_tasks}")

    for task, labels in section_classes.items():
        found = set(df.loc[df["task"] == task, "label"])
        extra = sorted(found - set(labels))
        if extra:
            raise ValueError(f"{task} has unknown labels: {extra}")

    print(f"Loaded {len(df)} instances from {df['study_id'].nunique()} studies")
    return df


def label_maps(section_classes):
    label2id = {
        task: {label: i for i, label in enumerate(labels)}
        for task, labels in section_classes.items()
    }
    id2label = {
        task: {i: label for label, i in mapping.items()}
        for task, mapping in label2id.items()
    }
    return label2id, id2label


def class_weights(df, task, label2id, device):
    subset = df[df["task"] == task]
    counts = Counter(subset["label"])
    n_classes = len(label2id[task])
    weights = torch.ones(n_classes, device=device)
    total = len(subset) or 1
    for label, index in label2id[task].items():
        freq = counts.get(label, 0)
        weights[index] = total / (n_classes * max(freq, 1))
    return weights


class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, label2id, max_length=256):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        task = row["task"]
        encoded = self.tokenizer(
            str(row["text"]),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.label2id[task][row["label"]], dtype=torch.long),
            "task": task,
            "label_name": row["label"],
            "instance_id": str(row["id"]),
            "study_id": str(row["study_id"]),
            "origin": str(row.get("origin", "original")),
        }


def iter_folds(df, folds_path):
    folds = pd.read_csv(folds_path)
    folds = folds[folds["split"] == "test"]
    for fold in sorted(folds["fold"].unique()):
        test_ids = set(folds.loc[folds["fold"] == fold, "study_id"].astype(int))
        is_test = df["study_id"].astype(int).isin(test_ids)
        yield int(fold), df[~is_test], df[is_test]
