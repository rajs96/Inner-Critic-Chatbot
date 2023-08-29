"""Load and filter reddit data, merge emotions predictions."""
from typing import List, Callable
import os
from os.path import join
from functools import partial
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import REDDIT_INSTANCE_GENERATION_CONFIG as CONFIG
from utils.data import TextDataset


def load_all_reddit_data(reddit_data_path: str) -> pd.DataFrame:
    """Assuming you have reddit csvs, loads them all into a pandas dataframe"""
    paths = [
        join(reddit_data_path, file)
        for file in tqdm(os.listdir(reddit_data_path))
        if "post" in file
    ]
    full_reddit_df = pd.DataFrame()
    for path in tqdm(paths):
        reddit_df = pd.read_csv(path)[["subreddit", "author", "date", "post"]]
        full_reddit_df = pd.concat([full_reddit_df, reddit_df])

    full_reddit_df.reset_index(inplace=True)
    return full_reddit_df


def filter_subreddits(
    full_reddit_df: pd.DataFrame, subreddits: List[str]
) -> pd.DataFrame:
    """Filter to only include supplied subreddits."""
    filtered_reddit_df = (
        full_reddit_df[full_reddit_df["subreddit"].isin(subreddits)]
        .reset_index()
        .drop(columns=["level_0", "index"])
    )
    return filtered_reddit_df


def filter_max_length(full_reddit_df: pd.DataFrame, max_length: int) -> pd.DataFrame:
    """Filter out posts that are higher than max_length"""
    filtered_reddit_df = full_reddit_df[
        full_reddit_df["post"].str.len() <= max_length
    ].reset_index(drop=True)
    return filtered_reddit_df


def load_emotion_tokenizer_func(model_name: str) -> Callable[[List[str]], Tensor]:
    """Load pretrained huggingface tokenizer"""
    emotion_tokenizer = AutoTokenizer.from_pretrained(model_name)
    emotion_tokenizer_func = partial(
        emotion_tokenizer.__call__,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )
    return emotion_tokenizer_func


def load_emotion_classifier(
    model_name: str, device: torch.device
) -> AutoModelForSequenceClassification:
    """Load pretrained huggingface model"""
    emotion_classifier = AutoModelForSequenceClassification.from_pretrained(
        model_name
    ).to(device)
    return emotion_classifier


def batch_predict(
    model: AutoModelForSequenceClassification,
    dataloader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Do batch probability predictions for a huggingface sequence classification model."""
    probs_batches = []
    model.eval()
    for batch in tqdm(dataloader):
        input_ids_batch, attention_mask_batch = batch
        input_ids_batch = input_ids_batch.to(device)
        attention_mask_batch = attention_mask_batch.to(device)
        logits = model(
            input_ids=input_ids_batch, attention_mask=attention_mask_batch
        ).logits
        probs = F.softmax(logits, dim=1)
        probs_batches.append(probs.detach().cpu().numpy())

    all_probs = np.concatenate(probs_batches, axis=0)
    return all_probs


def predict_and_merge_reddit_posts(
    model: AutoModelForSequenceClassification,
    tokenizer_func: Callable[[List[str]], Tensor],
    reddit_df: pd.DataFrame,
    labels: List[str],
    device: torch.device,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Make emotion classifier preds, merge with reddit dataframe"""
    subreddit_posts = reddit_df["post"].tolist()
    subreddit_text_dataset = TextDataset(subreddit_posts, tokenizer_func)
    subreddit_text_dataloader = DataLoader(
        subreddit_text_dataset,
        batch_size=batch_size,
        collate_fn=subreddit_text_dataset.collate_fn,
    )
    probs = batch_predict(model, subreddit_text_dataloader, device)
    probs_df = pd.DataFrame(probs, columns=labels)
    reddit_with_probs = pd.concat([reddit_df, probs_df], axis=1)
    return reddit_with_probs


if __name__ == "__main__":
    print("Loading reddit data...")
    full_reddit_df = load_all_reddit_data(CONFIG["reddit_data_folder_path"])
    print("Filtering by length and subreddit...")
    filtered_reddit_df_by_length = filter_max_length(
        full_reddit_df, int(CONFIG["max_post_length"])
    )
    relevant_subreddits = [
        subreddit.strip() for subreddit in CONFIG["relevant_subreddits"].split(",")
    ]
    filtered_reddit_df = filter_subreddits(
        filtered_reddit_df_by_length, relevant_subreddits
    )
    print("Loading emotion tokenizer and classifier...")
    emotion_tokenizer_func = load_emotion_tokenizer_func(
        CONFIG["huggingface_emotion_model"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emotion_classifier = load_emotion_classifier(
        CONFIG["huggingface_emotion_model"], device
    )
    labels = list(emotion_classifier.config.id2label.values())
    print("Making emotion predictions...")
    df = predict_and_merge_reddit_posts(
        model=emotion_classifier,
        tokenizer_func=emotion_tokenizer_func,
        reddit_df=filtered_reddit_df,
        labels=labels,
        device=device,
        batch_size=32,
    )
    print("Saving filtered reddit data...")
    df.to_csv(CONFIG["reddit_data_filtered_path"])
