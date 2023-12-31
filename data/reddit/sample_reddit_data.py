"""Sample reddit data based on emotion of post."""
from typing import List
from functools import partial
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config import REDDIT_INSTANCE_GENERATION_CONFIG as CONFIG


def normalized_mean(df: pd.DataFrame, columns: List[str]) -> pd.Series:
    """Creates a normalized mean given columns"""
    factors = df[columns]
    scaler = MinMaxScaler()
    scaled_factors = scaler.fit_transform(factors)
    return scaled_factors.mean(axis=1)


def stratified_weighted_sampling(
    all_data: pd.DataFrame,
    group: pd.core.groupby.DataFrameGroupBy,
    total_samples: str,
    weight_col: str,
    random_state: int,
) -> pd.DataFrame:
    """Stratifies based on groups provided, and samples with weighting within the group"""
    group_size = len(group)
    group_proportion = group_size / len(all_data)
    sample_size = int(total_samples * group_proportion)
    return group.sample(n=sample_size, weights=weight_col, random_state=random_state)


def two_phase_reddit_sampling(
    reddit_df: pd.DataFrame,
    negative_emotions_str: str,
    n_samples_phase_1: int,
    n_samples_final: int,
    random_seed: int,
) -> pd.DataFrame:
    # Phase 1 sampling

    # Create a normalized "negative emotion" score
    negative_emotions = [
        emotion.strip() for emotion in negative_emotions_str.split(",")
    ]
    reddit_df["negative_emotion_weight"] = normalized_mean(reddit_df, negative_emotions)
    sampling_func = partial(
        stratified_weighted_sampling,
        all_data=reddit_df,
        total_samples=n_samples_phase_1,
        weight_col="negative_emotion_weight",
        random_state=random_seed,
    )
    # Stratified random sampling on subreddit, apply negative emotion weighting
    sampled_reddit_df_phase1 = (
        reddit_df.groupby("subreddit")
        .apply(lambda x: sampling_func(group=x))
        .reset_index(drop=True)
    )

    # Phase 2 sampling

    # Identify the "dominant" negative emotion, encompass disgust into anger.
    sampled_reddit_df_phase1["dominant_negative_emotion"] = (
        sampled_reddit_df_phase1[["anger", "disgust", "fear", "sadness"]]
        .idxmax(axis=1)
        .apply(lambda x: "anger" if x == "disgust" else x)
    )
    inverse_freq_weight = (
        1.0 / sampled_reddit_df_phase1["dominant_negative_emotion"].value_counts()
    ).to_dict()
    sampled_reddit_df_phase1[
        "dominant_negative_emotion_weight"
    ] = sampled_reddit_df_phase1["dominant_negative_emotion"].map(inverse_freq_weight)
    # Do weighted sampling to even out each dominant negative emotion
    sampled_reddit_df_final = sampled_reddit_df_phase1.sample(
        n=n_samples_final,
        weights="dominant_negative_emotion_weight",
        random_state=random_seed,
    )
    return sampled_reddit_df_final


if __name__ == "__main__":
    filtered_reddit_df = pd.read_csv(CONFIG["reddit_data_filtered_path"])
    sampled_reddit_df = two_phase_reddit_sampling(
        reddit_df=filtered_reddit_df,
        negative_emotions_str=CONFIG["negative_emotions"],
        n_samples_phase_1=int(CONFIG["n_samples_phase_1"]),
        n_samples_final=int(CONFIG["n_samples_final"]),
        random_seed=42,
    )
    sampled_reddit_df.to_csv(CONFIG["reddit_data_sampled_path"], index=False)
