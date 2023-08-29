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


if __name__ == "__main__":
    filtered_reddit_df = pd.read_csv(CONFIG["reddit_data_filtered_path"])

    # Phase 1 sampling
    negative_emotions = [
        emotion.strip() for emotion in CONFIG["negative_emotions"].split(",")
    ]
    filtered_reddit_df["negative_emotion_weight"] = normalized_mean(
        filtered_reddit_df, negative_emotions
    )
    sampling_func = partial(
        stratified_weighted_sampling,
        all_data=filtered_reddit_df,
        total_samples=int(CONFIG["n_samples_phase1"]),
        weight_col="negative_emotion_weight",
        random_state=int(CONFIG["random_seed"]),
    )
    sampled_reddit_df_phase1 = (
        filtered_reddit_df.groupby("subreddit")
        .apply(lambda x: sampling_func(group=x))
        .reset_index(drop=True)
    )

    # Phase 2 sampling
    sampled_reddit_df_phase1["dominant_negative_emotion"] = (
        sampled_reddit_df_phase1[["anger", "disgust", "fear", "sadness"]]
        .idxmax(axis=1)
        .apply(lambda x: "anger" if x == "disgust" else x)
    )
    inverse_freq_weight = (
        1.0 / sampled_reddit_df_phase1["dominant_negative_emotion"].value_counts()
    )
    sampled_reddit_df_final = filtered_reddit_df.sample(
        n=int(CONFIG["n_samples_final"]),
        weights=sampled_reddit_df_phase1["dominant_negative_emotion"],
        random_state=int(CONFIG["random_seed"]),
    )
    sampled_reddit_df_final.to_csv(CONFIG["reddit_data_sampled_path"])
