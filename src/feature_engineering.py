import os
import pandas as pd
import numpy as np


def load_data(path="data/processed/ball_by_ball.csv"):
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded shape: {df.shape}")
    return df


def clean_data(df):
    before = len(df)

    required_cols = [
        "balls_remaining", "runs_remaining", "wickets_remaining",
        "current_run_rate", "required_run_rate", "target",
        "wickets_fallen", "runs_scored", "balls_bowled", "label"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=required_cols)
    df = df[df["balls_remaining"] >= 0]
    df = df[df["target"] > 0]
    df = df[df["wickets_fallen"] <= 10]
    df = df[df["wickets_remaining"] >= 0]
    df = df[df["runs_scored"] >= 0]
    df = df[df["label"].isin([0, 1])]

    print(f"Rows before cleaning : {before:,}")
    print(f"Rows after cleaning  : {len(df):,}")
    return df


def engineer_features(df):
    df["run_rate_diff"] = df["current_run_rate"] - df["required_run_rate"]

    df["balls_remaining_ratio"] = df["balls_remaining"] / df.groupby("match_id")["balls_bowled"].transform("max").clip(lower=1)
    df["balls_remaining_ratio"] = df["balls_remaining_ratio"].clip(0, 1)

    df["wickets_remaining_ratio"] = df["wickets_remaining"] / 10.0

    df["runs_per_ball_required"] = df["runs_remaining"] / df["balls_remaining"].replace(0, 1)

    total_balls_col = df.groupby("match_id")["balls_bowled"].transform("max").clip(lower=1)
    df["match_completion"] = (df["balls_bowled"] / total_balls_col).clip(0, 1)

    df["pressure_index"] = (
        (df["runs_remaining"].clip(0) / df["target"].replace(0, 1)) *
        (1.0 / (df["wickets_remaining"] + 1)) *
        df["match_completion"]
    ).clip(0, 10)

    df["chase_difficulty"] = (df["required_run_rate"] / 8.0).clip(0, 5)

    if "recent_12_balls_runs" in df.columns:
        df["recent_form"] = (df["recent_12_balls_runs"] / 24.0).clip(0, 1)
    else:
        df["recent_form"] = 0.5

    df["win_momentum"] = (df["run_rate_diff"] > 0).astype(float)

    df["danger_zone"] = (
        (df["required_run_rate"] > 12) & (df["wickets_remaining"] <= 5)
    ).astype(float)

    return df


def select_final_columns(df):
    feature_columns = [
        "balls_remaining", "runs_remaining", "wickets_remaining", "runs_scored",
        "current_run_rate", "required_run_rate", "target",
        "run_rate_diff", "balls_remaining_ratio", "wickets_remaining_ratio",
        "runs_per_ball_required", "match_completion", "pressure_index",
        "chase_difficulty", "recent_form", "win_momentum", "danger_zone",
    ]
    keep_cols = ["match_id"] + feature_columns + ["label"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].copy(), feature_columns


def fix_nan_inf(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.columns:
        if col != "match_id" and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def save_features(df, feature_columns, output_path="data/processed/features.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}  shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}")


if __name__ == "__main__":
    df = load_data("data/processed/ball_by_ball.csv")
    df = clean_data(df)
    df = engineer_features(df)
    df, feature_columns = select_final_columns(df)
    df = fix_nan_inf(df)
    save_features(df, feature_columns, "data/processed/features.csv")