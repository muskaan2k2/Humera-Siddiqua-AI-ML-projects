import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf
from model import build_model

# ── Config ──────────────────────────────────────────
FEATURES_PATH  = "data/processed/features.csv"
MODEL_PATH     = "models/cricket_model.keras"
SCALER_PATH    = "models/scaler.pkl"
SEQUENCE_LEN   = 10       # use last 10 balls as one input window
BATCH_SIZE     = 1024
EPOCHS         = 20
# ────────────────────────────────────────────────────

FEATURE_COLS = [
    "balls_remaining", "runs_remaining", "wickets_remaining", "runs_scored",
    "current_run_rate", "required_run_rate", "target",
    "run_rate_diff", "balls_remaining_ratio", "wickets_remaining_ratio",
    "runs_per_ball_required", "match_completion", "pressure_index",
    "chase_difficulty", "recent_form", "win_momentum", "danger_zone",
]


def load_and_scale(path):
    print("Loading features.csv ...")
    df = pd.read_csv(path, low_memory=False)
    print(f"Shape: {df.shape}")

    scaler = StandardScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved → {SCALER_PATH}")
    return df


def make_sequences(df):
    """
    For each match, slide a window of SEQUENCE_LEN balls across the innings.
    Each window becomes one training sample.
    X shape: (num_samples, SEQUENCE_LEN, num_features)
    y shape: (num_samples,)
    """
    print(f"Building sequences (window={SEQUENCE_LEN}) ...")
    X_list, y_list = [], []

    groups = df.groupby("match_id")
    total = len(groups)

    for i, (_, match_df) in enumerate(groups):
        if i % 2000 == 0:
            print(f"  {i}/{total} matches processed", end="\r")

        values = match_df[FEATURE_COLS].values
        labels = match_df["label"].values

        if len(values) < SEQUENCE_LEN:
            continue

        for j in range(SEQUENCE_LEN, len(values)):
            X_list.append(values[j - SEQUENCE_LEN: j])
            y_list.append(labels[j])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    print(f"\nSequences built — X: {X.shape}  y: {y.shape}")
    return X, y


def split(X, y):
    # 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def train(X_train, y_train, X_val, y_val):
    model = build_model(input_shape=(SEQUENCE_LEN, len(FEATURE_COLS)))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=4, restore_best_weights=True, mode="max"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_auc", save_best_only=True, mode="max", verbose=1
        ),
    ]

    print("\nTraining ...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    return model, history


def save_test_data(X_test, y_test):
    np.save("models/X_test.npy", X_test)
    np.save("models/y_test.npy", y_test)
    print(f"Test data saved → models/X_test.npy  models/y_test.npy")


if __name__ == "__main__":
    df                                      = load_and_scale(FEATURES_PATH)
    X, y                                    = make_sequences(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split(X, y)
    model, history                          = train(X_train, y_train, X_val, y_val)
    save_test_data(X_test, y_test)
    print(f"\nModel saved → {MODEL_PATH}")