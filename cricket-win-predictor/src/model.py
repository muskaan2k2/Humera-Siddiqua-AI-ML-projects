import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam


def build_model(input_shape):
    """
    LSTM model for cricket win probability prediction.

    input_shape: (sequence_length, num_features)
                 e.g. (10, 17) means last 10 balls, 17 features each

    Architecture:
        Input → LSTM(128) → Dropout → LSTM(64) → Dropout → Dense(32) → Dense(1, sigmoid)

    Output: single value between 0 and 1 (win probability)
    """
    model = Sequential([
        Input(shape=input_shape),

        # First LSTM layer — learns broad match patterns
        LSTM(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),

        # Second LSTM layer — learns finer ball-by-ball patterns
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),

        # Dense layers — final decision making
        Dense(32, activation="relu"),
        Dropout(0.2),

        # Output: win probability (0 to 1)
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    return model


if __name__ == "__main__":
    # Quick test: build and print summary
    model = build_model(input_shape=(10, 17))
    model.summary()