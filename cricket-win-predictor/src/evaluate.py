import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # save plots without needing a display window

from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve
)
import tensorflow as tf

MODEL_PATH = "models/cricket_model.keras"
X_TEST_PATH = "models/X_test.npy"
Y_TEST_PATH = "models/y_test.npy"


def load():
    print("Loading model and test data ...")
    model = tf.keras.models.load_model(MODEL_PATH)
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)
    print(f"Test samples: {len(y_test):,}")
    return model, X_test, y_test


def evaluate(model, X_test, y_test):
    print("Running predictions ...")
    y_prob = model.predict(X_test, batch_size=2048, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_prob)
    logloss = log_loss(y_test, y_prob)

    print(f"\n{'='*35}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  Log Loss  : {logloss:.4f}")
    print(f"{'='*35}\n")

    return y_pred, y_prob


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Loss", "Win"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix")
    path = "models/confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved → {path}")


def plot_roc_curve(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    path = "models/roc_curve.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved → {path}")


if __name__ == "__main__":
    model, X_test, y_test = load()
    y_pred, y_prob        = evaluate(model, X_test, y_test)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_prob)