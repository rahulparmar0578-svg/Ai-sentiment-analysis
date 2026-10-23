"""
train.py — Train LSTM sentiment model on IMDb reviews.

Pipeline:
  1. Load IMDb dataset via tf.keras.datasets.imdb (already labeled)
  2. Clean / decode reviews (utility handles integer sequences)
  3. Tokenize via Keras Tokenizer (fit on training texts)
  4. Pad / truncate to fixed length (MAX_LEN)
  5. Embedding -> LSTM -> Dropout -> Dense(sigmoid)
  6. Train with early stopping, save model + tokenizer + metrics
"""

import json
import os
import pickle
import random
import re

import numpy as np

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ── Config ──────────────────────────────────────────────────────────
VOCAB_SIZE = 10000        # top-N most frequent words
MAX_LEN = 200             # pad / truncate length
EMBEDDING_DIM = 128
LSTM_UNITS = 64
DROPOUT_RATE = 0.5
BATCH_SIZE = 128
EPOCHS = 5
TEST_SAMPLES = 5000       # held-out test slice size

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ── Helpers ─────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Basic normalisation: lowercase, strip HTML breaks, keep alphanumerics."""
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def decode_imdb_sequence(seq, word_index_reverse):
    """Convert integer sequence back to text (for tokenizer fitting)."""
    return " ".join(word_index_reverse.get(i, "") for i in seq)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading IMDb dataset...")
    (x_train_raw, y_train), (x_test_raw, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

    word_index = imdb.get_word_index()
    # imdb reserves 0=pad, 1=start, 2=unk, 3=unused — shift
    reverse_word_index = {v + 3: k for k, v in word_index.items()}
    reverse_word_index[0] = ""
    reverse_word_index[1] = ""
    reverse_word_index[2] = ""

    # Decode integer sequences to text so we can use a single Tokenizer
    # pipeline that is shared with inference (app.py).
    print("Decoding sequences to text...")
    train_texts = [clean_text(decode_imdb_sequence(s, reverse_word_index)) for s in x_train_raw]
    test_texts = [clean_text(decode_imdb_sequence(s, reverse_word_index)) for s in x_test_raw]

    # ── Tokenize ────────────────────────────────────────────────
    print("Fitting tokenizer...")
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)

    train_seq = tokenizer.texts_to_sequences(train_texts)
    test_seq = tokenizer.texts_to_sequences(test_texts)

    # ── Pad / truncate ──────────────────────────────────────────
    x_train = pad_sequences(train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
    x_test = pad_sequences(test_seq, maxlen=MAX_LEN, padding="post", truncating="post")
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print(f"Train shape: {x_train.shape}  Test shape: {x_test.shape}")

    # ── Model: Embedding -> LSTM -> Dropout -> Dense(sigmoid) ──
    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
        LSTM(LSTM_UNITS),
        Dropout(DROPOUT_RATE),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

    print("Training...")
    history = model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1,
    )

    # ── Evaluate on held-out test set ───────────────────────────
    print("Evaluating on test set...")
    y_prob = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": cm,
        "confusion_matrix_labels": ["Negative", "Positive"],
        "vocab_size": VOCAB_SIZE,
        "max_len": MAX_LEN,
        "embedding_dim": EMBEDDING_DIM,
        "lstm_units": LSTM_UNITS,
    }
    print(json.dumps(metrics, indent=2))

    # ── Save artifacts ──────────────────────────────────────────
    model_path = os.path.join(MODEL_DIR, "sentiment_model.keras")
    tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.pkl")
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")

    model.save(model_path)
    print(f"Model saved -> {model_path}")

    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved -> {tokenizer_path}")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved -> {metrics_path}")
    print("Done.")


if __name__ == "__main__":
    main()
