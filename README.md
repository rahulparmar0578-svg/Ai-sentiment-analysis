# AI-Based Sentiment Analysis System

A college-level **NLP + Deep Learning** project that classifies movie reviews as **Positive** or **Negative** using an LSTM model trained on the IMDb dataset. Simple Streamlit UI that runs locally in your browser.

## Technologies Used

| Component | Library |
|---|---|
| UI | Streamlit |
| Deep Learning | TensorFlow / Keras |
| Dataset | IMDb Movie Reviews (via `tf.keras.datasets.imdb`) |
| NLP | Keras Tokenizer, pad_sequences |
| Evaluation | Scikit-learn, Matplotlib |

## NLP Workflow

1. **Load** — 50,000 labeled IMDb reviews via `tf.keras.datasets.imdb`.
2. **Clean & normalize** — lowercasing, strip HTML breaks, remove non-alphanumeric characters, collapse whitespace.
3. **Tokenize** — `Tokenizer(num_words=10000)` maps each word to an integer ID.
4. **Pad / truncate** — `pad_sequences(maxlen=200, padding="post")` gives every review the same length.
5. **Embedding** — `Embedding(10000 → 128)` converts integer IDs to dense vectors.
6. **Same pipeline at inference** — user-entered text goes through identical `clean → tokenize → pad` steps before the model.

## Deep Learning Architecture

```
Input (200 tokens)
  → Embedding (vocab 10,000 → 128-dim)
  → LSTM (64 units)
  → Dropout (0.5)
  → Dense (1 unit, sigmoid)
```

- **Loss:** binary_crossentropy
- **Optimizer:** Adam
- **Early stopping** on `val_loss` (patience 2, restores best weights)
- **Reproducible seed:** 42

## Dataset

**IMDb Movie Reviews** — 50,000 reviews (25k train / 25k test), binary labeled. Loaded automatically on first training run; no manual download needed.

## Project Structure

```
sentiment-analysis/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
├── model/
│   ├── sentiment_model.keras
│   └── tokenizer.pkl
├── results/
│   └── metrics.json
└── report/
    ├── Rahul_Parmar_Sentiment_Analysis_Report.docx
    └── Rahul_Parmar_Sentiment_Analysis_Report.pdf
```

## Setup — Windows PowerShell

### 1. Create a virtual environment

```powershell
cd "C:\Users\loq\OneDrive\Desktop\New folder\sentiment-analysis"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> If activation is blocked, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### 2. Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Train the model (only if the saved model files are absent)

```powershell
python train.py
```

- Downloads the IMDb dataset (~30 MB) on first run.
- Trains for up to 5 epochs with early stopping (~5–15 min on CPU).
- Saves `model/sentiment_model.keras`, `model/tokenizer.pkl`, `results/metrics.json`.

### 4. Launch the web app

```powershell
streamlit run app.py
```

Open the URL shown in the terminal (typically **http://localhost:8501**) and analyze reviews!

## Expected Workflow

1. Install dependencies
2. Train the model once (`python train.py`)
3. Launch the app (`streamlit run app.py`)
4. Open `http://localhost:8501` → type a review → click **Analyze Sentiment**

## Model Evaluation

After training, `results/metrics.json` and the app's metrics section show performance on the **held-out test set** (never seen during training/validation):

- Accuracy, Precision, Recall, F1-score
- Confusion matrix

No metrics are hardcoded — all values come from the actual evaluation run.

## Known Limitations

- Vocabulary limited to top 10,000 words; rare words become `<OOV>`.
- Reviews truncated/padded to 200 tokens — very long reviews lose tail content.
- Binary only (Positive / Negative); no neutral class.
- Model is lightweight for college/demo use, not production-grade.
- First training run requires internet to download the IMDb dataset.
