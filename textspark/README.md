# MoodDetector

A sentiment analysis web app that detects the emotional tone of any
sentence — Positive or Negative — with a confidence score.
Built with Python, Flask, and DistilBERT running locally via
HuggingFace Transformers.

---

## Features

- Detects sentiment of any English sentence
- Returns confidence score as an animated percentage bar
- Color coded result card — green for positive, red for negative
- Six one-click example sentences for quick testing
- Model runs locally — no API call per request after initial load

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| AI Model | DistilBERT (distilbert-base-uncased-finetuned-sst-2-english) |
| Inference | HuggingFace Transformers (local) |
| Frontend | HTML, CSS, JavaScript |

---

## How it works

```
User types sentence
        ↓
Browser sends POST request to Flask /analyze
        ↓
Flask passes text to local DistilBERT pipeline
        ↓
Model returns label (POSITIVE/NEGATIVE) + confidence score
        ↓
Result displayed with color card and animated confidence bar
```

---

## Project Structure

```
mooddetector/
├── app.py        — Flask backend, DistilBERT pipeline, sentiment logic
├── index.html    — Result card UI with confidence bar and example chips
└── README.md
```

---

## Getting Started

**1. Clone the repo**

```bash
git clone https://github.com/NAjamsher/gen-ai-projects.git
cd gen-ai-projects/mooddetector
```

**2. Install dependencies**

```bash
pip install flask flask-cors transformers torch
```

**3. Run**

```bash
python app.py
```

Model downloads automatically on first run (~250MB, one time only).

**4. Open in browser**

```
http://127.0.0.1:5000
```

---

## Key Implementation Details

**Local inference**

Unlike API-based projects, DistilBERT runs directly on your machine
using HuggingFace Transformers. The model downloads once and is
cached locally. No internet required after the initial download.
No API key needed.

**DistilBERT**

A distilled version of BERT fine-tuned on the Stanford Sentiment
Treebank (SST-2) dataset. 40% smaller and 60% faster than BERT
while retaining 97% of its accuracy on sentiment tasks.

**Confidence score**

The model returns a raw float score between 0 and 1 alongside
the label. Scores above 0.9 indicate high confidence. Scores
near 0.5 indicate ambiguous sentiment.

```python
result     = sentiment_pipeline(text)[0]
label      = result["label"]        # POSITIVE or NEGATIVE
score      = result["score"]        # 0.9998
confidence = round(score * 100, 2)  # 99.98
```

---

## Environment Variables

No API key required. Model runs locally.

---

## License

MIT