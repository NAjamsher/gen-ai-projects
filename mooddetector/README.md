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

- **Backend** — Python, Flask
- **AI Model** — DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)
- **Inference** — HuggingFace Transformers pipeline (local)
- **Frontend** — HTML, CSS, JavaScript

---

## How it works

```
User types sentence → clicks Detect Mood
        ↓
Browser sends POST request to Flask /analyze endpoint
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

The model returns a raw float score between 0 and 1 alongside the
label. Scores above 0.9 indicate high confidence. Scores near 0.5
indicate ambiguous sentiment.

---

## License

MIT

markdown# Summarizer

A text summarization web app that condenses any long paragraph or
article into 2-3 clear sentences. Built with Python, Flask, and
Llama 3.1 via HuggingFace Inference API using prompt engineering.

---

## Features

- Summarizes any text of 30 words or more
- Displays original word count, summary word count, and reduction percentage
- Live word counter updates as you type
- Three built-in sample texts — Climate Change, AI, Healthcare
- Animated stat cards showing compression metrics

---

## Tech Stack

- **Backend** — Python, Flask
- **AI Model** — Llama 3.1 8B Instruct via HuggingFace Inference API
- **Provider** — Novita (HuggingFace router)
- **Frontend** — HTML, CSS, JavaScript
- **Auth** — python-dotenv for API key management

---

## How it works

```
User pastes text → clicks Summarize
        ↓
Flask validates word count (minimum 30 words)
        ↓
Prompt engineering — text wrapped with summarization instructions
        ↓
Llama 3.1 generates 2-3 sentence summary (temperature 0.3)
        ↓
Flask calculates word counts and reduction percentage
        ↓
Summary and stats displayed in browser
```

---

## Project Structure

```
summarizer/
├── app.py        — Flask backend, prompt construction, word count logic
├── index.html    — Textarea input, summary card, stat display
├── .env          — API key (not committed)
└── README.md
```

---

## Getting Started

**1. Clone the repo**
```bash
git clone https://github.com/NAjamsher/gen-ai-projects.git
cd gen-ai-projects/summarizer
```

**2. Install dependencies**
```bash
pip install flask flask-cors huggingface_hub python-dotenv
```

**3. Set up environment variables**

Create a `.env` file in the `summarizer/` folder:
```
HUGGINGFACE_API_KEY=your_key_here
```

Get your free API key at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

**4. Run**
```bash
python app.py
```

**5. Open in browser**
```
http://127.0.0.1:5000
```

---

## Key Implementation Details

**Prompt engineering**

The model is instructed via a structured prompt to return only the
summary with no additional commentary. The prompt uses a
`Summary:` suffix to anchor the model output — a standard
prompt engineering technique for controlled generation.

**Temperature**

Set to 0.3 for summarization tasks. Lower than conversational
applications (0.7) to ensure faithful compression of the source
text without creative deviation.

**Word reduction calculation**

```python
reduced_by = round((1 - summary_words / original_words) * 100)
```

Calculates percentage compression as a display metric.

---

## Environment Variables

| Variable | Description |
|---|---|
| HUGGINGFACE_API_KEY | HuggingFace user access token |

---

## License

MIT

