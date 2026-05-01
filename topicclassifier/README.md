# TopicClassifier

A zero-shot text classification web app that categorises any text
into user-defined topic categories. Built with Python, Flask, and
Llama 3.1 via HuggingFace Inference API.

---

## Features

- Classifies any text into one of 10 default topic categories
- User can deselect categories or add custom ones at runtime
- Returns category, confidence percentage, and one-sentence reason
- Zero-shot — no training data or model fine-tuning required
- Animated confidence bar
- Four built-in example texts

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
User pastes text → selects categories → clicks Classify
        ↓
Browser sends text + selected categories to Flask
        ↓
Flask builds numbered category list and injects into prompt
        ↓
Llama 3.1 picks best matching category (temperature 0.1)
        ↓
Returns category + confidence + reason as JSON
        ↓
Result displayed with animated confidence bar
```

---

## Project Structure

```
topicclassifier/
├── app.py        — Flask backend, dynamic prompt, zero-shot logic
├── index.html    — Category chip UI, custom category input, result card
├── .env          — API key (not committed)
└── README.md
```

---

## Getting Started

**1. Clone the repo**
```bash
git clone https://github.com/NAjamsher/gen-ai-projects.git
cd gen-ai-projects/topicclassifier
```

**2. Install dependencies**
```bash
pip install flask flask-cors huggingface_hub python-dotenv
```

**3. Set up environment variables**

Create a `.env` file in the `topicclassifier/` folder:
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

**Zero-shot classification**

Categories are passed to the model at runtime via the prompt.
The model has not been fine-tuned on these specific categories —
it uses general language understanding to determine the best match.
Categories can be changed, added, or removed without any retraining.

**Dynamic prompt construction**

Selected categories are formatted as a numbered list using
Python's `enumerate()` and injected into the prompt at request
time. This allows the model to adapt to any set of user-defined
labels.

**Explainability**

The model is prompted to return a one-sentence reason alongside
the category and confidence score. This makes classification
decisions transparent and auditable.

**Structured output**

Response is returned as JSON with three fields — category,
confidence, and reason. Parsed with try/except to handle
malformed model output gracefully.

---

## Default Categories

Technology, Health & Medicine, Sports, Politics,
Business & Finance, Science, Entertainment,
Education, Environment, Crime & Law

Custom categories can be added via the UI at runtime.

---

## Environment Variables

| Variable | Description |
|---|---|
| HUGGINGFACE_API_KEY | HuggingFace user access token |

---

## License

MIT