# EntityExtractor

A named entity recognition (NER) web app that extracts and
categorizes key information from any text — people, organisations,
locations, dates, and skills — displayed as colour coded tags.
Built with Python, Flask, and Llama 3.1 via HuggingFace Inference API.

---

## Features

- Extracts 5 entity types from any text
- Colour coded tags — blue (persons), purple (organisations),
  green (locations), orange (dates), pink (skills)
- Returns total entity count and word count
- Three built-in sample texts — News Article, Job Description, Biography
- Structured JSON output from LLM with fallback error handling

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
User pastes text → clicks Extract Entities
        ↓
Flask validates minimum word count (10 words)
        ↓
Structured prompt instructs model to return JSON only
        ↓
Llama 3.1 extracts entities (temperature 0.1)
        ↓
Flask cleans markdown artifacts → parses JSON
        ↓
Colour coded tags rendered per category in browser
```

---

## Project Structure

```
entityextractor/
├── app.py        — Flask backend, structured prompt, JSON parsing
├── index.html    — Tag rendering UI with colour coded categories
├── .env          — API key (not committed)
└── README.md
```

---

## Getting Started

**1. Clone the repo**
```bash
git clone https://github.com/NAjamsher/gen-ai-projects.git
cd gen-ai-projects/entityextractor
```

**2. Install dependencies**
```bash
pip install flask flask-cors huggingface_hub python-dotenv
```

**3. Set up environment variables**

Create a `.env` file in the `entityextractor/` folder:
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

**Structured output**

The model is prompted to return a strict JSON schema with five
keys — persons, organizations, locations, dates, skills. Each
value is a list of strings. Empty lists are returned for
categories with no matches.

**Defensive JSON parsing**

Model output is sanitized before parsing to strip markdown code
fences that LLMs occasionally inject despite instructions.
All JSON parsing is wrapped in try/except to prevent server
crashes on malformed output.

```python
try:
    entities = json.loads(raw)
except json.JSONDecodeError:
    return {"error": "Could not parse entities. Please try again."}
```

**Temperature**

Set to 0.1 — the lowest across all projects. Entity extraction
requires deterministic, precise output. The same text should
return the same entities on every run.

---

## Environment Variables

| Variable | Description |
|---|---|
| HUGGINGFACE_API_KEY | HuggingFace user access token |

---

## License

MIT