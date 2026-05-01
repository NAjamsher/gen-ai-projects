# Summarizer

A text summarization web app that condenses any long paragraph or
article into 2-3 clear sentences. Built with Python, Flask, and
Llama 3.1 via HuggingFace Inference API using prompt engineering.

---

## Features

- Summarizes any text of 30 words or more
- Displays original word count, summary word count, and reduction percentage
- Live word counter that updates as you type
- Three built-in sample texts — Climate Change, AI, Healthcare
- Animated stat cards showing compression metrics

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| AI Model | Llama 3.1 8B Instruct |
| Provider | Novita via HuggingFace |
| Frontend | HTML, CSS, JavaScript |
| Auth | python-dotenv |

---

## How it works

```
User pastes text
        ↓
Flask validates minimum 30 words
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

Create a `.env` file inside the `summarizer/` folder:

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

The model is instructed via a structured prompt to return only
the summary with no additional commentary. A `Summary:` suffix
anchors the model output — a standard prompt engineering
technique for controlled generation.

```python
prompt = f"""Summarize the following text in 2-3 sentences.
Only give the summary, nothing else.

Text:
{text}

Summary:"""
```

**Temperature**

Set to 0.3 for summarization tasks. Lower than conversational
applications to ensure faithful compression of the source text
without creative deviation.

**Word reduction calculation**

```python
reduced_by = round((1 - summary_words / original_words) * 100)
```

Calculates percentage compression displayed as a stat card.

---

## Environment Variables

| Variable | Description |
|---|---|
| HUGGINGFACE_API_KEY | HuggingFace user access token |

---

## License

MIT
