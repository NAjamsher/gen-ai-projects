# NLP Studio

A multi-tool AI platform combining 6 NLP capabilities in one
interface. Built with Python, Flask, and Llama 3.1 via HuggingFace
Inference API.

---

## Tools

| Tool | Description | Model |
|---|---|---|
| Chat | General AI chat | Llama 3.1 8B |
| Mood Detector | Sentiment analysis with confidence score | DistilBERT (local) |
| Summarizer | Condenses any text into 2-3 sentences | Llama 3.1 8B |
| Entity Extractor | Extracts people, places, orgs, dates, skills | Llama 3.1 8B |
| Topic Classifier | Zero-shot classification into any category | Llama 3.1 8B |
| Q&A Bot | Answers questions grounded to your document | Llama 3.1 8B |

---

## Features

- Single page application — all 6 tools in one interface, no page reload
- Sidebar navigation with active state highlighting
- Dark themed UI with centered layout and fade-in animations
- One shared Flask backend serving all 6 tools
- Chat history and memory in Q&A Bot
- Intent detection in Q&A Bot — greetings handled separately
- Structured JSON output for Entity Extractor and Topic Classifier
- Local DistilBERT inference for Mood Detector — no API call needed

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| AI Model | Llama 3.1 8B Instruct via HuggingFace Inference API |
| Sentiment Model | DistilBERT (local, HuggingFace Transformers) |
| Provider | Novita via HuggingFace router |
| Frontend | HTML, CSS, JavaScript |
| Auth | python-dotenv |

---

## How it works

```
User opens NLP Studio
        ↓
Clicks a tool in the sidebar
        ↓
JavaScript shows that panel — no page reload
        ↓
User submits input
        ↓
Browser sends POST to the corresponding Flask route
        ↓
Flask calls call_llama() or sentiment_pipeline()
        ↓
Result returned as JSON
        ↓
Browser displays result with animation
```

---

## Architecture

```
index.html
├── showPanel()          — single page routing, no reload
├── sendChat()           → POST /chat
├── runSentiment()       → POST /sentiment
├── runSummarizer()      → POST /summarize
├── runEntities()        → POST /entities
├── runClassifier()      → POST /classify
└── sendQA()             → POST /qa

app.py
├── call_llama()         — shared core function used by 5 tools
├── sentiment_pipeline() — local DistilBERT for mood detection
├── /chat                — general AI chat
├── /sentiment           — sentiment classification
├── /summarize           — text summarization
├── /entities            — named entity recognition
├── /classify            — zero-shot topic classification
└── /qa                  — context grounded question answering
```

---

## Project Structure

```
nlpstudio/
├── app.py        — Flask backend, all 6 tool routes, shared AI function
├── index.html    — Single page UI, sidebar navigation, all 6 tool panels
├── .env          — API key (not committed)
└── README.md
```

---

## Getting Started

**1. Clone the repo**

```bash
git clone https://github.com/NAjamsher/gen-ai-projects.git
cd gen-ai-projects/nlpstudio
```

**2. Install dependencies**

```bash
pip install flask flask-cors huggingface_hub transformers torch python-dotenv
```

**3. Set up environment variables**

Create a `.env` file inside the `nlpstudio/` folder:

```
HUGGINGFACE_API_KEY=your_key_here
```

Get your free API key at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

**4. Run**

```bash
python app.py
```

Wait for:
```
All systems ready!
* Running on http://127.0.0.1:5000
```

**5. Open in browser**

```
http://127.0.0.1:5000
```

---

## Key Implementation Details

**Shared core function**

All 5 LLM-based tools call one shared `call_llama()` function
with different parameters. This follows the DRY principle —
Don't Repeat Yourself.

```python
def call_llama(prompt, system="You are a helpful assistant.", temp=0.7, tokens=300):
    response = client.chat.completions.create(...)
    return response.choices[0].message.content.strip()

# each tool calls it differently
call_llama(prompt, temp=0.7, tokens=300)  # chat — creative
call_llama(prompt, temp=0.3, tokens=150)  # summarizer — accurate
call_llama(prompt, temp=0.1, tokens=400)  # entities — precise
call_llama(prompt, temp=0.1, tokens=200)  # classifier — precise
```

**Single page application routing**

JavaScript toggles CSS classes to show and hide tool panels
without any page reload. No frameworks — pure vanilla JS.

```javascript
function showPanel(id) {
    document.querySelectorAll('.tool-panel').forEach(p => p.classList.remove('active'))
    document.getElementById('panel-' + id).classList.add('active')
}
```

**Temperature strategy**

| Tool | Temperature | Reason |
|---|---|---|
| Chat | 0.7 | Conversational, needs some creativity |
| Summarizer | 0.3 | Faithful to source, limited creativity |
| Entity Extractor | 0.1 | Deterministic, same result every time |
| Topic Classifier | 0.1 | Must pick exact category, no variation |
| Q&A Bot | 0.1 | Grounded to context, no hallucination |

---

## Environment Variables

| Variable | Description |
|---|---|
| HUGGINGFACE_API_KEY | HuggingFace user access token |

---

## License

MIT