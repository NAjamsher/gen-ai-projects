# IntentRouter

Type anything — the AI detects what you want and routes it to the right processing chain automatically.
Powered by Llama 3.1 via HuggingFace.

---

## What it does

- Detects user intent from any free-text message
- Routes to one of five specialized LangChain chains
- SUMMARIZE — compresses long text into 2-3 sentences
- SENTIMENT — detects emotion with confidence score and reason
- QA — answers factual questions clearly
- GENERATE — produces blog intro, tweet, and tagline from a topic
- CHAT — handles general conversation naturally
- Each chain has its own prompt template and temperature tuned for its task

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| AI Model | Llama 3.1 8B Instruct |
| Provider | Novita via HuggingFace |
| Framework | LangChain (PromptTemplate, RunnableLambda, StrOutputParser) |
| Frontend | HTML, CSS, JavaScript |
| Auth | python-dotenv |

---

## How it works
User types any message
↓
Intent chain reads message → returns one keyword
SUMMARIZE / SENTIMENT / QA / GENERATE / CHAT
↓
for-else pattern cleans and validates the intent
↓
Router sends message to matching chain
↓
Each chain uses its own prompt and temperature
↓
Flask returns intent name + result as JSON
↓
Browser shows colored intent badge and formatted result

---

## Project Structure
intent-router/
├── app.py        — Flask server, 6 LangChain chains, router function
├── index.html    — Single input UI with intent badge and result display
├── .env          — API key (never pushed)
└── README.md

---

## Getting Started

```bash
cd intent-router
pip install flask flask-cors huggingface-hub langchain-core python-dotenv
```

Create `.env`:
HUGGINGFACE_API_KEY=hf_your_key_here

Run:
```bash
python app.py
```

Open: `http://127.0.0.1:5000`

---

## Key Implementation Details

- Intent detection chain uses temperature 0.1 — must return exact one-word classification
- Each of the five chains uses its own lambda with custom temperature and token limit
- for-else pattern provides safe fallback to CHAT when intent is unrecognized
- `keyword in intent` used instead of exact match — handles cases where model adds extra words
- Sentiment chain returns structured JSON with sentiment, confidence, and reason fields

---

## Environment Variables

| Variable | Description |
|---|---|
| HUGGINGFACE_API_KEY | HuggingFace user access token |

---

## License

MIT