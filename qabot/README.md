# QABot

A context-aware question answering chatbot that answers questions
strictly from user-provided text. Built with Python, Flask, and
Llama 3.1 via HuggingFace Inference API.

---

## Features

- Paste any document, article, or paragraph as context
- Ask unlimited questions about the provided text
- Multi-turn conversation with memory of last 3 exchanges
- Greeting and casual message detection — handled separately from QA
- Answers grounded strictly to provided context — no hallucination
- Returns "I could not find the answer" when answer is not in the text
- Suggested questions for sample contexts
- Dark themed chat UI

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
User pastes context → asks question
        ↓
Flask receives context + question + chat history
        ↓
Intent detection — greeting or actual question?
        ↓
If greeting → casual response (temperature 0.7)
If question → grounded QA prompt (temperature 0.1)
        ↓
Llama 3.1 answers strictly from provided context
        ↓
Response displayed in chat UI
```

---

## Project Structure

```
qabot/
├── app.py        — Flask backend, intent detection, QA logic
├── index.html    — Chat UI with context input and message bubbles
├── .env          — API key (not committed)
└── README.md
```

---

## Getting Started

**1. Clone the repo**
```bash
git clone https://github.com/NAjamsher/gen-ai-projects.git
cd gen-ai-projects/qabot
```

**2. Install dependencies**
```bash
pip install flask flask-cors huggingface_hub python-dotenv
```

**3. Set up environment variables**

Create a `.env` file in the `qabot/` folder:
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

**Context grounding**

The model is instructed via prompt to answer only from the provided
context and return a fixed fallback message when the answer is not found.
This prevents hallucination and keeps responses factual.

**Intent detection**

User input is checked against a list of common greetings and casual
phrases before hitting the QA pipeline. Casual messages are routed
to a separate prompt with higher temperature for natural responses.

**Conversation memory**

The last 3 question-answer pairs are injected into each new prompt
as conversation history. This allows follow-up questions without
losing context from previous exchanges.

---

## Environment Variables

| Variable | Description |
|---|---|
| HUGGINGFACE_API_KEY | HuggingFace user access token |

---

## License

MIT