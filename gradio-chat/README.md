# GradioChat

AI chatbot with conversation memory — built entirely with Gradio.
No Flask. No HTML. No JavaScript. Just Python.
Powered by Llama 3.1 via HuggingFace.

---

## What it does

- Full AI chat interface built in pure Python
- Remembers full conversation history across messages
- Gradio generates the entire UI automatically — input box, chat bubbles, send button
- No frontend code required

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI Framework | Gradio |
| AI Model | Llama 3.1 8B Instruct |
| Provider | Novita via HuggingFace |
| Framework | LangChain (PromptTemplate, RunnableLambda) |
| Memory | Manual history from Gradio history object |
| Auth | python-dotenv |

---

## How it works
User types message in Gradio UI
↓
Gradio calls chatbot(user_message, history) automatically
↓
Function converts Gradio history list to plain text string
↓
LangChain chain invoked with history + human_input
↓
PromptTemplate fills {history} and {human_input}
↓
Llama 3.1 reads full context and generates reply
↓
Function returns plain string
↓
Gradio displays reply as chat bubble automatically

---

## Project Structure
gradio-chat/
├── app.py      — Gradio ChatInterface, LangChain chain, history handling
├── .env        — API key (never pushed)
└── README.md

---

## Getting Started

```bash
cd gradio-chat
pip install gradio huggingface-hub langchain-core python-dotenv
```

Create `.env`:
HUGGINGFACE_API_KEY=hf_your_key_here

Run:
```bash
python app.py
```

Open: `http://127.0.0.1:7860`

> Note: Gradio runs on port 7860 — not 5000 like Flask.

---

## Key Implementation Details

- Gradio replaces Flask and index.html entirely — one `gr.ChatInterface` call builds the full UI
- Gradio passes conversation history automatically as a list of role/content dictionaries
- History converted to plain text string before injecting into PromptTemplate
- Port is 7860 by default — different from all other projects in this series

---

## Environment Variables

| Variable | Description |
|---|---|
| HUGGINGFACE_API_KEY | HuggingFace user access token |

---

## License

MIT