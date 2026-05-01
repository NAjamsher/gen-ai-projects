# TextSpark

An AI-powered chat application that generates intelligent responses
to any user prompt. Built with Python, Flask, and Llama 3.1 via
HuggingFace Inference API.

---

## Features

- Real-time AI chat in the browser
- Powered by Meta's Llama 3.1 8B Instruct model
- Flask backend keeps API key secure and hidden from client
- Clean chat UI with message bubbles
- Enter key support for quick messaging

---

## Tech Stack

- **Backend** — Python, Flask
- **AI Model** — Llama 3.1 8B Instruct via HuggingFace Inference API
- **Provider** — Novita (HuggingFace router)
- **Frontend** — HTML, CSS, JavaScript
- **Auth** — python-dotenv for API key management

---

## How it works

```User types message → clicks Send
↓
Browser sends POST request to Flask /chat endpoint
↓
Flask forwards message to Llama 3.1 via HuggingFace API
↓
Model generates response
↓
Response displayed as chat bubble in browser

---

## Project Structuretextspark/
├── app.py        — Flask backend, API call to HuggingFace
├── index.html    — Chat UI with message bubbles
├── .env          — API key (not committed)
└── README.md

---

## Getting Started

**1. Clone the repo**
```bashgit clone https://github.com/NAjamsher/gen-ai-projects.git
cd gen-ai-projects/textspark

**2. Install dependencies**
```bashpip install flask flask-cors huggingface_hub python-dotenv

**3. Set up environment variables**

Create a `.env` file in the `textspark/` folder:HUGGINGFACE_API_KEY=your_key_here

Get your free API key at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

**4. Run**
```bashpython app.py

**5. Open in browser**http://127.0.0.1:5000

---

## Key Implementation Details

**Client-server architecture**

The API key never leaves the server. The browser communicates only
with the Flask backend which handles all HuggingFace API calls.
This is the standard architecture for any AI-powered web application.

**HuggingFace Inference API**

Uses the `/v1/chat/completions` compatible endpoint via the Novita
provider for fast and reliable inference on the free tier.

---

## Environment Variables

| Variable | Description |
|---|---|
| HUGGINGFACE_API_KEY | HuggingFace user access token |

---

## License

MIT
