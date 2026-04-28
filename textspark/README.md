# TextSpark

A simple AI chat app powered by Meta's Llama 3.1 model via HuggingFace.
Type any question in the browser and get an intelligent response back.

---

## What this project does

- Takes a user message from a browser chat interface
- Sends it to a Flask backend running on your laptop
- Flask calls the Llama 3.1 AI model via HuggingFace API
- The AI response comes back and displays as a chat bubble

---

## The core concept — Client Server Model
Browser (index.html)  →  POST request with your message
↓
Flask server (app.py)  →  calls HuggingFace API
↓
HuggingFace (Llama 3.1)  →  generates AI response
↓
Flask  →  sends JSON response back
↓
Browser  →  displays reply in chat bubble

This loop happens every single time you click Send.
Every AI web app on the planet follows this exact pattern.

---

## Concepts you learned building this

**Python concepts**
- Variables — named containers that store values
- Functions — reusable blocks of code defined with def
- While loop — keeps the program alive and waiting for input
- If conditions — makes decisions based on what user typed
- Strings and methods — .lower() .strip() to clean user input

**API concepts**
- API — a way for two programs to talk over the internet
- API key — your secret password sent with every request
- POST request — sending data to a server (vs GET which just fetches)
- JSON — the universal format both Python and JavaScript understand
- Tokens — the unit AI uses to measure text, roughly 3/4 of a word
- Temperature — controls AI creativity, 0 is factual, 1 is creative

**Flask concepts**
- Flask — a Python library that turns your code into a web server
- Route — an address inside your server like /chat or /
- @app.route() — decorator that maps a URL to a Python function
- request.get_json() — reads JSON data sent from the browser
- jsonify() — converts Python dict into JSON to send back
- CORS — allows browser to talk to a different server (your Flask)

**JavaScript concepts**
- fetch() — browser's way of sending a request to your Flask server
- async/await — waits for server response without freezing the browser
- document.createElement() — creates new HTML elements on the fly
- JSON.stringify() — converts JS object to JSON string to send
- response.json() — converts server's JSON response back to JS object

---

## AI concepts behind the model

**LLM (Large Language Model)**
A model trained on billions of words that can generate human-like text.
Llama 3.1 is Meta's open source LLM — same category as ChatGPT.

**Inference**
When you send a prompt and the model generates a response, that is
called inference. You are not training the model — you are using it.

**Parameters (max_tokens, temperature)**
- max_tokens — how long the response can be
- temperature — how creative or random the response is

**HuggingFace**
A platform that hosts thousands of open source AI models.
You access them via API using your free HuggingFace token.

**Novita provider**
HuggingFace routes free API calls through third party providers.
Novita is one such provider that hosts Llama 3.1 for free usage.

---

## Files in this project
textspark/
├── app.py        → Flask backend, handles API call to HuggingFace
└── index.html    → Browser frontend, chat UI with HTML CSS JS

---

## How to run

Step 1 — install dependencies
```bash
pip install flask flask-cors huggingface_hub
```

Step 2 — add your HuggingFace API key in app.py
```python
API_KEY = "hf_xxxxxxxxxxxxxxxx"
```

Step 3 — run the server
```bash
cd textspark
python app.py
```

Step 4 — open in browser
http://127.0.0.1:5000

---

## What to remember

- Never share your API_KEY publicly — treat it like a password
- Flask must be running before you open the browser
- The browser never touches HuggingFace directly — always goes through Flask
- JSON is the language browser and server use to talk to each other
- Every AI web app you build follows the same Client → Server → AI → Server → Client loop