# Memory Chatbot

A chatbot that remembers the full conversation history.
Built with LangChain ConversationBufferMemory and Llama 3.1
via HuggingFace Inference API.

---

## What it does

- Chat with an AI that remembers everything you said
- Reference earlier messages and the AI responds correctly
- Turn counter shows how many exchanges have happened
- Clear button wipes memory and starts fresh conversation

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Framework | LangChain |
| Memory | ConversationBufferMemory |
| AI Model | Llama 3.1 8B Instruct via HuggingFace |
| Provider | Novita via HuggingFace router |
| Frontend | HTML, CSS, JavaScript |
| Auth | python-dotenv |

---

## How memory works

```python
# step 1 — load history before every response
history = memory.load_memory_variables({})["history"]

# step 2 — generate response with full context
response = chat_chain.invoke({
    "history"    : history,
    "human_input": user_message
})

# step 3 — save exchange to memory after every response
memory.save_context(
    {"input" : user_message},
    {"output": response}
)
```

---

## Project Structure

```
memorychatbot/
├── app.py        — Flask backend, LangChain memory, chat logic
├── index.html    — Chat UI with memory indicator and clear button
├── .env          — API key (not committed)
└── README.md
```

---

## Getting Started

**1. Clone the repo**

```bash
git clone https://github.com/NAjamsher/gen-ai-projects.git
cd gen-ai-projects/memorychatbot
```

**2. Install dependencies**

```bash
pip install flask flask-cors huggingface_hub langchain langchain-core langchain-classic python-dotenv
```

**3. Set up environment variables**

Create a `.env` file inside the `memorychatbot/` folder:

```
HUGGINGFACE_API_KEY=your_key_here
```

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

**ConversationBufferMemory**

Stores every exchange in a growing buffer. Returns history
as plain text string injected directly into the PromptTemplate.
Memory persists for the entire server session.

**Two variable PromptTemplate**

```python
chat_template = PromptTemplate(
    input_variables=["history", "human_input"],
    template="Conversation so far: {history} Human: {human_input} Reply:"
)
```

{history} contains all previous messages.
{human_input} contains the current message.

**Clear memory route**

```python
@app.route("/clear", methods=["POST"])
def clear_memory():
    memory.clear()
    return jsonify({"status": "Memory cleared"})
```

Resets the conversation without restarting the server.

---

## Environment Variables

| Variable | Description |
|---|---|
| HUGGINGFACE_API_KEY | HuggingFace user access token |

---

## License

MIT