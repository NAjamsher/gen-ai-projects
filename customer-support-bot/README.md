# Customer Support Bot

A production-grade AI customer support system with session management, intent detection, order lookup, ticket creation, and escalation — built with LangChain, FAISS, and Llama 3.1.

---

## What it does

- Detects customer intent across 8 categories — ORDER_STATUS, RETURN, CANCEL, COMPLAINT, PAYMENT, GENERAL, GREETING, ESCALATE
- Manages isolated sessions per customer — multiple users can chat simultaneously without interference
- Looks up real order data from a database and passes details to the LLM as context
- Retrieves relevant company policies using RAG (FAISS + semantic search)
- Creates support tickets automatically for complaints and escalations
- Remembers the last 6 messages per session for natural follow-up conversations

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| AI Model | Llama 3.1 8B Instruct |
| Provider | Novita via HuggingFace |
| Framework | LangChain (LCEL) |
| Vector Store | FAISS |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Frontend | HTML, CSS, JavaScript |
| Auth | python-dotenv |

---

## How it works

```
Customer sends message
        |
        ▼
Detect intent (8 categories) using Llama at temperature 0.1
        |
        ▼
Extract order ID if present → lookup in order database
        |
        ▼
Retrieve relevant company policies from FAISS vector store
        |
        ▼
Generate response using Llama with full context:
  - retrieved policies
  - order details
  - last 6 conversation turns
        |
        ▼
Create ticket if ESCALATE or COMPLAINT intent detected
        |
        ▼
Save to session history → return response to customer
```

---

## Project Structure

```
customer-support-bot/
├── app.py        ← Flask server, RAG pipeline, session management
├── index.html    ← Dark themed chat UI with order lookup sidebar
├── .env          ← API key (not pushed to GitHub)
└── README.md
```

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/NAjamsher/gen-ai-projects
cd gen-ai-projects/customer-support-bot

# Install dependencies
pip install flask flask-cors huggingface-hub langchain-core
pip install langchain-community langchain-huggingface
pip install faiss-cpu sentence-transformers python-dotenv

# Add your API key
echo "HUGGINGFACE_API_KEY=your_key_here" > .env

# Run the app
python app.py
```

Open browser at `http://127.0.0.1:5001`

---

## Key Implementation Details

- Session management isolates each customer's conversation history — `sessions[session_id]` stores history, ticket ID, and escalation state
- Intent detection uses a separate LangChain chain at `temperature=0.1` for reliable single-word classification
- Context routing: different intents retrieve different company policies from FAISS before generating a response
- Ticket IDs are generated as `TKT` + 6 random digits — stored with priority, issue, creation time, and status
- Support agent persona (Aria) is defined in the system prompt with `temperature=0.4` for professional but warm responses

---

## Test Order IDs

```
ORD001 — Delivered
ORD002 — In Transit
ORD003 — Processing
ORD004 — Cancelled
ORD005 — Out for Delivery
```

---

## Environment Variables

| Variable | Description |
|---|---|
| HUGGINGFACE_API_KEY | HuggingFace user access token |

---

## License

MIT