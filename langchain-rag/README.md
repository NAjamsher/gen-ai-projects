# LangChain RAG — Document Q&A System

A Retrieval Augmented Generation system built with LangChain, FAISS, and Llama 3.1 — paste any document, ask unlimited questions, get answers grounded strictly in your content.

---

## What it does

- Paste any document — articles, research papers, reports, notes
- System automatically chunks the document into semantically searchable pieces
- Ask unlimited questions — answers come only from your document
- Semantic retrieval via FAISS — finds relevant chunks by meaning, not exact words
- Hallucination prevention built into the prompt — AI cannot answer from outside the document
- Built entirely with LangChain LCEL — clean, composable chain architecture

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
User pastes document
        |
        ▼
Text split into sentence-based chunks (3 sentences per chunk)
        |
        ▼
Each chunk embedded → 384-dimensional vector
        |
        ▼
All vectors stored in FAISS index
        |
        ▼
User asks question
        |
        ▼
Question embedded → FAISS finds top 3 closest chunks
        |
        ▼
Retrieved chunks joined as context string
        |
        ▼
RAG chain invoked:
  rag_template | llm | parser
        |
        ▼
Llama answers using ONLY retrieved context
```

---

## Project Structure

```
langchain-rag/
├── app.py        ← Flask server, LangChain RAG pipeline, FAISS
├── index.html    ← Dark themed chat UI with document input
├── .env          ← API key (not pushed to GitHub)
└── README.md
```

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/NAjamsher/gen-ai-projects
cd gen-ai-projects/langchain-rag

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

- `FAISS.from_texts(chunks, embeddings)` handles embedding, indexing, and storage in one call
- `similarity_search(query, k=3)` retrieves the 3 most semantically relevant chunks
- RAG prompt explicitly instructs LLM: "Answer only from the context below. If not found, say so."
- `temperature=0.1` for factual, grounded responses — prevents creative hallucination
- Vector store is rebuilt in memory for each new document — no cross-document contamination
- This is an upgrade from the basic RAG pipeline (Day 18) — rebuilt using LangChain components

---

## The 5 Core RAG Lines

```python
embeddings   = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_texts(chunks, embeddings)
docs         = vector_store.similarity_search(query, k=3)
context      = "\n\n".join([doc.page_content for doc in docs])
answer       = rag_chain.invoke({"context": context, "question": query})
```

---

## Difference From Basic RAG (Day 18)

| Feature | Day 18 Basic RAG | LangChain RAG (This) |
|---|---|---|
| Embedding | Manual SentenceTransformer | HuggingFaceEmbeddings |
| FAISS setup | Manual float32 conversion | FAISS.from_texts() — 1 line |
| Retrieval | Manual index.search() | similarity_search() |
| Chain | Direct function calls | LCEL pipe: template \| llm \| parser |
| Code length | ~80 lines | ~40 lines |

---

## Environment Variables

| Variable | Description |
|---|---|
| HUGGINGFACE_API_KEY | HuggingFace user access token |

---

## License

MIT