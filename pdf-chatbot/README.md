AskRAG
A Retrieval Augmented Generation system built from scratch — paste any document, ask unlimited questions, get grounded answers with source citations.

What it does

Paste any document — article, research paper, notes, reports
System automatically chunks the document into searchable pieces
Ask unlimited questions — answers come only from your document
Every answer shows the exact source chunks used to generate it
Hallucination prevention built into the prompt — AI cannot make things up


Tech Stack
LayerTechnologyBackendPython, FlaskAI ModelLlama 3.1 8B InstructProviderNovita via HuggingFaceEmbeddingssentence-transformers/all-MiniLM-L6-v2Vector StoreFAISSFrameworkLangChainFrontendHTML, CSS, JavaScriptAuthpython-dotenv

How it works
User pastes document
      |
      ▼
Text split into sentence-based chunks (3 sentences per chunk)
      |
      ▼
Each chunk converted to 384-dimensional vector
      |
      ▼
All vectors stored in FAISS index
      |
      ▼
User asks question
      |
      ▼
Question converted to vector
      |
      ▼
FAISS finds top 3 closest chunks by cosine similarity
      |
      ▼
Retrieved chunks sent to Llama as context
      |
      ▼
Llama answers using ONLY the retrieved context
      |
      ▼
Answer + source chunks returned to user

Project Structure
askrag/
├── app.py        ← Flask server, RAG pipeline, FAISS logic
├── index.html    ← Dark themed chat UI with source display
├── .env          ← API key (not pushed to GitHub)
└── README.md

Getting Started
bash# Clone the repository
git clone https://github.com/NAjamsher/gen-ai-projects
cd gen-ai-projects/askrag

# Install dependencies
pip install flask flask-cors huggingface-hub langchain-core
pip install langchain-community langchain-huggingface
pip install faiss-cpu sentence-transformers python-dotenv

# Add your API key
echo "HUGGINGFACE_API_KEY=your_key_here" > .env

# Run the app
python app.py
Open browser at http://127.0.0.1:5000

Key Implementation Details

FAISS.from_texts() handles embedding, indexing, and storage in one call
similarity_search(query, k=3) retrieves the 3 most semantically relevant chunks
Prompt strictly instructs the LLM to answer only from context — prevents hallucination
Temperature set to 0.1 for factual, grounded responses
In-memory vector store resets when a new document is loaded


Environment Variables
VariableDescriptionHUGGINGFACE_API_KEYHuggingFace user access token

License
MIT