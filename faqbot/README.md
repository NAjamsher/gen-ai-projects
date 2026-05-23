FAQ Bot
An intelligent customer support bot that answers questions semantically — not by keyword matching — powered by RAG, LangChain, and Llama 3.1.

What it does

15 business FAQs embedded into a FAISS vector store at startup
Customers ask questions in natural language — any phrasing works
Semantic search finds the most relevant FAQ even without exact word match
Llama generates a friendly, professional answer in customer support tone
Full FAQ list browsable directly on the page
Zero keyword matching — pure meaning-based retrieval


Tech Stack
LayerTechnologyBackendPython, FlaskAI ModelLlama 3.1 8B InstructProviderNovita via HuggingFaceEmbeddingssentence-transformers/all-MiniLM-L6-v2Vector StoreFAISSFrameworkLangChainFrontendHTML, CSS, JavaScriptAuthpython-dotenv

How it works
App starts
      |
      ▼
15 FAQ Q+A pairs combined into single searchable chunks
      |
      ▼
All chunks embedded and indexed into FAISS at startup
      |
      ▼
Customer types question in any phrasing
      |
      ▼
Question embedded → FAISS finds 2 most relevant FAQ chunks
      |
      ▼
Retrieved chunks sent to Llama with customer support prompt
      |
      ▼
Llama answers in friendly, professional tone
      |
      ▼
Answer returned to customer instantly

Project Structure
faqbot/
├── app.py        ← Flask server, FAQ knowledge base, RAG pipeline
├── index.html    ← Dark themed chat UI with FAQ browser
├── .env          ← API key (not pushed to GitHub)
└── README.md

Getting Started
bash# Clone the repository
git clone https://github.com/NAjamsher/gen-ai-projects
cd gen-ai-projects/faqbot

# Install dependencies
pip install flask flask-cors huggingface-hub langchain-core
pip install langchain-community langchain-huggingface
pip install faiss-cpu sentence-transformers python-dotenv

# Add your API key
echo "HUGGINGFACE_API_KEY=your_key_here" > .env

# Run the app
python app.py
Open browser at http://127.0.0.1:5001

Key Implementation Details

Question and answer combined into one chunk — FAISS matches on both question phrasing and answer content
Vector store built once at startup — FAQ data is static, no per-request indexing
top_k set to 2 — FAQ answers are specific, fewer chunks gives cleaner context
Temperature 0.3 — warmer than factual tasks, gives natural customer support tone
/faqs GET route returns full FAQ list for frontend display
Semantic matching means "how do I return something" correctly matches "return policy" FAQ


FAQ Categories Covered

Business hours and availability
Returns, exchanges, and refunds
Shipping — domestic and international
Payment methods and security
Order tracking and cancellation
Customer support contact details
Loyalty program and discounts
Data privacy and security


Environment Variables
VariableDescriptionHUGGINGFACE_API_KEYHuggingFace user access token

License
MIT