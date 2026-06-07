# LangChain RAG — Day 18 rebuilt using LangChain components
# Day 19 | Pillar: RAG | Concept: LangChain FAISS, HuggingFaceEmbeddings, RetrievalQA
# Same pipeline as Day 18 — LangChain handles embedding, indexing, retrieval

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import os

# ── CONFIGURE ─────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("HUGGINGFACE_API_KEY")
print(f"API KEY LOADED: {API_KEY}")

# ── SETUP ─────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

client = InferenceClient(provider="novita", api_key=API_KEY)

# ── LANGCHAIN EMBEDDING MODEL ─────────────────────────────────
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Embedding model ready.")

# ── GLOBAL VECTOR STORE — resets per document ─────────────────
vector_store = None

# ── CORE LLM FUNCTION ─────────────────────────────────────────
def call_llama(prompt_value, temp=0.1, tokens=300):
    prompt_text = prompt_value.text if hasattr(prompt_value, "text") else str(prompt_value)
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": prompt_text}],
        max_tokens=tokens,
        temperature=temp
    )
    content = response.choices[0].message.content
    if isinstance(content, list):
        return content[0]["text"]
    return str(content).strip()

# ── LANGCHAIN COMPONENTS ───────────────────────────────────────
llm    = RunnableLambda(call_llama)
parser = StrOutputParser()

# ── RAG PROMPT TEMPLATE ───────────────────────────────────────
rag_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context say exactly: I could not find the answer in the provided document.
Do not use any outside knowledge.

Context:
{context}

Question: {question}

Answer:"""
)

rag_chain = rag_template | llm | parser

# ── STEP 1 — CHUNK DOCUMENT ───────────────────────────────────
def chunk_document(text, chunk_size=3):
    sentences       = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    document_chunks = []
    for i in range(0, len(sentences), chunk_size):
        group = sentences[i: i + chunk_size]
        chunk = ". ".join(group) + "."
        document_chunks.append(chunk)
    return document_chunks

# ── STEP 2 — BUILD LANGCHAIN FAISS VECTOR STORE ───────────────
def build_vector_store(document_chunks):
    global vector_store
    # LangChain FAISS — embeds and indexes in one call
    vector_store = FAISS.from_texts(document_chunks, embeddings)
    print(f"Vector store built with {len(document_chunks)} chunks.")
    return len(document_chunks)

# ── STEP 3 — RETRIEVE ─────────────────────────────────────────
def retrieve(query, top_k=3):
    if vector_store is None:
        return []
    # LangChain similarity search — returns Document objects
    docs = vector_store.similarity_search(query, k=top_k)
    return [{"text": doc.page_content, "rank": i + 1} for i, doc in enumerate(docs)]

# ── STEP 4 — GENERATE ─────────────────────────────────────────
def generate(query, retrieved_chunks):
    context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])
    answer  = rag_chain.invoke({
        "context" : context,
        "question": query
    })
    return answer

# ── ROUTES ────────────────────────────────────────────────────
@app.route("/")
def home():
    return send_from_directory(os.getcwd(), "index.html")

@app.route("/load", methods=["POST"])
def load():
    data     = request.get_json()
    document = data.get("document", "").strip()
    if not document:
        return jsonify({"error": "Please paste a document."})
    if len(document.split()) < 20:
        return jsonify({"error": "Document too short. Paste at least a paragraph."})
    try:
        document_chunks = chunk_document(document)
        count           = build_vector_store(document_chunks)
        return jsonify({
            "message"    : f"Document loaded. Created {count} chunks.",
            "chunk_count": count,
            "chunks"     : document_chunks
        })
    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({"error": str(e)})

@app.route("/ask", methods=["POST"])
def ask():
    data  = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Please enter a question."})
    if vector_store is None:
        return jsonify({"error": "No document loaded yet. Paste a document first."})
    try:
        retrieved = retrieve(query, top_k=3)
        answer    = generate(query, retrieved)
        return jsonify({
            "answer" : answer,
            "sources": retrieved
        })
    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({"error": str(e)})

@app.route("/clear", methods=["POST"])
def clear():
    global vector_store
    vector_store = None
    return jsonify({"message": "Vector store cleared."})

@app.route("/stats", methods=["GET"])
def stats():
    return jsonify({
        "loaded": vector_store is not None,
        "chunks": vector_store.index.ntotal if vector_store else 0
    })

if __name__ == "__main__":
    app.run(debug=True, port=5001)