# Basic RAG Pipeline — paste any document, ask unlimited questions
# Day 18 | Pillar: RAG | Concept: Auto chunking, FAISS, Grounded QA
# Upgrade from Day 17 — automatic document chunking + clean single flow

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import faiss
import numpy as np
import os

# ── CONFIGURE ─────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("HUGGINGFACE_API_KEY")
print(f"API KEY LOADED: {API_KEY}")

# ── SETUP ─────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

client = InferenceClient(provider="novita", api_key=API_KEY)

print("Loading embedding model...")
embedder  = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model ready.")

DIMENSION = 384

# ── IN-MEMORY STORE — resets when new document is loaded ──────
index     = faiss.IndexFlatL2(DIMENSION)
chunks    = []

# ── STEP 1 — CHUNK THE DOCUMENT ───────────────────────────────
def chunk_document(text, chunk_size=3):
    # split into sentences by period
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]

    # group sentences into chunks of chunk_size
    document_chunks = []
    for i in range(0, len(sentences), chunk_size):
        group = sentences[i : i + chunk_size]
        chunk = ". ".join(group) + "."
        document_chunks.append(chunk)

    return document_chunks

# ── STEP 2 — EMBED AND STORE IN FAISS ─────────────────────────
def build_index(document_chunks):
    global index, chunks

    # reset index and chunks for new document
    index  = faiss.IndexFlatL2(DIMENSION)
    chunks = []

    if not document_chunks:
        return 0

    # embed all chunks
    embeddings = embedder.encode(document_chunks)
    embeddings = np.array(embeddings).astype("float32")

    # add to FAISS
    index.add(embeddings)
    chunks = document_chunks

    print(f"Index built with {len(chunks)} chunks.")
    return len(chunks)

# ── STEP 3 — RETRIEVE RELEVANT CHUNKS ─────────────────────────
def retrieve(query, top_k=3):
    if index.ntotal == 0:
        return []

    query_embedding = embedder.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        results.append({
            "text"    : chunks[idx],
            "distance": float(distances[0][i]),
            "rank"    : i + 1
        })

    return results

# ── STEP 4 — GENERATE GROUNDED ANSWER ─────────────────────────
def generate(query, context_chunks):
    context = "\n\n".join([c["text"] for c in context_chunks])

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context say exactly: I could not find the answer in the provided document.
Do not use any outside knowledge. Only use what is in the context.

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.1
    )
    return response.choices[0].message.content.strip()

# ── ROUTES ────────────────────────────────────────────────────
@app.route("/")
def home():
    return send_from_directory(os.getcwd(), "index.html")

# load document — chunk it — build FAISS index
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
        count           = build_index(document_chunks)
        return jsonify({
            "message"     : f"Document loaded. Created {count} chunks.",
            "chunk_count" : count,
            "chunks"      : document_chunks
        })
    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({"error": str(e)})

# ask a question — retrieve + generate
@app.route("/ask", methods=["POST"])
def ask():
    data  = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Please enter a question."})
    if index.ntotal == 0:
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

# get current index stats
@app.route("/stats", methods=["GET"])
def stats():
    return jsonify({
        "chunks_loaded": index.ntotal,
        "ready"        : index.ntotal > 0
    })

# clear index
@app.route("/clear", methods=["POST"])
def clear():
    global index, chunks
    index  = faiss.IndexFlatL2(DIMENSION)
    chunks = []
    return jsonify({"message": "Index cleared. Load a new document."})

if __name__ == "__main__":
    app.run(debug=True, port=5001)