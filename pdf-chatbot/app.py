# PDF Chatbot — upload any PDF, ask unlimited questions
# Day 20 | Pillar: RAG | Concept: PDF ingestion, text extraction, LangChain RAG
# Upgrade from Day 19 — reads real PDF files instead of pasted text

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import fitz  # PyMuPDF
import os
from dotenv import load_dotenv
from pathlib import Path
# ── CONFIGURE ─────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parent / ".env")
API_KEY = os.getenv("HUGGINGFACE_API_KEY")
print(f"API KEY LOADED: {API_KEY}")

# ── SETUP ─────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

client = InferenceClient(provider="novita", api_key=API_KEY)

# ── UPLOAD FOLDER ─────────────────────────────────────────────
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── LANGCHAIN EMBEDDING MODEL ─────────────────────────────────
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Embedding model ready.")

# ── GLOBAL STATE ──────────────────────────────────────────────
vector_store  = None
pdf_name      = None
total_chunks  = 0

# ── CORE LLM FUNCTION ─────────────────────────────────────────
def call_llama(prompt_value, temp=0.1, tokens=400):
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

# ── RAG PROMPT ────────────────────────────────────────────────
rag_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant answering questions about a PDF document.
Answer using ONLY the context provided below.
If the answer is not in the context say: I could not find the answer in the provided PDF.
Do not use outside knowledge.

Context:
{context}

Question: {question}

Answer:"""
)

rag_chain = rag_template | llm | parser

# ── STEP 1 — EXTRACT TEXT FROM PDF ────────────────────────────
def extract_text_from_pdf(filepath):
    doc   = fitz.open(filepath)
    text  = ""
    pages = []

    for page_num in range(len(doc)):
        page      = doc[page_num]
        page_text = page.get_text()
        text     += page_text
        pages.append({
            "page"  : page_num + 1,
            "text"  : page_text,
            "chars" : len(page_text)
        })

    doc.close()
    return text, pages

# ── STEP 2 — CHUNK TEXT ───────────────────────────────────────
def chunk_text(text, chunk_size=3):
    # clean text
    text      = " ".join(text.split())
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 40]

    chunks = []
    for i in range(0, len(sentences), chunk_size):
        group = sentences[i : i + chunk_size]
        chunk = ". ".join(group) + "."
        if len(chunk.split()) > 15:
            chunks.append(chunk)

    return chunks

# ── STEP 3 — BUILD VECTOR STORE ───────────────────────────────
def build_vector_store(chunks):
    global vector_store, total_chunks
    vector_store = FAISS.from_texts(chunks, embeddings)
    total_chunks = len(chunks)
    print(f"Vector store built with {total_chunks} chunks.")
    return total_chunks

# ── STEP 4 — RETRIEVE ─────────────────────────────────────────
def retrieve(query, top_k=3):
    if vector_store is None:
        return []
    docs = vector_store.similarity_search(query, k=top_k)
    return [{"text": doc.page_content, "rank": i + 1} for i, doc in enumerate(docs)]

# ── STEP 5 — GENERATE ─────────────────────────────────────────
def generate(query, retrieved_chunks):
    context = "\n\n".join([c["text"] for c in retrieved_chunks])
    return rag_chain.invoke({"context": context, "question": query})

# ── ROUTES ────────────────────────────────────────────────────
@app.route("/")
def home():
    return send_from_directory(os.getcwd(), "index.html")

# upload and process PDF
@app.route("/upload", methods=["POST"])
def upload():
    global pdf_name, vector_store, total_chunks

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."})

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."})

    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported."})

    try:
        # save uploaded PDF
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        pdf_name = file.filename

        # extract text from PDF
        text, pages = extract_text_from_pdf(filepath)

        if len(text.strip()) < 50:
            return jsonify({"error": "PDF appears to be empty or scanned. Text extraction failed."})

        # chunk the text
        chunks = chunk_text(text)

        if not chunks:
            return jsonify({"error": "Could not extract enough text to create chunks."})

        # build vector store
        count = build_vector_store(chunks)

        return jsonify({
            "message"     : f"PDF loaded successfully.",
            "filename"    : pdf_name,
            "pages"       : len(pages),
            "chunk_count" : count,
            "char_count"  : len(text),
            "word_count"  : len(text.split())
        })

    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({"error": str(e)})

# ask a question
@app.route("/ask", methods=["POST"])
def ask():
    data  = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Please enter a question."})

    if vector_store is None:
        return jsonify({"error": "No PDF loaded. Upload a PDF first."})

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

# get current stats
@app.route("/stats", methods=["GET"])
def stats():
    return jsonify({
        "loaded"  : vector_store is not None,
        "filename": pdf_name,
        "chunks"  : total_chunks
    })

# clear everything
@app.route("/clear", methods=["POST"])
def clear():
    global vector_store, pdf_name, total_chunks
    vector_store = None
    pdf_name     = None
    total_chunks = 0
    return jsonify({"message": "Cleared. Upload a new PDF."})

if __name__ == "__main__":
    app.run(debug=True, port=5001)