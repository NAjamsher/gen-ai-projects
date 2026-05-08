# LangChain Summarizer

Day 3 Summarizer rebuilt using LangChain. Same result, cleaner code.
Powered by Llama 3.1 via HuggingFace Inference API.

---

## What it does

- Paste any text of 30 words or more
- LangChain PromptTemplate builds the prompt
- RunnableLambda calls Llama 3.1 via InferenceClient
- StrOutputParser cleans the output
- Returns summary with word count and reduction percentage

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Framework | LangChain |
| AI Model | Llama 3.1 8B Instruct via HuggingFace |
| Provider | Novita via HuggingFace router |
| Frontend | HTML, CSS, JavaScript |
| Auth | python-dotenv |

---

## LangChain Components Used

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# chain
summarize_chain = summarize_template | llm | StrOutputParser()

# run
summary = summarize_chain.invoke({"text": text})
```

---

## Project Structure

```
langchain-intro/
├── app.py        — Flask backend, LangChain chain, summarization logic
├── index.html    — Textarea input, summary card, stat display
├── .env          — API key (not committed)
└── README.md
```

---

## Getting Started

**1. Clone the repo**

```bash
git clone https://github.com/NAjamsher/gen-ai-projects.git
cd gen-ai-projects/langchain-intro
```

**2. Install dependencies**

```bash
pip install flask flask-cors huggingface_hub langchain langchain-core langchain-huggingface python-dotenv
```

**3. Set up environment variables**

Create a `.env` file inside the `langchain-intro/` folder:

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

**PromptTemplate**

Reusable prompt with named variables. Define once, use with any input.
LangChain validates variables before running the chain.

**RunnableLambda**

Wraps the proven InferenceClient call into a LangChain Runnable
so it participates in the pipe chain using the `|` operator.

**LCEL Pipe Operator**

```
PromptTemplate | RunnableLambda | StrOutputParser
```

Left side output flows into right side input automatically.

---

## Environment Variables

| Variable | Description |
|---|---|
| HUGGINGFACE_API_KEY | HuggingFace user access token |

---

## License

MIT