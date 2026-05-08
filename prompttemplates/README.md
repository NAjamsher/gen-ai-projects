# Content Generator

Type one topic — get a blog introduction, tweet, and tagline instantly.
Powered by LangChain PromptTemplates and Llama 3.1 via HuggingFace.

---

## What it does

- Type any topic in the input box
- Three independent LangChain chains run simultaneously
- Each chain uses a different PromptTemplate with different instructions
- Returns blog intro, tweet under 280 characters, and a tagline
- Copy button on each result card

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
# three templates — same variable, different instructions
blog_template    = PromptTemplate(input_variables=["topic"], template="...")
tweet_template   = PromptTemplate(input_variables=["topic"], template="...")
tagline_template = PromptTemplate(input_variables=["topic"], template="...")

# three independent chains
blog_chain    = blog_template    | llm | parser
tweet_chain   = tweet_template   | llm | parser
tagline_chain = tagline_template | llm | parser

# run all three
blog    = blog_chain.invoke({"topic": topic})
tweet   = tweet_chain.invoke({"topic": topic})
tagline = tagline_chain.invoke({"topic": topic})
```

---

## Project Structure

```
prompttemplates/
├── app.py        — Flask backend, three chains, content generation logic
├── index.html    — Topic input, three result cards, copy buttons
├── .env          — API key (not committed)
└── README.md
```

---

## Getting Started

**1. Clone the repo**

```bash
git clone https://github.com/NAjamsher/gen-ai-projects.git
cd gen-ai-projects/prompttemplates
```

**2. Install dependencies**

```bash
pip install flask flask-cors huggingface_hub langchain langchain-core python-dotenv
```

**3. Set up environment variables**

Create a `.env` file inside the `prompttemplates/` folder:

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

**Multiple PromptTemplates**

Three separate templates all taking the same `{topic}` variable
but with completely different instructions. Same model, same input,
three different outputs — because the prompt controls everything.

**Independent chains**

All three chains use the same `llm` and `parser` objects.
Only the template changes. Swap the template, get a different output.
One AI connection serving three different tasks.

**Temperature per task**

Blog and tweet use temperature 0.7 for creativity.
Tagline uses 0.5 for balance between catchy and concise.

---

## Environment Variables

| Variable | Description |
|---|---|
| HUGGINGFACE_API_KEY | HuggingFace user access token |

---

## License

MIT