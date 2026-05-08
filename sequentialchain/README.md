# News Analyzer

Paste any news article and get a summary, sentiment analysis,
and a suggested tweet — processed through a three-step sequential chain.
Built with LangChain and Llama 3.1 via HuggingFace.

---

## What it does

- Paste any news article of 30 words or more
- Step 1 summarizes the article into 2-3 sentences
- Step 2 analyses the sentiment of the summary
- Step 3 writes a tweet using both the summary and sentiment
- Each step's output feeds into the next step
- Animated pipeline shows which step is running

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

## Sequential Chain Flow

```
Article
    ↓
summarize_chain (temp=0.3)  →  Summary
    ↓
sentiment_chain (temp=0.1)  →  Sentiment (one word)
    ↓
tweet_chain (summary + sentiment, temp=0.7)  →  Tweet
```

Each step receives the output of the previous step as input.

---

## LangChain Components Used

```python
# step 1
summary = summarize_chain.invoke({"article": article})

# step 2 — receives step 1 output
sentiment = sentiment_chain.invoke({"summary": summary})

# step 3 — receives both step 1 and step 2 outputs
tweet = tweet_chain.invoke({
    "summary"  : summary,
    "sentiment": sentiment
})
```

---

## Project Structure

```
sequentialchain/
├── app.py        — Flask backend, three-step sequential pipeline
├── index.html    — Article input, animated pipeline, three result cards
├── .env          — API key (not committed)
└── README.md
```

---

## Getting Started

**1. Clone the repo**

```bash
git clone https://github.com/NAjamsher/gen-ai-projects.git
cd gen-ai-projects/sequentialchain
```

**2. Install dependencies**

```bash
pip install flask flask-cors huggingface_hub langchain langchain-core python-dotenv
```

**3. Set up environment variables**

Create a `.env` file inside the `sequentialchain/` folder:

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

**Sequential chaining**

Output of each step becomes input of the next.
The tweet chain receives both the summary and sentiment
so it can write a contextually appropriate, tone-aware tweet.

**Multiple input variables**

```python
tweet_template = PromptTemplate(
    input_variables=["summary", "sentiment"],
    template="Summary: {summary} Sentiment: {sentiment} Tweet:"
)
```

One template taking two variables from two different previous steps.

**Lambda functions for temperature control**

```python
sentiment_chain = sentiment_template | RunnableLambda(
    lambda x: call_llama(x, temp=0.1, tokens=10)
) | parser
```

Each step gets its own temperature — precise for sentiment,
creative for tweet generation.

---

## Environment Variables

| Variable | Description |
|---|---|
| HUGGINGFACE_API_KEY | HuggingFace user access token |

---

## License

MIT