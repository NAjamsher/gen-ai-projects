markdown# MoodDetector

Detects the emotion behind any sentence — Positive or Negative —
with a confidence score showing how sure the AI is.
Powered by DistilBERT running locally via HuggingFace Transformers.

---

## What this project does

- You type any sentence in the browser
- The app sends it to a Flask backend
- Flask runs a sentiment analysis model locally on your laptop
- The model classifies the sentence as POSITIVE or NEGATIVE
- Returns a confidence score showing how sure the model is
- The browser displays the result with color, emoji, and animated confidence bar

---

## How it looks

- Green card + 😊 = Positive mood
- Red card + 😔 = Negative mood
- Confidence bar fills up based on how sure the model is
- 6 example chips to test quickly without typing

---

## The core concept — Sentiment Analysis

Sentiment Analysis is reading a piece of text and classifying
the emotion behind it. It is one of the most widely used AI
tasks in the real world.

Real world usage:
- Amazon classifies product reviews as positive or negative
- Twitter tracks public mood on trending topics
- Companies run sentiment on customer support tickets
- Stock trading apps analyze news headlines to predict market mood
- Netflix measures audience reaction to shows

---

## Classification vs Generation — most important concept

This is the biggest difference between Day 1 and Day 2.

### TextSpark — Generation
```python
# model creates brand new text
response = client.chat.completions.create(...)
# output → "Artificial intelligence is the simulation of..."
```

The model reads your prompt and **creates** something new.
Like asking someone to write an essay — they produce new content.

### MoodDetector — Classification
```python
# model puts text into a category
result = sentiment_pipeline("I love this product!")[0]
# output → {"label": "POSITIVE", "score": 0.9998}
```

The model reads your text and **categorizes** it.
Like asking someone to sort letters into boxes — they don't create,
they decide which box it belongs to.

### Rule to remember

```
Generation     →  model creates new text
Classification →  model puts text into a category
```

Most real AI products use classification far more than generation.
Spam filters, content moderation, fraud detection, disease diagnosis
— all classification.

---

## Local pipeline vs API call — key concept

MoodDetector runs the model locally on your laptop.
TextSpark called a remote API over the internet.

### How local pipeline works
```python
from transformers import pipeline

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
```

```
Your code
    ↓
Downloads DistilBERT model to your laptop (~250MB, one time)
    ↓
Your laptop CPU runs the model
    ↓
Result comes back instantly — no internet needed
```

### How API call works (TextSpark)
```python
from huggingface_hub import InferenceClient
client = InferenceClient(provider="novita", api_key=API_KEY)
```

```
Your code
    ↓
Sends text over internet to HuggingFace servers
    ↓
Their GPU runs the model
    ↓
Result comes back over internet
```

### Comparison

| | Local Pipeline | API Call |
|---|---|---|
| Internet needed | No | Yes |
| API key needed | No | Yes |
| Speed | Depends on your CPU | Fast (their GPU) |
| Model downloads | Yes, one time | Never |
| Your laptop load | High CPU usage | Almost zero |
| Works offline | Yes | No |
| Best for | Private data, offline | Web apps, prototyping |

---

## The model — DistilBERT

```
distilbert-base-uncased-finetuned-sst-2-english
```

Breaking down the name:

`distilbert` — a lightweight version of BERT made by Google.
40% smaller and 60% faster than BERT while keeping 97% accuracy.

`base` — medium sized version. Not the smallest, not the largest.

`uncased` — treats uppercase and lowercase as the same.
"HAPPY" and "happy" are treated identically.

`finetuned` — this model was taken from a general base and trained
further on a specific task. Like a general doctor who did a
specialization in cardiology.

`sst-2` — Stanford Sentiment Treebank dataset. A collection of
movie reviews labeled as positive or negative. This is what the
model was fine-tuned on.

`english` — only works on English text.

### What fine-tuned means

Pre-trained = model learned general language from billions of words.
Fine-tuned = model was then trained specifically on sentiment data.

This is like hiring someone who already knows how to read and write
(pre-trained) and then training them specifically to be a film critic
(fine-tuned on SST-2).

---

## Confidence score — why it matters

The model never just says positive or negative.
It always returns a score between 0 and 1.

```python
result = sentiment_pipeline("I love this!")[0]
# {"label": "POSITIVE", "score": 0.9998}

result = sentiment_pipeline("It was okay")[0]
# {"label": "POSITIVE", "score": 0.5312}
```

`0.9998` = 99.98% sure → very confident, clear sentiment
`0.5312` = 53.12% sure → barely sure, almost a coin flip

### Why this matters in production

```
confidence > 0.90  →  show result confidently
confidence 0.60-0.90  →  show result with caution
confidence < 0.60  →  show "unclear" instead of guessing
```

In real apps you never blindly show the label.
You use the confidence score to decide how to present the result.
A medical sentiment app would need 0.95+ confidence before acting.

---

## How the result is processed

```python
result     = sentiment_pipeline(text)[0]
label      = result["label"]       # "POSITIVE" or "NEGATIVE"
score      = result["score"]       # 0.9998
confidence = round(score * 100, 2) # 99.98
```

`sentiment_pipeline(text)` — passes text to the model, returns a list.
`[0]` — gets the first result from the list.
`result["label"]` — extracts the classification label from the dict.
`result["score"]` — extracts the confidence score from the dict.
`round(score * 100, 2)` — converts 0.9998 to 99.98 for display.

---

## Input validation — always check before processing

```python
if not user_text.strip():
    return jsonify({"error": "Please enter some text"})
```

Never send empty or whitespace-only text to the model.
Always validate input before processing.
This prevents unnecessary model calls, crashes, and bad outputs.

Rule — validate first, process second. Always.

---

## Flow — what happens when you click Detect Mood

```
User types sentence → clicks Detect Mood
        ↓
Browser sends POST to /analyze with text as JSON
        ↓
Flask reads text from request.get_json()
        ↓
Checks if text is empty — yes? return error immediately
        ↓
Passes text to sentiment_pipeline
        ↓
DistilBERT model reads the sentence
        ↓
Returns label (POSITIVE/NEGATIVE) + score (0.0 to 1.0)
        ↓
Flask converts score to percentage, picks emoji
        ↓
Returns dictionary — mood, emoji, confidence, raw_label
        ↓
Browser receives JSON
        ↓
Sets card color (green/red), shows emoji, fills confidence bar
```

---

## JavaScript concepts used

### Dynamic CSS classes
```javascript
resultBox.className = `result ${mood}`
```
Changes the card color by swapping CSS classes at runtime.
CSS already defines what .positive and .negative look like.
JavaScript just switches between them based on the result.

### Animated confidence bar
```javascript
confBar.style.width = `${data.confidence}%`
```
Sets the bar width to the confidence percentage.
CSS transition property animates it smoothly over 0.6 seconds.
97% confidence = bar fills 97% of the way.

### this keyword
```javascript
onclick="tryExample(this)"
```
`this` refers to the element that was clicked.
Inside the function, `el.textContent` reads the chip text
and fills it into the textarea automatically.

### e.preventDefault() and e.shiftKey
```javascript
if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault()
    analyzeMood()
}
```
`preventDefault()` stops the browser's default Enter behavior
which would add a new line in the textarea.
`!e.shiftKey` means — only submit on plain Enter.
Shift+Enter still adds a new line normally.
This is the standard UX pattern used in WhatsApp, Slack, ChatGPT.

---

## Concepts learned in this project

### Sentiment Analysis
Reading text and classifying the emotion behind it.
One of the most used AI tasks in industry.
Used in reviews, social media, customer support, finance.

### Classification vs Generation
Two completely different AI tasks.
Classification puts text in a category.
Generation creates new text.
Know the difference — it comes up in every interview.

### Pre-trained and fine-tuned models
Pre-trained = learned general language.
Fine-tuned = specialized for a specific task.
You used a model fine-tuned on sentiment data — not a general model.

### Confidence scores
The model always returns how sure it is.
Use confidence to decide how to present results.
Never blindly show a label without checking the score.

### Local pipeline
Running a model directly on your laptop.
Model downloads once, runs offline forever after.
Higher privacy, no API costs, but needs CPU/RAM.

---

## Difference from previous projects

| | TextSpark | MoodDetector |
|---|---|---|
| AI task | Text generation | Sentiment classification |
| Model runs on | HuggingFace servers | Your laptop locally |
| Library used | huggingface_hub | transformers |
| API key needed | Yes | No |
| Internet needed | Yes | No (after download) |
| Output | New generated text | Label + confidence score |
| Key concept | API calls + client-server | Classification + local models |
| Temperature | 0.7 | Not applicable |

---

## Files in this project

```
mooddetector/
├── app.py        → Flask backend, loads DistilBERT, runs classification
├── index.html    → Browser frontend, color cards, confidence bar, chips
└── README.md     → This file
```

---

## How to run

Step 1 — install dependencies
```bash
pip install flask flask-cors transformers torch
```

Step 2 — run the server
```bash
cd mooddetector
python app.py
```

First run downloads DistilBERT model — about 30 seconds.
You will see "Model loaded and ready!" when done.
After first run it is cached — instant startup every time.

Step 3 — open in browser
```
http://127.0.0.1:5000
```

---

## What to remember

- Classification and generation are two different AI tasks
- Confidence score is as important as the label itself
- Local pipeline = model on your laptop, API = model on their server
- Fine-tuned models outperform general models on specific tasks
- Always validate user input before sending to the model
- Use confidence threshold in production — never blindly trust the label
- The transformers pipeline() works the same way for any task —
  just change the task name and model