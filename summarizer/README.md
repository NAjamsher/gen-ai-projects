# Summarizer

Paste any long text and get a clean 2-3 sentence summary instantly.
Powered by Llama 3.1 via HuggingFace API.

---

## What this project does

- You paste any long paragraph or article into the browser
- The app sends it to a Flask backend
- Flask builds a prompt and sends it to Llama 3.1 AI
- Llama reads the text and returns a clean 2-3 sentence summary
- The browser displays the summary with original words, summary words, and reduction percentage

---

## How it looks

- Purple summary card with the condensed text
- 3 stat cards below — original words, summary words, reduced by %
- Live word counter that turns purple when you hit 30 words
- 3 example buttons — Climate Change, AI, Healthcare

---

## The core concept — Prompt Engineering

This is the most important concept in this project.

Instead of just sending raw text to the AI, you wrap it with
clear instructions that tell the AI exactly what to do and
how to format the output.

```python
prompt = f"""Please summarize the following text in 2-3 clear sentences.
Only give the summary, nothing else.

Text to summarize:
{text}

Summary:"""
```

The quality of your prompt = the quality of the AI output.
This skill is called prompt engineering and it is one of the
most in-demand skills in the GenAI industry right now.

---

## System message vs Prompt — key concept

This project uses both and they serve different purposes.

```python
# system message — defines WHO the AI is, set once, never changes
{"role": "system", "content": "You are a summarization assistant. 
You only output clean concise summaries. Never explain yourself."}

# prompt — defines WHAT to do and HOW to format output, changes per request
prompt = f"""Summarize in 2-3 sentences. Only give the summary.
Text: {text}
Summary:"""
```

### The rule to remember

| | System Message | Prompt |
|---|---|---|
| Purpose | AI personality | Task + output format |
| Changes? | Never | Every request |
| Controls | Who the AI is | What the AI outputs |
| Best for | Tone, language, restrictions | Format, structure, specific instructions |

### Best practice

Always put output customization in the prompt.
Keep system message to one or two lines describing who the AI is.

```python
# correct
system  →  "You are a summarization assistant. Be concise."
prompt  →  "Summarize in 2 sentences. Add a relevant emoji at the end."

# wrong
system  →  "Add emoji first, then summary, then joke at the end"
prompt  →  "Summarize this: {text}"
```

---

## API approach vs Local pipeline — key concept

This project uses the API approach, not the local pipeline approach.
Understanding the difference is important.

### Local pipeline (not used here)
```python
from transformers import pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
result = summarizer(text)
```
- Model downloads to your laptop (500MB - 2GB)
- Your laptop CPU runs the model
- No internet needed after download
- Slow on laptops without GPU
- Best for: private data, offline use

### API approach (used here)
```python
from huggingface_hub import InferenceClient
client = InferenceClient(provider="novita", api_key=API_KEY)
result = client.chat.completions.create(...)
```
- Model runs on HuggingFace servers
- Your laptop just sends and receives text
- Internet required
- Fast — their powerful GPUs do the work
- Best for: web apps, prototyping, learning

### When to use which
Private/sensitive data    →  local pipeline
Building a web app        →  API approach
No internet available     →  local pipeline
Best model quality        →  API approach
Learning and prototyping  →  always API approach

---

## temperature=0.3 — why so low?

```python
temperature=0.3   # used in this project
temperature=0.7   # used in TextSpark Day 1
```

Temperature controls how creative or random the AI output is.

- `0.1` — very precise, almost robotic, same answer every time
- `0.7` — balanced, some creativity, slight variation
- `1.0` — very creative, different answer every time

Summarization needs accuracy and faithfulness to the original text.
You do not want the AI to be creative — you want it to stick to the facts.
That is why temperature is set low at 0.3.

Rule to remember:
Factual tasks  →  low temperature  (0.1 - 0.4)
Creative tasks →  high temperature (0.6 - 1.0)

---

## How the word reduction is calculated

```python
word_count    = len(text.split())
summary_words = len(summary.split())
reduced_by    = round((1 - summary_words / word_count) * 100)
```

`text.split()` — splits the string into a list of individual words.
"I love coding" becomes ["I", "love", "coding"]

`len(...)` — counts items in a list. len(["I", "love", "coding"]) = 3

`1 - summary_words / word_count` — the reduction ratio.
If original = 200 words and summary = 50 words:
1 - 50/200 = 1 - 0.25 = 0.75 = 75% reduction

`round(...) * 100` — converts 0.75 to 75. round() removes decimals.

---

## New Python concept — f-string with triple quotes

```python
prompt = f"""Please summarize the following text.
Text: {text}
Summary:"""
```

Triple quotes `"""` let you write strings across multiple lines.
The `f` prefix lets you embed variables directly using `{variable}`.
Both combined let you build clean multi-line prompts with dynamic content.

---

## Flow — what happens when you click Summarize
User pastes text → clicks Summarize
↓
Browser sends POST to /summarize with the text as JSON
↓
Flask reads the text from request.get_json()
↓
Checks word count — less than 30? return error immediately
↓
Builds prompt with instructions + text
↓
Sends prompt to Llama 3.1 via InferenceClient
↓
Llama reads system message + prompt → writes summary
↓
Flask calculates word counts and reduction percentage
↓
Returns dictionary — summary, original_words, summary_words, reduced_by
↓
Browser receives JSON → displays summary card + 3 stat cards

---

## Concepts learned in this project

### Prompt engineering
Wrapping your text with clear instructions to control AI output.
The structure, wording, and specificity of your prompt directly
determines the quality of the AI response. This is a real job skill —
companies hire people specifically to write and optimize prompts.

### System message vs prompt separation
System message defines the AI personality — set once, permanent.
Prompt defines the task and output format — changes every request.
Keeping these two separate is the professional way to build AI apps.

### Temperature selection
Choosing the right temperature based on the task type.
Low for factual, high for creative. This decision affects every
AI project you build.

### Dynamic prompt building
Using f-strings to inject user input into a structured prompt template.
This is how every real AI app works — a template with variables
filled in at runtime.

### Input validation
Checking word count before sending to AI. Always validate user input
before processing — prevents unnecessary API calls and bad outputs.

### Post processing AI output
Taking the raw AI response and calculating additional stats
(word count, reduction %) before sending to the browser.
Real apps always process AI output before showing it to users.

---

## Difference from previous projects

| | TextSpark | MoodDetector | Summarizer |
|---|---|---|---|
| AI task | Text generation | Classification | Summarization |
| Model location | HF servers | Local laptop | HF servers |
| Temperature | 0.7 | Not applicable | 0.3 |
| Key concept | API calls | Pipelines | Prompt engineering |
| Output | Generated text | Label + score | Condensed text + stats |
| System message | No | No | Yes |

---

## Files in this project
summarizer/
├── app.py        → Flask backend, prompt engineering, Llama API call
├── index.html    → Browser frontend, textarea, summary card, stat cards
└── README.md     → This file

---

## How to run

Step 1 — install dependencies
```bash
pip install flask flask-cors huggingface_hub
```

Step 2 — add your HuggingFace API key in app.py
```python
API_KEY = "your_huggingface_api_key_here"
```

Step 3 — run the server
```bash
cd summarizer
python app.py
```

Step 4 — open in browser
http://127.0.0.1:5000

---

## What to remember

- Prompt engineering is the most important skill in GenAI
- System message = AI personality, Prompt = output format
- Always put output customization in the prompt not the system message
- Low temperature for factual tasks, high for creative tasks
- API approach is better than local pipeline for web apps
- Always validate input before sending to the AI
- Post process AI output before showing it to users
