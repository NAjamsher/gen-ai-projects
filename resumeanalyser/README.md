# ResumeAnalyser

Paste a job description and a resume — get a match score, missing skills, and improvement suggestions.
Powered by Llama 3.1 via HuggingFace.

---

## What it does

- Compares any resume against any job description
- Returns a match score from 0 to 100
- Lists skills present in both the JD and resume
- Lists skills required by JD but missing from resume
- Gives actionable suggestions to improve the resume for that role
- Match score color changes — green above 70, yellow 40-70, red below 40

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| AI Model | Llama 3.1 8B Instruct |
| Provider | Novita via HuggingFace |
| Framework | LangChain (PromptTemplate, RunnableLambda) |
| Frontend | HTML, CSS, JavaScript |
| Auth | python-dotenv |

---

## How it works
Browser sends JD + Resume
↓
Flask reads both inputs
↓
Multi-input PromptTemplate fills {job_description} and {resume}
↓
LangChain chain invoked with two variables
↓
Llama 3.1 returns JSON — score, matched, missing, suggestions
↓
Flask cleans and parses JSON
↓
Browser renders score with dynamic color and skill tags

---

## Project Structure
resumeanalyser/
├── app.py          — Flask server, LangChain chain, JSON parsing
├── index.html      — Two textarea UI with score display
├── .env            — API key (never pushed)
└── README.md

---

## Getting Started

```bash
cd resumeanalyser
pip install flask flask-cors huggingface-hub langchain-core python-dotenv
```

Create `.env`:
HUGGINGFACE_API_KEY=hf_your_key_here

Run:
```bash
python app.py
```

Open: `http://127.0.0.1:5000`

---

## Key Implementation Details

- Two input variables in one PromptTemplate — `{job_description}` and `{resume}`
- Temperature 0.1 — document comparison must be precise and consistent
- JSON cleaning block strips markdown fences before parsing
- Dynamic score coloring calculated in JavaScript based on returned number

---

## Environment Variables

| Variable | Description |
|---|---|
| HUGGINGFACE_API_KEY | HuggingFace user access token |

---

## License

MIT