# Resume Analyser — compares resume against job description
# Day 12 | Pillar: LangChain | Multi-input comparison chain

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from huggingface_hub import InferenceClient
import os
import json

app = Flask(__name__)
CORS(app)

load_dotenv()
API_KEY = os.getenv("HUGGINGFACE_API_KEY")
print(f"API KEY LOADED: {API_KEY}")

print("Setting up Resume Analyser...")

# ── PROVEN CLIENT ──────────────────────────────────────────────
client = InferenceClient(
    provider="novita",
    api_key=API_KEY,
)

# ── CORE LLM FUNCTION ─────────────────────────────────────────
def call_llama(prompt_value, temp=0.3, tokens=600):
    prompt_text = prompt_value.text if hasattr(prompt_value, 'text') else str(prompt_value)
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": prompt_text}],
        max_tokens=tokens,
        temperature=temp
    )
    return response.choices[0].message.content.strip()

parser = StrOutputParser()

# ── PROMPT TEMPLATE — two inputs ──────────────────────────────
analyse_template = PromptTemplate(
    input_variables=["job_description", "resume"],
    template="""You are an expert HR recruiter and resume analyst.
Compare the resume against the job description carefully.

Return ONLY a valid JSON object with these exact keys:
- match_score: a number from 0 to 100 showing how well the resume matches
- matched_skills: list of skills in both the job description and resume
- missing_skills: list of skills required in job but missing from resume
- suggestions: list of 3 specific actionable suggestions to improve the resume
- summary: one sentence overall assessment

Job Description:
{job_description}

Resume:
{resume}

JSON:"""
)

# ── CHAIN ─────────────────────────────────────────────────────
analyse_chain = analyse_template | RunnableLambda(
    lambda x: call_llama(x, temp=0.1, tokens=600)
) | parser

print("Resume Analyser ready!")

# ── FUNCTION ──────────────────────────────────────────────────
def analyse_resume(job_description, resume):
    if len(job_description.split()) < 20:
        return {"error": "Job description too short. Please paste the full job description."}
    if len(resume.split()) < 20:
        return {"error": "Resume too short. Please paste your full resume."}

    raw = analyse_chain.invoke({
        "job_description": job_description,
        "resume"         : resume
    })

    # clean markdown if model adds it
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        result = json.loads(raw)
        return result
    except json.JSONDecodeError:
        return {"error": "Could not parse result. Please try again."}

# ── ROUTES ────────────────────────────────────────────────────
@app.route("/")
def home():
    return send_from_directory(os.getcwd(), "index.html")

@app.route("/analyse", methods=["POST"])
def analyse():
    data            = request.get_json()
    job_description = data.get("job_description", "")
    resume          = data.get("resume", "")

    if not job_description.strip():
        return jsonify({"error": "Please paste the job description."})
    if not resume.strip():
        return jsonify({"error": "Please paste your resume."})

    try:
        result = analyse_resume(job_description, resume)
        return jsonify(result)
    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)