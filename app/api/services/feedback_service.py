import os
import json
import re
import time
import requests
from dotenv import load_dotenv

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

MODEL_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}"
}

SYSTEM_PROMPT = """
You are a senior technical interviewer and career mentor with high hiring standards.

Your job is NOT to be polite.
Your job is to be honest, precise, and useful.

You must:
- Explicitly call out vague, weak, or careless phrasing
- Explain how specific parts reduce clarity, confidence, or credibility
- Focus on communication quality and structure
- Give advice the candidate can immediately apply

Return raw JSON only.
Do NOT wrap JSON in markdown.
"""

DEFAULT_FEEDBACK = {
    "verbal_feedback": "Mentor evaluation unavailable due to system issue.",
    "key_issues": ["Unable to parse AI response."],
    "actionable_tips": ["Work on structuring answers clearly."],
    "ideal_answer": "",
    "verdict": "Undetermined"
}


def _clean_json(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group()

    return text


def generate_feedback(
    question: str,
    answer: str,
    analysis: dict,
    feedback_mode: str = "harsh"
) -> dict:

    scores = analysis.get("scores", {})

    prompt = f"""
{SYSTEM_PROMPT}

Mode: {feedback_mode.upper()}

Interview Question:
{question}

Candidate Answer:
{answer}

Analysis Summary:
Clarity: {scores.get("clarity")}
Communication: {scores.get("communication")}
Confidence: {scores.get("confidence")}
Structure: {scores.get("structure")}
English: {scores.get("english")}

Return ONLY valid JSON in this format:

{{
  "verbal_feedback": "Detailed paragraph explaining evaluation",
  "key_issues": ["issue1", "issue2", "issue3"],
  "actionable_tips": ["tip1", "tip2", "tip3"],
  "ideal_answer": "Improved professional version of answer",
  "verdict": "Strong Hire / Hire / Borderline / No Hire"
}}
"""

    for attempt in range(2):
        try:
            response = requests.post(
                MODEL_URL,
                headers=headers,
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 800,
                        "temperature": 0.4,
                        "return_full_text": False
                    }
                },
                timeout=90
            )

            result = response.json()

            if isinstance(result, dict) and "error" in result:
                raise Exception(result["error"])

            raw_text = result[0]["generated_text"]
            cleaned = _clean_json(raw_text)
            parsed = json.loads(cleaned)

            return {
                "verbal_feedback": parsed.get("verbal_feedback", ""),
                "key_issues": parsed.get("key_issues", []),
                "actionable_tips": parsed.get("actionable_tips", []),
                "ideal_answer": parsed.get("ideal_answer", ""),
                "verdict": parsed.get("verdict", "Undetermined"),
            }

        except Exception as e:
            print("FEEDBACK ERROR:", e)
            if attempt == 1:
                break
            time.sleep(2)

    fallback = DEFAULT_FEEDBACK.copy()
    fallback["ideal_answer"] = answer
    return fallback
