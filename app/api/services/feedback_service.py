import os
import json
import re
import time
import requests
from dotenv import load_dotenv

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = os.getenv("HF_MODEL") or "google/gemma-2b-it"

MODEL_URL = "https://router.huggingface.co/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
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


def generate_feedback(question, answer, analysis, feedback_mode="harsh"):

    scores = analysis.get("scores", {})

    user_prompt = f"""
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
                    "model": HF_MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": 800,
                    "temperature": 0.4
                },
                timeout=90
            )

            if response.status_code != 200:
                raise Exception(f"HF API error {response.status_code}: {response.text}")

            result = response.json()
            raw_text = result["choices"][0]["message"]["content"]

            cleaned = _clean_json(raw_text)
            parsed = json.loads(cleaned)

            return parsed

        except Exception as e:
            print("FEEDBACK ERROR:", e)
            if attempt == 1:
                break
            time.sleep(2)

    fallback = DEFAULT_FEEDBACK.copy()
    fallback["ideal_answer"] = answer
    return fallback
