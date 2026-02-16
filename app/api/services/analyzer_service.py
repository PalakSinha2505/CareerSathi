import json
import requests
import time
import os
from dotenv import load_dotenv

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = os.getenv("HF_MODEL") or "mistralai/Mistral-7B-Instruct-v0.2"

MODEL_URL = "https://router.huggingface.co/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}

SYSTEM_PROMPT = """
You are an expert interview evaluator.

Evaluate the candidate based on:
- Clarity of thought
- Communication skills
- Confidence
- Structure of the answer
- English language quality

Focus on HOW the answer is delivered rather than technical correctness.
Be strict but fair.
Do NOT give identical scores unless truly deserved.
Do NOT give 7+ unless the answer is structured and confident.
Return raw JSON only.
Do NOT wrap JSON in markdown.
"""

DEFAULT_RESPONSE = {
    "scores": {
        "clarity": 5,
        "communication": 5,
        "confidence": 5,
        "structure": 5,
        "english": 5
    },
    "strengths": "Answer attempted but evaluation unavailable.",
    "improvements": "Could not evaluate due to system issue.",
    "suggested_rewrite": "Expand your answer with better structure and confidence."
}


def _clean_json(raw_text: str):
    raw_text = raw_text.strip()

    if raw_text.startswith("```"):
        parts = raw_text.split("```")
        if len(parts) >= 2:
            raw_text = parts[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    return raw_text


def analyze_answer(question: str, answer: str, role: str, experience_level: str) -> dict:

    user_prompt = f"""
Interview Context:
Role: {role}
Experience Level: {experience_level}

Question:
{question}

Candidate Answer:
{answer}

Return ONLY valid JSON in this format:

{{
  "scores": {{
    "clarity": 1-10,
    "communication": 1-10,
    "confidence": 1-10,
    "structure": 1-10,
    "english": 1-10
  }},
  "strengths": "string",
  "improvements": "string",
  "suggested_rewrite": "string"
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
                    "max_tokens": 600,
                    "temperature": 0.3
                },
                timeout=90
            )

            print("HF STATUS:", response.status_code)
            print("HF RAW:", response.text)

            if response.status_code != 200:
                raise Exception(f"HF API error {response.status_code}: {response.text}")

            result = response.json()
            raw_text = result["choices"][0]["message"]["content"]

            cleaned = _clean_json(raw_text)
            parsed = json.loads(cleaned)

            return parsed

        except Exception as e:
            print("REAL ERROR:", e)
            if attempt == 1:
                return DEFAULT_RESPONSE
            time.sleep(1)
