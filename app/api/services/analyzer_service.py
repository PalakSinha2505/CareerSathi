import json
import requests
import time
import os
from dotenv import load_dotenv

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = os.getenv("HF_MODEL") or "google/gemma-2b-it"

MODEL_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"


headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}"
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

    prompt = f"""
{SYSTEM_PROMPT}

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
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 600,
                        "temperature": 0.3,
                        "return_full_text": False
                    }
                },
                timeout=90
            )

            print("HF STATUS:", response.status_code)
            print("HF RAW:", response.text)

            if response.status_code != 200:
                raise Exception(f"HF API error {response.status_code}: {response.text}")

            if not response.text.strip():
                raise Exception("Empty response from HuggingFace")

            result = response.json()

            # Handle both router response formats
            if isinstance(result, list):
                raw_text = result[0].get("generated_text", "")
            elif isinstance(result, dict):
                raw_text = result.get("generated_text", "")
            else:
                raise Exception("Unexpected HF response format")

            if not raw_text:
                raise Exception("No generated_text in response")

            cleaned = _clean_json(raw_text)
            parsed = json.loads(cleaned)

            return parsed


        except Exception as e:
            print("REAL ERROR:", e)
            if attempt == 1:
                return DEFAULT_RESPONSE
            time.sleep(1)
