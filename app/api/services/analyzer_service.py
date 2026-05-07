import json
import requests
import time
import os
import re
from dotenv import load_dotenv

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = os.getenv("HF_MODEL") or "Qwen/Qwen2.5-7B-Instruct"
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
Ensure newline characters inside strings are escaped using \\n.
Return raw JSON only.
Do NOT wrap JSON in markdown.

STRICT RULES:
- Return ONLY valid JSON
- DO NOT add any explanation before or after
- DO NOT use markdown
- DO NOT write anything except JSON
- Ensure all quotes are properly escaped

FORMAT:
{
  "scores": {
    "clarity": number,
    "communication": number,
    "confidence": number,
    "structure": number,
    "english": number
  },
  "strengths": "text",
  "improvements": "text",
  "suggested_rewrite": "text"
}

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

    # Remove markdown fences
    if raw_text.startswith("```"):
        parts = raw_text.split("```")
        if len(parts) >= 2:
            raw_text = parts[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    # Extract JSON block only
    match = re.search(r"\{[\s\S]*\}", raw_text)
    if match:
        raw_text = match.group()

    # Remove problematic control characters
    raw_text = raw_text.replace("\r", "")
    raw_text = raw_text.replace("\t", " ")

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

            if "choices" not in result or not result["choices"]:
                raise Exception(f"No choices returned: {result}")

            choice = result["choices"][0]

            if "message" not in choice or "content" not in choice["message"]:
                raise Exception(f"Malformed response structure: {choice}")

            raw_text = choice["message"]["content"]

            cleaned = _clean_json(raw_text)
            print("CLEANED:", cleaned)

            # strict=False prevents control character crash
            try:
                parsed = json.loads(cleaned, strict=False)
            except Exception as e:
                print("JSON PARSE FAILED:", cleaned)
                raise e
            # Ensure required fields exist
            if "scores" not in parsed:
                return {
                    **DEFAULT_RESPONSE,
                    "error": "MODEL_FAILED"
                }

            return parsed

        except Exception as e:
            print("REAL ERROR:", e)
            if attempt == 1:
                return {
                    **DEFAULT_RESPONSE,
                    "error": str(e)
                }
            time.sleep(1)
