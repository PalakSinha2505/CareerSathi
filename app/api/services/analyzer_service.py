from cohere import ClientV2
import os
import json
from dotenv import load_dotenv
import time
from httpx import RemoteProtocolError, HTTPError

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
MODEL_NAME = os.getenv("COHERE_MODEL")

if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY not found in environment variables.")

co = ClientV2(api_key=COHERE_API_KEY)

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
Return ONLY valid JSON.
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
            print("MODEL:", MODEL_NAME)
            print("KEY PRESENT:", bool(COHERE_API_KEY))

            response = co.chat(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            print("RAW COHERE RESPONSE:")
            print(response)

            raw_text = response.message.content[0].text
            cleaned = _clean_json(raw_text)
            parsed = json.loads(cleaned)

            return parsed

        except Exception as e:
            print("REAL ERROR:", e)
            if attempt == 1:
                return DEFAULT_RESPONSE
            time.sleep(1)


