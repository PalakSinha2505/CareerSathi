import os
import json
import time
import requests
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
You are a senior technical interviewer and career mentor.

You are generating FEEDBACK — NOT evaluation scores.

Do NOT return:
- scores
- clarity
- communication
- confidence
- structure
- english

Return ONLY this JSON structure:

{
  "verbal_feedback": "string",
  "key_issues": ["string"],
  "actionable_tips": ["string"],
  "ideal_answer": "string",
  "verdict": "Strong Hire / Hire / Borderline / No Hire"
}

Rules:
- Do NOT wrap JSON in markdown.
- Do NOT include explanations outside JSON.
- Ensure newline characters inside strings are escaped using \\n.
"""

DEFAULT_FEEDBACK = {
    "verbal_feedback": "Mentor evaluation unavailable due to system issue.",
    "key_issues": ["Unable to parse AI response."],
    "actionable_tips": ["Work on structuring answers clearly."],
    "ideal_answer": "",
    "verdict": "Undetermined"
}


def _clean_json(text: str) -> str:
    """
    Extracts the first valid JSON object from model output safely.
    Removes markdown fences and trims incomplete endings.
    """

    if not text:
        return ""

    text = text.strip()

    # Remove markdown fences if present
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    # Extract JSON boundaries manually (safer than regex)
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    # Remove problematic control characters
    text = text.replace("\r", "")
    text = text.replace("\t", " ")

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

Return ONLY valid JSON.
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
                    # Reduced to avoid cutoff
                    "max_tokens": 550,
                    "temperature": 0.3
                },
                timeout=90
            )

            print("HF STATUS:", response.status_code)
            print("HF RAW:", response.text)

            if response.status_code != 200:
                raise Exception(f"HF API error {response.status_code}: {response.text}")

            result = response.json()

            choice = result["choices"][0]
            raw_text = choice["message"]["content"]

            # If model was cut due to length, retry
            if choice.get("finish_reason") == "length":
                raise Exception("Model output truncated (length limit reached)")

            cleaned = _clean_json(raw_text)

            parsed = json.loads(cleaned, strict=False)

            # Final structural validation
            required_keys = [
                "verbal_feedback",
                "key_issues",
                "actionable_tips",
                "ideal_answer",
                "verdict"
            ]

            if not all(k in parsed for k in required_keys):
                raise Exception("Missing required JSON fields")

            return parsed

        except Exception as e:
            print("FEEDBACK ERROR:", e)
            if attempt == 1:
                break
            time.sleep(2)

    fallback = DEFAULT_FEEDBACK.copy()
    fallback["ideal_answer"] = answer
    return fallback
