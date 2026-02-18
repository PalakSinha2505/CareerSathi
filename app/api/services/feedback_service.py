import os
import json
import time
import re
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
  "verbal_feedback": "Concise but impactful paragraph (5-8 sentences maximum)",
  "key_issues": ["string"],
  "actionable_tips": ["string"],
  "ideal_answer": "string",
  "verdict": "Strong Hire / Hire / Borderline / No Hire"
}

Rules:
- Do NOT wrap JSON in markdown.
- Do NOT include explanations outside JSON.
- Ensure newline characters inside strings are escaped using \\n.
- If you generate invalid JSON, the response will be discarded.
- Keep responses concise and avoid unnecessary elaboration.

"""

DEFAULT_FEEDBACK = {
    "verbal_feedback": "Mentor evaluation unavailable due to system issue.",
    "key_issues": ["Unable to parse AI response."],
    "actionable_tips": ["Work on structuring answers clearly."],
    "ideal_answer": "",
    "verdict": "Undetermined"
}


def _extract_json(text: str):
    """
    Extract first valid JSON object using balanced brace tracking.
    Safer than naive slicing.
    """

    if not text:
        return None

    # Remove markdown fences if present
    text = re.sub(r"```json|```", "", text).strip()

    stack = []
    start_index = None

    for i, char in enumerate(text):
        if char == "{":
            if not stack:
                start_index = i
            stack.append("{")
        elif char == "}":
            if stack:
                stack.pop()
                if not stack and start_index is not None:
                    return text[start_index:i + 1]

    return None


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
                    "max_tokens": 800,
                    "temperature": 0.3
                },
                timeout=90
            )

            print("HF STATUS:", response.status_code)

            if response.status_code != 200:
                raise Exception(f"HF API error {response.status_code}: {response.text}")

            result = response.json()

            # Validate structure
            if "choices" not in result or not result["choices"]:
                raise Exception("No choices returned from model")

            choice = result["choices"][0]

            if "message" not in choice or "content" not in choice["message"]:
                raise Exception("Malformed response structure")

            raw_text = choice["message"]["content"]

            # Retry if truncated
            finish_reason = choice.get("finish_reason")

            if finish_reason == "length":
                print("WARNING: Model output reached token limit. Attempting recovery...")


            json_text = _extract_json(raw_text)

            if not json_text:
                raise Exception("No JSON found in model output")

            parsed = json.loads(json_text)

            # Structural validation
            required_keys = [
                "verbal_feedback",
                "key_issues",
                "actionable_tips",
                "ideal_answer",
                "verdict"
            ]

            if not all(k in parsed for k in required_keys):
                raise Exception("Missing required JSON fields")

            # Type validation
            if not isinstance(parsed["key_issues"], list):
                raise Exception("key_issues must be a list")

            if not isinstance(parsed["actionable_tips"], list):
                raise Exception("actionable_tips must be a list")

            return parsed

        except Exception as e:
            print("FEEDBACK ERROR:", str(e))
            if attempt == 1:
                break
            time.sleep(2)

    fallback = DEFAULT_FEEDBACK.copy()
    fallback["ideal_answer"] = answer
    return fallback
