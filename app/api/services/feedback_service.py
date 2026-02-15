from cohere import ClientV2
import os
import json
import re
import time
from dotenv import load_dotenv
from httpx import RemoteProtocolError, HTTPError

load_dotenv()

co = ClientV2(api_key=os.getenv("COHERE_API_KEY"))

SYSTEM_PROMPT = """
You are a senior technical interviewer and career mentor with high hiring standards.

Your job is NOT to be polite.
Your job is to be honest, precise, and useful.

Evaluate the candidate as if this were a real interview that affects their career.

You must:
- Explicitly call out vague, weak, or careless phrasing
- Explain how specific parts reduce clarity, confidence, or credibility
- Focus on communication quality and structure
- Give advice the candidate can immediately apply

Return ONLY valid JSON.
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
    Cleans markdown formatting and extracts JSON safely.
    """
    text = text.strip()

    # Remove markdown code blocks
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    # Extract first JSON object
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

    for attempt in range(2):  # retry once if network or parsing fails
        try:
            response = co.chat(
                model="command-a-03-2025",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4
            )

            raw_text = response.message.content[0].text
            cleaned = _clean_json(raw_text)

            parsed = json.loads(cleaned)

            return {
                "verbal_feedback": parsed.get("verbal_feedback", ""),
                "key_issues": parsed.get("key_issues", []),
                "actionable_tips": parsed.get("actionable_tips", []),
                "ideal_answer": parsed.get("ideal_answer", ""),
                "verdict": parsed.get("verdict", "Undetermined"),
            }

        except (RemoteProtocolError, HTTPError):
            time.sleep(1)
            continue

        except Exception:
            break

    # Final fallback (only if both attempts fail)
    fallback = DEFAULT_FEEDBACK.copy()
    fallback["ideal_answer"] = answer
    return fallback
