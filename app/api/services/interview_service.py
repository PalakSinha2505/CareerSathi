import os
import requests
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = os.getenv("HF_MODEL") or "google/gemma-2b-it"

MODEL_URL = "https://router.huggingface.co/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}

SYSTEM_PROMPT = """
You are a professional technical interviewer.

Rules:
- Ask one clear, concise question at a time.
- Do not provide feedback.
- Do not explain anything.
- Only return the next interview question.
- Keep questions relevant to the role and experience level.
- Adjust difficulty based on previous answers.
- Maintain a realistic interview tone.

Return only the question text.
"""

FALLBACK_QUESTIONS = {
    "Software Engineer": [
        "Can you describe a project you worked on and your specific role in it?",
        "What data structures are you most comfortable with and why?",
        "Explain a challenging bug you encountered and how you solved it."
    ],
    "default": [
        "Can you introduce yourself and describe your recent experience?",
        "What is one professional achievement you are proud of?",
        "Describe a challenging situation and how you handled it."
    ]
}


def _build_prompt(role: str, experience_level: str, history: List[Dict]) -> str:
    conversation = ""

    for turn in history[-3:]:
        if "question" in turn and "answer" in turn:
            conversation += f"\nInterviewer: {turn['question']}\n"
            conversation += f"Candidate: {turn['answer']}\n"

    return f"""
Interview Context:
Role: {role}
Experience Level: {experience_level}

Previous Conversation:
{conversation}

Ask the next interview question.
Only return the question.
"""


def _fallback_question(role: str, history: List[Dict]) -> str:
    questions = FALLBACK_QUESTIONS.get(role, FALLBACK_QUESTIONS["default"])
    index = len(history) % len(questions)
    return questions[index]


def generate_question(role: str, experience_level: str, history: List[Dict]) -> str:

    user_prompt = _build_prompt(role, experience_level, history)

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
                "max_tokens": 200,
                "temperature": 0.6
            },
            timeout=90
        )

        if response.status_code != 200:
            raise Exception(f"HF API error {response.status_code}: {response.text}")

        result = response.json()
        question = result["choices"][0]["message"]["content"].strip()

        if not question or len(question) < 10:
            return _fallback_question(role, history)

        return question

    except Exception as e:
        print(f"[ERROR] Question generation failed: {e}")
        return _fallback_question(role, history)
