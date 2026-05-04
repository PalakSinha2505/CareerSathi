import os
import requests
from dotenv import load_dotenv
from typing import List, Dict
import random

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = os.getenv("HF_MODEL") or "mistralai/Mistral-7B-Instruct-v0.2"

MODEL_URL = "https://router.huggingface.co/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}

SYSTEM_PROMPT = """
You are a professional interviewer.

STRICT RULES:
- Ask ONLY ONE question.
- NEVER repeat a previous question.
- DO NOT ask generic HR questions repeatedly (like intro, achievements, etc).
- Focus on ROLE-SPECIFIC and SKILL-BASED questions.
- Progressively increase difficulty.
- Use previous answers to guide next question.
- Avoid vague or broad questions.
- Be specific and technical where possible.

OUTPUT FORMAT:
Return ONLY the question text.
No explanations. No comments.
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
    asked_questions = "\n".join([f"- {h['question']}" for h in history if "question" in h])

    conversation = ""
    for turn in history[-5:]:
        if "question" in turn and "answer" in turn:
            conversation += f"\nInterviewer: {turn['question']}\n"
            conversation += f"Candidate: {turn['answer']}\n"

    return f"""
Interview Context:
Role: {role}
Experience Level: {experience_level}

DO NOT REPEAT THESE QUESTIONS:
{asked_questions}

Previous Conversation:
{conversation}

Generate the NEXT UNIQUE and ROLE-SPECIFIC question.
"""

def _fallback_question(role: str, history: List[Dict]) -> str:
    role = role.lower()

    ROLE_QUESTIONS = {
        "frontend developer": [
            "Explain the difference between useEffect and useLayoutEffect.",
            "How does the virtual DOM improve performance?",
            "What are React hooks and why are they used?",
            "Explain CSS specificity with examples.",
            "How do you optimize frontend performance?"
        ],
        "backend developer": [
            "What is the difference between REST and GraphQL?",
            "Explain database indexing and its importance.",
            "How do you handle authentication in APIs?",
            "What are microservices and when would you use them?",
            "Explain caching strategies in backend systems."
        ],
        "software developer": [
            "Explain time complexity with an example.",
            "What is a deadlock and how do you prevent it?",
            "Difference between stack and heap memory?",
            "Explain OOP principles with real examples.",
            "What is multithreading?"
        ],
        "default": [
            "Describe a challenging problem you solved recently.",
            "How do you approach learning a new skill?",
            "Explain a project where you made a significant impact.",
            "What is your problem-solving approach?",
            "How do you handle failure?"
        ]
    }

    questions = ROLE_QUESTIONS.get(role, ROLE_QUESTIONS["default"])

    # FIX: avoid repetition
    asked = set([h["question"] for h in history if "question" in h])
    available = [q for q in questions if q not in asked]

    if not available:
        available = questions

    return random.choice(available)


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
