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
You are a STRICT professional interviewer.

CRITICAL RULES:
- Ask ONLY ONE question.
- NEVER repeat any previous question.
- NO generic HR questions unless it's the first question.
- Questions MUST be ROLE-SPECIFIC and TECHNICAL wherever possible.
- Increase difficulty gradually.
- Avoid vague questions like:
  - "Tell me about yourself"
  - "What are your strengths?"
  - "Describe a challenge"
- Focus on REAL interview-style grilling questions.

GOOD EXAMPLES:
- "Explain how React's reconciliation works."
- "How would you design a scalable REST API?"
- "What indexing strategy would you use for a large dataset?"

BAD EXAMPLES:
- "Tell me about yourself"
- "What is your biggest achievement?"

OUTPUT:
Return ONLY the question text.
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
You are conducting a REALISTIC TECHNICAL INTERVIEW.

ROLE: {role}
EXPERIENCE LEVEL: {experience_level}

STRICT INSTRUCTIONS:
- Ask ROLE-SPECIFIC questions for {role}
- Avoid HR/general questions
- Ask practical, scenario-based or technical questions
- Increase difficulty gradually
- DO NOT repeat questions

ALREADY ASKED QUESTIONS:
{asked_questions}

CONVERSATION:
{conversation}

Now ask the NEXT QUESTION.
"""

def _fallback_question(role: str, history: List[Dict]) -> str:
    role = role.lower()

    ROLE_QUESTIONS = {
        "frontend developer": [
            "Explain how React handles re-rendering and how you optimize it.",
            "What is the difference between controlled and uncontrolled components?",
            "How would you improve the performance of a slow React app?",
            "Explain event delegation in JavaScript.",
            "How do you manage global state in large frontend apps?"
        ],
        "backend developer": [
            "How would you design a scalable authentication system?",
            "Explain database normalization with an example.",
            "How do you handle concurrency in backend systems?",
            "What are rate limiting strategies in APIs?",
            "Explain how caching improves backend performance."
        ],
        "software developer": [
            "Explain time complexity of quicksort and when it degrades.",
            "What is a race condition and how do you prevent it?",
            "Explain memory management in your preferred language.",
            "How would you debug a production issue?",
            "Explain multithreading vs multiprocessing."
        ],
        "hr": [
            "How do you handle conflict resolution between employees?",
            "What metrics do you use to measure employee performance?",
            "How do you design an effective hiring process?",
            "Describe a difficult HR case you handled.",
            "How do you improve employee retention?"
        ],
        "management": [
            "How do you prioritize tasks in a high-pressure environment?",
            "Explain a leadership challenge you faced.",
            "How do you handle underperforming team members?",
            "What strategies do you use for decision-making?",
            "How do you manage cross-functional teams?"
        ],
        "default": [
            "Explain a recent complex problem you solved.",
            "How do you approach learning a new technical skill?",
            "Describe a situation where you had to make a tough decision.",
        ]
    }

    questions = ROLE_QUESTIONS.get(role, ROLE_QUESTIONS["default"])

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
