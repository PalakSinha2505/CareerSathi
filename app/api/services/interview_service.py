from cohere import ClientV2
import os
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
COHERE_MODEL = os.getenv("COHERE_MODEL", "command-a-03-2025")

if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY not found in environment variables")

co = ClientV2(api_key=COHERE_API_KEY)


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


# Basic fallback questions if AI fails
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


def _build_messages(role: str, experience_level: str, history: List[Dict]) -> List[Dict]:
    """
    Build conversation history for Cohere chat.
    """
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"""
Interview Context:
Role: {role}
Experience Level: {experience_level}
"""
        }
    ]

    # Only use last 3 exchanges to limit token usage
    for turn in history[-3:]:
        if "question" in turn and "answer" in turn:
            messages.append({"role": "assistant", "content": turn["question"]})
            messages.append({"role": "user", "content": turn["answer"]})

    messages.append({
        "role": "user",
        "content": "Ask the next interview question. Only return the question."
    })

    return messages


def _fallback_question(role: str, history: List[Dict]) -> str:
    """
    Return a fallback question if AI fails.
    """
    questions = FALLBACK_QUESTIONS.get(role, FALLBACK_QUESTIONS["default"])
    index = len(history) % len(questions)
    return questions[index]


def generate_question(role: str, experience_level: str, history: List[Dict]) -> str:
    """
    Generate next interview question using Cohere.
    Safe version with fallback handling.
    """

    try:
        messages = _build_messages(role, experience_level, history)

        response = co.chat(
            model=COHERE_MODEL,
            messages=messages,
            temperature=0.6
        )

        question = response.message.content[0].text.strip()

        # Basic validation
        if not question or len(question) < 10:
            return _fallback_question(role, history)

        return question

    except Exception as e:
        # Log error in production (replace with logging system later)
        print(f"[ERROR] Question generation failed: {e}")
        return _fallback_question(role, history)
