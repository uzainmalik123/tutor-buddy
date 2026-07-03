import os

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

from src.agent.retriever import CurriculumRetriever

load_dotenv()

_SYSTEM_INSTRUCTION = (
    "You are TutorBuddy, a friendly AI tutor for Grade 4–8 students. "
    "Always use simple, clear language appropriate for children aged 9–14. "
    "Never produce adult content, harmful content, or anything unrelated to education. "
    "Stay focused on the student's question only."
)


class TutorAgent:
    """AI tutor that uses Gemini with RAG from a CurriculumRetriever."""

    def __init__(self, retriever: CurriculumRetriever):
        self.retriever = retriever
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = "gemini-2.5-flash"

    def _build_system_instruction(self, language: str = "english") -> str:
        instruction = _SYSTEM_INSTRUCTION
        if language.lower() == "urdu":
            instruction += " Respond in Urdu."
        return instruction

    def _generate(self, prompt: str, system: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=system,
            ),
        )
        return response.text

    def answer_question(self, question: str, language: str = "english") -> str:
        try:
            context = self.retriever.retrieve(question)
        except Exception as e:
            return f"Could not search the curriculum: {e}"

        system = self._build_system_instruction(language)
        prompt = (
            f"Here is some relevant information:\n{context}\n\n"
            f"Answer the following question in a friendly, simple way:\n{question}"
        )
        try:
            return self._generate(prompt, system)
        except Exception as e:
            err = str(e).lower()
            if "api key" in err or "not found" in err or "permission" in err or "access" in err:
                return "The API key for Gemini is missing, invalid, or lacks permission. Make sure GEMINI_API_KEY is set in your .env file and the model is enabled in Google Cloud."
            if "safety" in err or "blocked" in err:
                return "Gemini blocked the response due to safety filters. Try rephrasing your question."
            if "quota" in err or "rate" in err or "limit" in err or "429" in err:
                return "API rate limit reached — too many requests. Please wait a moment and try again."
            return f"Gemini API error: {e}"

    def summarize_lesson(self, text: str, language: str = "english") -> str:
        if not text.strip():
            return "Please provide some lesson text to summarize."

        system = self._build_system_instruction(language)
        prompt = (
            "Summarize the following lesson in simple words that a Grade 4–8 student can understand. "
            f"Keep it short and clear:\n\n{text}"
        )
        try:
            return self._generate(prompt, system)
        except Exception as e:
            err = str(e).lower()
            if "api key" in err or "not found" in err or "permission" in err or "access" in err:
                return "The API key for Gemini is missing, invalid, or lacks permission. Make sure GEMINI_API_KEY is set in your .env file and the model is enabled in Google Cloud."
            if "safety" in err or "blocked" in err:
                return "Gemini blocked the response due to safety filters. The lesson text may contain sensitive content."
            if "quota" in err or "rate" in err or "limit" in err or "429" in err:
                return "API rate limit reached — too many requests. Please wait a moment and try again."
            return f"Gemini API error: {e}"

    def generate_quiz(self, topic: str, num_questions: int = 3) -> str:
        if not topic.strip():
            return "Please provide a topic for the quiz."

        try:
            context = self.retriever.retrieve(topic)
        except Exception as e:
            return f"Could not search the curriculum: {e}"

        system = self._build_system_instruction("english")
        prompt = (
            f"Here is some relevant information:\n{context}\n\n"
            f"Create a {num_questions}-question multiple-choice quiz on the topic '{topic}'. "
            "For each question, provide four options (A, B, C, D), the correct answer, "
            "and a short explanation. Use simple language."
        )
        try:
            return self._generate(prompt, system)
        except Exception as e:
            err = str(e).lower()
            if "api key" in err or "not found" in err or "permission" in err or "access" in err:
                return "The API key for Gemini is missing, invalid, or lacks permission. Make sure GEMINI_API_KEY is set in your .env file and the model is enabled in Google Cloud."
            if "safety" in err or "blocked" in err:
                return "Gemini blocked the response due to safety filters. Try a different topic."
            if "quota" in err or "rate" in err or "limit" in err or "429" in err:
                return "API rate limit reached — too many requests. Please wait a moment and try again."
            return f"Gemini API error: {e}"

    def generate_quiz_json(self, topic: str, num_questions: int = 3) -> list[dict]:
        """Generate a quiz and return structured JSON data for interactive rendering."""
        if not topic.strip():
            return []

        try:
            context = self.retriever.retrieve(topic)
        except Exception:
            return []

        system = self._build_system_instruction("english")
        prompt = (
            f"Here is some relevant information:\n{context}\n\n"
            f"Create a {num_questions}-question multiple-choice quiz on the topic '{topic}'. "
            "For each question, provide exactly 4 options with one correct answer and a short explanation."
        )
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {
                            "questions": {
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "question": {"type": "STRING"},
                                        "options": {
                                            "type": "ARRAY",
                                            "items": {"type": "STRING"},
                                        },
                                        "correct_index": {"type": "INTEGER"},
                                        "explanation": {"type": "STRING"},
                                    },
                                    "required": [
                                        "question",
                                        "options",
                                        "correct_index",
                                        "explanation",
                                    ],
                                },
                            }
                        },
                        "required": ["questions"],
                    },
                ),
            )
            import json
            data = json.loads(response.text)
            return data.get("questions", [])
        except Exception as e:
            print(f"[TutorAgent] generate_quiz_json error: {e}")
            return []
