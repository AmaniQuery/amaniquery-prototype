import os
import json
import google.generativeai as genai
from groq import Groq
from .prompts import GEMINI_SYSTEM_PROMPT, GROQ_SYSTEM_PROMPT


class LLMClient:
    def __init__(self):
        # Initialize Gemini
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        if self.gemini_key:
            genai.configure(api_key=self.gemini_key)
            self.gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        else:
            self.gemini_model = None
            print("Warning: GEMINI_API_KEY not found.")

        # Initialize Groq
        self.groq_key = os.getenv("GROQ_API_KEY")
        if self.groq_key:
            self.groq_client = Groq(api_key=self.groq_key)
        else:
            self.groq_client = None
            print("Warning: GROQ_API_KEY not found.")

    def analyze_text(self, text: str) -> dict:
        """
        Analyzes text using Gemini, falling back to Groq if necessary.
        Returns a dictionary with sentiment analysis results.
        """
        # Try Gemini first (Primary)
        if self.gemini_model:
            try:
                response = self.gemini_model.generate_content(
                    f'{GEMINI_SYSTEM_PROMPT}\nText: "{text}"'
                )
                return self._parse_json(response.text)
            except Exception as e:
                print(f"Gemini failed: {e}. Trying fallback...")

        # Fallback to Groq
        if self.groq_client:
            try:
                chat_completion = self.groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": GROQ_SYSTEM_PROMPT},
                        {"role": "user", "content": f'Analyze this text: "{text}"'},
                    ],
                    model="llama-3.1-70b-versatile",
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                return self._parse_json(chat_completion.choices[0].message.content)
            except Exception as e:
                print(f"Groq failed: {e}")

        # Both failed
        return {
            "sentiment_score": 0.0,
            "sentiment_label": "Neutral",
            "is_sarcastic": False,
            "explanation": "LLM analysis failed.",
        }

    def _parse_json(self, text: str) -> dict:
        try:
            # Clean markdown code blocks if present
            text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON from LLM: {text}")
            return {
                "sentiment_score": 0.0,
                "sentiment_label": "Neutral",
                "is_sarcastic": False,
                "explanation": "JSON parse error.",
            }
