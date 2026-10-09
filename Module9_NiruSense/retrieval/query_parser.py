import json
import datetime
from ..routing.llm_client import LLMClient

SYSTEM_PROMPT = """You are a query understanding assistant for a Kenyan sentiment analysis system.
Extract search filters from the user's natural language query.

Current Date: {current_date}

Return a JSON object with the following fields:
- "topic": string or null (The main subject, e.g., "Finance Bill", "Police Brutality")
- "sentiment": string or null (e.g., "negative", "positive", "angry", "happy")
- "date_range": string or null (e.g., "last_24_hours", "last_7_days", "last_30_days", "2024-01-01_to_2024-01-31")
- "platform": string or null (e.g., "twitter", "tiktok", "news")
- "sort_by": string (one of: "relevance", "recency", "intensity_desc", "intensity_asc")

Example: "Show me the angriest reactions to police brutality last week"
Output:
{
    "topic": "police brutality",
    "sentiment": "angry",
    "date_range": "last_7_days",
    "platform": null,
    "sort_by": "intensity_desc"
}

Example: "What is the mood on Finance Bill today?"
Output:
{
    "topic": "Finance Bill",
    "sentiment": null,
    "date_range": "last_24_hours",
    "platform": null,
    "sort_by": "recency"
}
"""


class QueryParser:
    def __init__(self):
        self.llm_client = LLMClient()

    def parse_query(self, query: str) -> dict:
        """
        Parses a natural language query into structured filters.
        """
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        prompt = SYSTEM_PROMPT.format(current_date=current_date)

        # We use the LLMClient's internal logic which handles Gemini/Groq
        # We need to construct a prompt that fits the LLMClient's analyze_text method
        # or bypass it if it's too specific to sentiment analysis.
        # The LLMClient.analyze_text uses specific system prompts for sentiment.
        # We should probably extend LLMClient or just use the raw clients if exposed.
        # Looking at LLMClient, it initializes clients but analyze_text is hardcoded for sentiment.
        # Let's modify LLMClient to accept a system prompt or just access the clients directly here if possible.
        # Since LLMClient stores self.gemini_model and self.groq_client, we can access them.

        try:
            # Try Gemini
            if self.llm_client.gemini_model:
                response = self.llm_client.gemini_model.generate_content(
                    f'{prompt}\nQuery: "{query}"'
                )
                return self._parse_json(response.text)

            # Try Groq
            elif self.llm_client.groq_client:
                chat_completion = self.llm_client.groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f'Query: "{query}"'},
                    ],
                    model="llama-3.1-70b-versatile",
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                return self._parse_json(chat_completion.choices[0].message.content)

        except Exception as e:
            print(f"Query parsing failed: {e}")

        # Fallback
        return {
            "topic": None,
            "sentiment": None,
            "date_range": None,
            "platform": None,
            "sort_by": "relevance",
        }

    def _parse_json(self, text: str) -> dict:
        try:
            text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except json.JSONDecodeError:
            return {}
