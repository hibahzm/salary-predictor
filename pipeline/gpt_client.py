import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

client = OpenAI(api_key=API_KEY)


class GPTError(Exception):
    """Raised when GPT fails."""
    pass


def chat(
    user_message: str,
    system_message: str = "You are a helpful data analyst.",
    timeout: int = 120,
) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # fast + cheap (good replacement for gemini-2.5-flash)
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},  # ensures valid JSON output
        )

        text = response.choices[0].message.content

        if not text:
            raise GPTError("Empty response from GPT")

        return text

    except Exception as e:
        raise GPTError(str(e))