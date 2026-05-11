from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

def get_client():
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found.")

    return genai.Client(api_key=api_key)