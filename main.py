from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

client = genai.Client(api_key=api_key)

print("API KEY loaded successfully.")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents = "What is my name?",
    config = types.GenerateContentConfig(
        system_instruction= "The user's name is Ansel",
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    ),
)
print(response.text)