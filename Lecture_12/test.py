
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

key = os.getenv("OPENAI_API_KEY")
print("API key loaded:", bool(key), flush=True)

if not key:
    raise RuntimeError("OPENAI_API_KEY is missing from .env")

client = OpenAI(timeout=20.0, max_retries=0)

print("Sending request...", flush=True)

try:
    response = client.responses.create(
        model="gpt-4o-mini",
        input="Reply with just Hello."
    )
    print("Response:", response.output_text, flush=True)

except Exception as e:
    print("Error type:", type(e).__name__, flush=True)
    print("Error details:", str(e), flush=True)
