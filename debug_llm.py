from llama_index.core import Settings
from llama_index.llms.groq import Groq
import os

print(f"GROQ_API_KEY in environment: {'SET' if os.environ.get('GROQ_API_KEY') else 'NOT SET'}")
print(f"OPENAI_API_KEY in environment: {'SET' if os.environ.get('OPENAI_API_KEY') else 'NOT SET'}")

llm = Groq(model="llama-3.3-70b-versatile")
Settings.llm = llm

print(f"Settings.llm type: {type(Settings.llm)}")
print(f"Settings.llm model: {Settings.llm.model}")

try:
    response = Settings.llm.complete("Hi")
    print("Groq completion successful!")
except Exception as e:
    print(f"Groq completion failed with: {type(e).__name__}: {e}")
