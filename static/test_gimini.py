import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Use the correct model name for v1
model = genai.GenerativeModel(model_name="models/gemini-1.5-pro")

chat = model.start_chat()

try:
    response = chat.send_message("Tell me a programming joke.")
    print("✅ Response:")
    print(response.text)
except Exception as e:
    print("❌ Error:")
    print(e)
