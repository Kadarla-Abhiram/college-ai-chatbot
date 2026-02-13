import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load FAQ content once
with open("college_faq.txt", "r", encoding="utf-8") as f:
    FAQ_DATA = f.read()


def get_bot_response(user_msg):
    prompt = f"""
You are an AI assistant for VNR VJIET college.

Answer the user's question ONLY using the information below.
If the answer is not in the FAQ, reply:
"Sorry, I can only answer questions related to VNR VJIET college."

FAQ:
{FAQ_DATA}

User question: {user_msg}
"""

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.2,
        )

        return chat_completion.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"