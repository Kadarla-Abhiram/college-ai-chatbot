import os
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer, util

# ---------- LOAD ENV ----------
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------- LOAD FAQ ----------
with open("college_faq.txt", "r", encoding="utf-8") as f:
    faq_lines = [line.strip() for line in f if line.strip()]

# ---------- EMBEDDINGS ----------
model = SentenceTransformer("all-MiniLM-L6-v2")
faq_embeddings = model.encode(faq_lines, convert_to_tensor=True)


# ---------- MAIN FUNCTION ----------
def get_bot_response(user_input: str) -> str:

    query_embedding = model.encode(user_input, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, faq_embeddings)[0]

    best_idx = scores.argmax().item()
    best_score = scores[best_idx].item()
    best_faq = faq_lines[best_idx]

    # ---------- STRONG MATCH ----------
    if best_score > 0.70:
        return best_faq

    # ---------- WEAK MATCH → LLM WITH CONTEXT ----------
    if best_score > 0.45:
        context = "\n".join(faq_lines)

        prompt = f"""
You are an AI assistant for VNR VJIET college.

Use ONLY the information below to answer.
If answer not present → say:
"I can answer only questions related to VNR VJIET college."

FAQ DATA:
{context}

Question: {user_input}
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
        )

        return response.choices[0].message.content.strip()

    # ---------- NO MATCH ----------
    return "I can answer only questions related to VNR VJIET college."
