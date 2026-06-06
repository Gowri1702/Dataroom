import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI


def _load_env():
    root = Path(__file__).parent.parent
    for name in (" .env", ".env"):
        candidate = root / name
        if candidate.exists():
            load_dotenv(dotenv_path=candidate)
            return
    load_dotenv()


_load_env()


def get_openai_client():

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return None

    client = OpenAI(api_key=api_key)
    return client


def build_context_from_chunks(chunks):

    context_parts = []

    for chunk in chunks:
        page_number = chunk["page_number"]
        text = chunk["text"]

        context_parts.append(
            f"[Page {page_number}]\n{text}"
        )

    context = "\n\n".join(context_parts)
    return context


def answer_pdf_question(question, retrieved_chunks):

    client = get_openai_client()

    if client is None:
        return (
            "OpenAI API key not found. Add your API key to the .env file as "
            "OPENAI_API_KEY=your_api_key_here."
        )

    context = build_context_from_chunks(retrieved_chunks)

    prompt = f"""
You are DataRoom AI, an evidence-grounded business analyst.

Answer the user's question using ONLY the provided PDF evidence.

Rules:
1. Do not use outside knowledge.
2. If the evidence is not enough, say: "Not enough evidence in the uploaded PDF."
3. Include page citations in the answer using this format: (Page X).
4. Be concise, professional, and business-focused.

User question:
{question}

PDF evidence:
{context}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a careful business analyst who answers only from provided evidence."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    answer = response.choices[0].message.content
    return answer