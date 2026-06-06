import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


def chunk_pdf_pages(pdf_pages, chunk_size=800, overlap=150):
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")
    """
    Split PDF page text into smaller overlapping chunks.

    Args:
        pdf_pages: List of dictionaries with page_number and text.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters repeated between chunks.

    Returns:
        chunks: List of chunk dictionaries.
    """

    chunks = []

    for page in pdf_pages:
        page_number = page["page_number"]
        text = page["text"]

        if not text or text.strip() == "":
            continue

        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            chunk = {
                "chunk_id": len(chunks),
                "page_number": page_number,
                "text": chunk_text
            }

            chunks.append(chunk)

            start = end - overlap

    return chunks


def create_faiss_index(chunks, model):

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(texts, convert_to_numpy=True)

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index, embeddings


def retrieve_relevant_chunks(question, chunks, index, model, top_k=4):

    question_embedding = model.encode([question], convert_to_numpy=True)
    question_embedding = question_embedding.astype("float32")

    distances, indices = index.search(question_embedding, top_k)

    results = []

    for distance, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue

        result = {
            "chunk_id": chunks[idx]["chunk_id"],
            "page_number": chunks[idx]["page_number"],
            "text": chunks[idx]["text"],
            "distance": float(distance)
        }

        results.append(result)

    return results