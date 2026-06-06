import numpy as np
import pytest
from src.rag_utils import chunk_pdf_pages, create_faiss_index, retrieve_relevant_chunks


@pytest.fixture(scope="module")
def embedding_model():
    from src.rag_utils import load_embedding_model
    return load_embedding_model()


def _make_pages(texts):
    return [{"page_number": i + 1, "text": t} for i, t in enumerate(texts)]


class TestChunkPdfPages:
    def test_basic_chunking(self):
        pages = _make_pages(["A" * 2000])
        chunks = chunk_pdf_pages(pages, chunk_size=800, overlap=150)
        assert len(chunks) > 1

    def test_chunk_keys(self):
        pages = _make_pages(["Hello world " * 50])
        chunks = chunk_pdf_pages(pages)
        assert all({"chunk_id", "page_number", "text"}.issubset(c) for c in chunks)

    def test_page_number_preserved(self):
        pages = _make_pages(["Text on page 1.", "Text on page 2."])
        chunks = chunk_pdf_pages(pages, chunk_size=100, overlap=10)
        page_nums = {c["page_number"] for c in chunks}
        assert 1 in page_nums
        assert 2 in page_nums

    def test_empty_page_skipped(self):
        pages = _make_pages(["", "Real content " * 20])
        chunks = chunk_pdf_pages(pages)
        assert all(c["page_number"] == 2 for c in chunks)

    def test_overlap_less_than_chunk_size(self):
        pages = _make_pages(["X" * 1000])
        chunks = chunk_pdf_pages(pages, chunk_size=500, overlap=100)
        assert len(chunks) > 1

    def test_invalid_overlap_raises(self):
        pages = _make_pages(["X" * 1000])
        with pytest.raises(ValueError):
            chunk_pdf_pages(pages, chunk_size=100, overlap=100)

    def test_chunk_ids_sequential(self):
        pages = _make_pages(["A" * 2000])
        chunks = chunk_pdf_pages(pages, chunk_size=500, overlap=50)
        ids = [c["chunk_id"] for c in chunks]
        assert ids == list(range(len(ids)))

    def test_short_page_single_chunk(self):
        pages = _make_pages(["Short text."])
        chunks = chunk_pdf_pages(pages, chunk_size=800, overlap=150)
        assert len(chunks) == 1

    def test_overlap_creates_repeated_content(self):
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 40
        pages = _make_pages([text])
        chunks = chunk_pdf_pages(pages, chunk_size=100, overlap=20)
        if len(chunks) >= 2:
            end_of_first = chunks[0]["text"][-20:]
            start_of_second = chunks[1]["text"][:20]
            assert end_of_first == start_of_second


class TestCreateFaissIndex:
    def test_index_created(self, embedding_model):
        pages = _make_pages(["Hello world. " * 20])
        chunks = chunk_pdf_pages(pages)
        index, embeddings = create_faiss_index(chunks, embedding_model)
        assert index is not None
        assert embeddings.shape[0] == len(chunks)

    def test_embeddings_float32(self, embedding_model):
        pages = _make_pages(["Sample text. " * 20])
        chunks = chunk_pdf_pages(pages)
        _, embeddings = create_faiss_index(chunks, embedding_model)
        assert embeddings.dtype == np.float32

    def test_index_ntotal(self, embedding_model):
        pages = _make_pages(["A" * 2000])
        chunks = chunk_pdf_pages(pages)
        index, _ = create_faiss_index(chunks, embedding_model)
        assert index.ntotal == len(chunks)


class TestRetrieveRelevantChunks:
    @pytest.fixture
    def built_index(self, embedding_model):
        pages = _make_pages([
            "Revenue grew by 20% this year. Total ARR reached 1 million.",
            "Operating costs declined significantly. Profit margins improved.",
            "Customer acquisition cost fell by 15%. Retention rate increased.",
        ])
        chunks = chunk_pdf_pages(pages)
        from src.rag_utils import create_faiss_index
        index, _ = create_faiss_index(chunks, embedding_model)
        return chunks, index, embedding_model

    def test_returns_top_k(self, built_index):
        chunks, index, model = built_index
        results = retrieve_relevant_chunks("revenue", chunks, index, model, top_k=2)
        assert len(results) == 2

    def test_result_keys(self, built_index):
        chunks, index, model = built_index
        results = retrieve_relevant_chunks("revenue", chunks, index, model, top_k=1)
        assert {"chunk_id", "page_number", "text", "distance"}.issubset(results[0])

    def test_distance_is_float(self, built_index):
        chunks, index, model = built_index
        results = retrieve_relevant_chunks("revenue", chunks, index, model, top_k=1)
        assert isinstance(results[0]["distance"], float)

    def test_relevant_chunk_retrieved(self, built_index):
        chunks, index, model = built_index
        results = retrieve_relevant_chunks("profit margins", chunks, index, model, top_k=1)
        assert "profit" in results[0]["text"].lower() or "cost" in results[0]["text"].lower()

    def test_top_k_capped_by_index(self, built_index):
        chunks, index, model = built_index
        results = retrieve_relevant_chunks("revenue", chunks, index, model, top_k=100)
        assert len(results) <= len(chunks)
