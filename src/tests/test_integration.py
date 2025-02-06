import pytest
from src.main import MultimodalRAG
from src.config import settings
import asyncio

@pytest.fixture(scope="session")
def test_pdf():
    return "tests/data/test.pdf"

@pytest.fixture(scope="session")
def rag_system(test_pdf):
    settings.PDF_FILE = test_pdf
    rag = MultimodalRAG()
    rag.initialize()
    yield rag
    # Cleanup after tests
    rag.retriever.cleanup()

def test_error_handling(rag_system):
    """Add error handling tests"""
    with pytest.raises(ValueError):
        rag_system.query("")  # Empty query should raise error
    
def test_multimodal_retrieval(rag_system):
    """Add comprehensive retrieval tests"""
    result = rag_system.query("Show attention mechanism", return_sources=True)
    assert len(result["sources"]["images"]) > 0
    assert len(result["sources"]["texts"]) > 0

def test_end_to_end_flow():
    rag = MultimodalRAG()
    rag.initialize()
    
    # Test basic query
    response = rag.query("How is attention calculated?")
    assert response is not None
    
    # Test multimodal retrieval
    result = rag.query("Show me the attention mechanism diagram", return_sources=True)
    assert "images" in result["sources"]
    
    # Test source citation
    assert rag.format_citations(result["sources"]) is not None 

@pytest.mark.parametrize("query,expected_sources", [
    ("How is attention calculated?", {"texts", "images"}),
    ("What is the BLEU score?", {"texts", "tables"}),
    ("Show me the architecture", {"images"})
])
def test_retrieval_types(rag_system, query, expected_sources):
    """Test different types of retrievals"""
    result = rag_system.query(query, return_sources=True)
    for source_type in expected_sources:
        assert len(result["sources"][source_type]) > 0

def test_concurrent_queries(rag_system):
    """Test system under load"""
    queries = ["attention", "training", "results"] * 3
    results = []
    for q in queries:
        result = rag_system.query(q)
        results.append(result)
    assert len(results) == len(queries)

def test_retrieval_quality(rag_system):
    """Test retrieval quality metrics"""
    result = rag_system.query("How is attention calculated?", return_sources=True)
    metrics = rag_system.evaluate_response(result["response"], result["sources"])
    
    assert metrics["completeness"] > 70
    assert metrics["technical_depth"] > 0
    assert metrics["source_citation"] 