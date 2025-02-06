import pytest
from src.main import MultimodalRAG
from src.config import settings

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