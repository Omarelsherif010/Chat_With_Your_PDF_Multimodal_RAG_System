import streamlit as st
from main import MultimodalRAG
import logging
import time
from PIL import Image
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_rag_system():
    """Initialize RAG system if not already in session state"""
    if 'rag_system' not in st.session_state:
        try:
            with st.spinner('Initializing RAG system...'):
                rag = MultimodalRAG()
                if not rag.initialize():
                    st.error("Failed to initialize RAG system")
                    return False
                st.session_state.rag_system = rag
                return True
        except Exception as e:
            st.error(f"Error during initialization: {str(e)}")
            logger.error(f"Initialization error: {str(e)}", exc_info=True)
            return False
    return True

def display_image(base64_str):
    """Display base64 encoded image"""
    try:
        img_bytes = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_bytes))
        st.image(img, use_column_width=True)
    except Exception as e:
        st.error(f"Error displaying image: {str(e)}")

def display_metrics(metrics):
    """Display response quality metrics"""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Completeness", f"{metrics['completeness']:.1f}%")
    with col2:
        st.metric("Technical Depth", metrics['technical_depth'])
    with col3:
        st.metric("Visual References", "Yes" if metrics['visual_reference'] else "No")
    
    col4, col5 = st.columns(2)
    with col4:
        st.metric("Mathematical Clarity", "Yes" if metrics['mathematical_clarity'] else "No")
    with col5:
        st.metric("Source Citations", "Yes" if metrics['source_citation'] else "No")

def main():
    if not init_rag_system():
        return
    
    st.set_page_config(
        page_title="Research Paper RAG System",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Research Paper RAG System")
    st.markdown("""
    This system helps you interact with the research paper "Attention Is All You Need".
    Ask questions about the paper's content, methodology, or results.
    """)
    
    # System health check
    if st.sidebar.button("Check System Health"):
        health_status = st.session_state.rag_system.health_check()
        st.sidebar.json(health_status)
    
    # Show system metrics
    if st.sidebar.button("Show System Metrics"):
        metrics = st.session_state.rag_system.collect_metrics()
        st.sidebar.json(metrics)
    
    # Main query interface
    query = st.text_input("Enter your question:")
    return_sources = st.checkbox("Show sources")
    
    if st.button("Submit") and query:
        try:
            with st.spinner('Processing query...'):
                start_time = time.time()
                result = st.session_state.rag_system.query(query, return_sources=return_sources)
                
                # Display response
                st.markdown("### Response")
                if return_sources:
                    st.markdown(result["response"])
                    
                    # Display retrieved images
                    if result["sources"]["images"]:
                        st.markdown("### Retrieved Images")
                        for img in result["sources"]["images"]:
                            caption = img.metadata.get('caption', 'No caption')
                            filepath = img.metadata.get('filepath')
                            st.markdown(f"**Caption:** {caption}")
                            try:
                                with open(filepath, 'r') as f:
                                    base64_str = f.read()
                                    display_image(base64_str)
                            except Exception as e:
                                st.error(f"Error loading image: {str(e)}")
                    
                    # Display metrics
                    st.markdown("### Response Quality Metrics")
                    display_metrics(st.session_state.rag_system.evaluate_response(
                        result["response"], 
                        result["sources"]
                    ))
                    
                    # Display citations
                    st.markdown("### Sources")
                    st.markdown(st.session_state.rag_system.format_citations(result["sources"]))
                else:
                    st.markdown(result)
                
                st.sidebar.info(f"Query processed in {time.time() - start_time:.2f} seconds")
                
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            logger.error(f"Query processing error: {str(e)}", exc_info=True)
    
    # Display usage tips
    with st.sidebar.expander("Usage Tips"):
        st.markdown("""
        - Ask specific questions about the paper
        - Check 'Show sources' to see supporting evidence
        - View system health and metrics for diagnostics
        - Images and tables are displayed when relevant
        """)

if __name__ == "__main__":
    main() 