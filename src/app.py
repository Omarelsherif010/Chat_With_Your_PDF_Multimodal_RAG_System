import streamlit as st
import os
from dotenv import load_dotenv
from PIL import Image
from pathlib import Path
import sys

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

def initialize_session_state():
    """Initialize session state variables"""
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False

def display_image(image_path):
    """Display image in Streamlit"""
    try:
        path = Path(image_path)
        if not path.exists():
            st.error(f"Image file not found: {image_path}")
            return
            
        try:
            image = Image.open(path)
            st.image(image, use_column_width=True, caption=f"Image from {path.name}")
        except Exception as e:
            st.error(f"Error opening image: {e}")
            
    except Exception as e:
        st.error(f"Error handling image path: {e}")

def main():
    # Disable Streamlit's file watcher for torch modules
    st.set_page_config(
        page_title="Research Paper Q&A System",
        page_icon="📚",
        layout="wide"
    )
    
    # Import MultimodalRetriever here to avoid torch class issues
    from retrieval_llama_parse import MultimodalRetriever
    
    st.title("Research Paper Q&A System")
    st.markdown("### Ask questions about the Attention Is All You Need paper")
    
    # Initialize session state
    initialize_session_state()
    
    # API Key input section
    with st.sidebar:
        st.header("API Configuration")
        openai_key = st.text_input("OpenAI API Key", type="password")
        
        # Update API key if provided
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
            
        # Initialize button
        if not st.session_state.initialized:
            if st.button("Initialize System"):
                with st.spinner("Initializing retrieval system... This may take a few minutes."):
                    try:
                        st.session_state.retriever = MultimodalRetriever()
                        # Load and store content
                        json_path = "llama_parse_output/llama_parse_output_4.json"
                        summaries_path = "llama_parse_summary/summaries_5.json"
                        st.session_state.retriever.load_and_store_content(json_path, summaries_path)
                        st.session_state.initialized = True
                        st.success("System initialized successfully!")
                    except Exception as e:
                        st.error(f"Error initializing system: {str(e)}")
    
    # Main content area
    if st.session_state.initialized:
        # Create two columns for input
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input("Enter your question about the paper:")
        with col2:
            show_sources = st.checkbox("Show sources", value=True)
        
        if st.button("Get Answer"):
            if query:
                with st.spinner("Retrieving and generating response..."):
                    try:
                        # Get retrieved content
                        retrieved_content = st.session_state.retriever.retrieve(query)
                        
                        # Generate response
                        response = st.session_state.retriever.generate_response(query, retrieved_content)
                        
                        # Display response
                        st.markdown("### Response")
                        st.markdown(response)
                        
                        if show_sources:
                            # Display sources in expandable sections
                            st.markdown("### Sources Used")
                            
                            # Text sources
                            if retrieved_content["texts"]:
                                with st.expander("📝 Text Sources", expanded=False):
                                    for doc in retrieved_content["texts"]:
                                        st.markdown(f"**Page {doc.metadata['page_num']}:**")
                                        st.markdown(doc.page_content)
                                        st.markdown(f"*Summary: {doc.metadata.get('summary', 'No summary available')}*")
                                        st.divider()
                            
                            # Table sources
                            if retrieved_content["tables"]:
                                with st.expander("📊 Table Sources", expanded=False):
                                    for doc in retrieved_content["tables"]:
                                        st.markdown(f"**{doc.metadata.get('title', 'Table')}**")
                                        st.markdown(f"Page {doc.metadata['page_num']}:")
                                        st.markdown(doc.page_content)
                                        st.divider()
                            
                            # Image sources
                            if retrieved_content["images"]:
                                with st.expander("🖼️ Image Sources", expanded=True):
                                    for doc in retrieved_content["images"]:
                                        st.markdown(f"**Image {doc.metadata.get('image_num', 'N/A')} "
                                                  f"(Page {doc.metadata.get('page_num', 'N/A')}):**")
                                        
                                        # Display image if path exists
                                        image_path = doc.metadata.get('image_path')
                                        if image_path:
                                            path = Path(image_path)
                                            if path.exists():
                                                st.info(f"Loading image from: {path}")
                                                display_image(path)
                                            else:
                                                st.warning(f"Image file not found at: {path}")
                                        else:
                                            st.warning("No image path provided in metadata")
                                            
                                        st.markdown(f"*Summary: {doc.metadata.get('summary', 'No summary available')}*")
                                        st.divider()
                            
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
            else:
                st.warning("Please enter a question.")
    else:
        st.info("Please initialize the system using the button in the sidebar.")

if __name__ == "__main__":
    main() 