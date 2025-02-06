import streamlit as st
import os
from main import MultimodalRAG
from dotenv import load_dotenv
import base64

# Load environment variables
load_dotenv()

def initialize_session_state():
    """Initialize session state variables"""
    if 'rag' not in st.session_state:
        st.session_state.rag = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False

def display_base64_image(base64_str):
    """Display base64 image in Streamlit"""
    if base64_str.startswith('data:image'):
        # Remove the data URI prefix
        base64_str = base64_str.split(',')[1]
    st.image(base64.b64decode(base64_str))

def main():
    st.set_page_config(
        page_title="Research Paper Q&A System",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("Research Paper Q&A System")
    st.markdown("### Ask questions about the Attention Is All You Need paper")
    
    # Initialize session state
    initialize_session_state()
    
    # API Key input section
    with st.sidebar:
        st.header("API Configuration")
        openai_key = st.text_input("OpenAI API Key (optional)", type="password")
        groq_key = st.text_input("Groq API Key (optional)", type="password")
        
        # Update API keys if provided
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
        if groq_key:
            os.environ["GROQ_API_KEY"] = groq_key
            
        # Initialize button
        if not st.session_state.initialized:
            if st.button("Initialize System"):
                with st.spinner("Initializing RAG system... This may take a few minutes."):
                    try:
                        st.session_state.rag = MultimodalRAG()
                        st.session_state.rag.initialize()
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
                with st.spinner("Generating response..."):
                    try:
                        result = st.session_state.rag.query(query, return_sources=show_sources)
                        
                        if show_sources:
                            # Display response in a container
                            with st.container():
                                st.markdown("### Response")
                                st.markdown(result["response"])
                            
                            # Display sources in expandable sections
                            st.markdown("### Sources Used")
                            
                            # Text sources
                            if result["sources"]["texts"]:
                                with st.expander("📝 Text Sources", expanded=False):
                                    for i, text in enumerate(result["sources"]["texts"]):
                                        st.markdown(f"**Text {i+1} (Page {text['metadata']['page']}):**")
                                        st.markdown(text['content'])
                                        st.markdown(f"*Summary: {text['metadata']['summary']}*")
                                        st.divider()
                            
                            # Table sources
                            if result["sources"]["tables"]:
                                with st.expander("📊 Table Sources", expanded=False):
                                    for i, table in enumerate(result["sources"]["tables"]):
                                        st.markdown(f"**Table {i+1} (Page {table['metadata']['page']}):**")
                                        st.markdown(table['content'])
                                        st.markdown(f"*Summary: {table['metadata']['summary']}*")
                                        st.divider()
                            
                            # Image sources
                            if result["sources"]["images"]:
                                with st.expander("🖼️ Image Sources", expanded=True):
                                    for i, img in enumerate(result["sources"]["images"]):
                                        st.markdown(f"**Image {i+1} (Page {img['metadata']['page']}):**")
                                        display_base64_image(img['content'])
                                        st.markdown(f"**Caption:** {img['metadata']['caption']}")
                                        st.markdown(f"*Summary: {img['metadata']['summary']}*")
                                        st.divider()
                        else:
                            st.markdown("### Response")
                            st.markdown(result)
                            
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
            else:
                st.warning("Please enter a question.")
    else:
        st.info("Please initialize the system using the button in the sidebar.")

if __name__ == "__main__":
    main() 