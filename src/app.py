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
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def display_base64_image(base64_str):
    """Display base64 image in Streamlit"""
    try:
        if base64_str.startswith('data:image'):
            # Remove the data URI prefix
            base64_str = base64_str.split(',')[1]
        st.image(base64.b64decode(base64_str))
    except Exception as e:
        st.error(f"Error displaying image: {str(e)}")

def format_source_text(text_source):
    """Format text source for display"""
    try:
        return {
            'content': text_source.get('content', ''),
            'page': text_source.get('metadata', {}).get('page', 'Unknown'),
            'type': text_source.get('metadata', {}).get('type', 'text'),
            'summary': text_source.get('metadata', {}).get('summary', 'No summary available')
        }
    except Exception as e:
        st.error(f"Error formatting text source: {str(e)}")
        return {
            'content': str(text_source),
            'page': 'Unknown',
            'type': 'text',
            'summary': 'Error formatting source'
        }

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
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "question": query,
                            "answer": result["response"] if show_sources else result,
                            "sources": result.get("sources") if show_sources else None
                        })
                        
                        # Display latest response
                        st.markdown("### Response")
                        st.markdown(st.session_state.chat_history[-1]["answer"])
                        
                        if show_sources and "sources" in result:
                            st.markdown("### Sources Used")
                            
                            # Text sources
                            if result["sources"].get("texts"):
                                with st.expander("📝 Text Sources", expanded=False):
                                    for i, text in enumerate(result["sources"]["texts"]):
                                        formatted_text = format_source_text(text)
                                        st.markdown(f"**Text {i+1} (Page {formatted_text['page']}):**")
                                        st.markdown(formatted_text['content'])
                                        st.markdown(f"*Summary: {formatted_text['summary']}*")
                                        st.divider()
                            
                            # Table sources
                            if result["sources"].get("tables"):
                                with st.expander("📊 Table Sources", expanded=False):
                                    for i, table in enumerate(result["sources"]["tables"]):
                                        st.markdown(f"**Table {i+1} (Page {table['metadata'].get('page', 'Unknown')}):**")
                                        st.markdown(table['text'])
                                        st.markdown(f"*Caption: {table['metadata'].get('caption', 'No caption')}*")
                                        st.divider()
                            
                            # Image sources
                            if result["sources"].get("images"):
                                with st.expander("🖼️ Image Sources", expanded=True):
                                    for i, img in enumerate(result["sources"]["images"]):
                                        st.markdown(f"**Image {i+1} (Page {img['metadata'].get('page', 'Unknown')}):**")
                                        if os.path.exists(img['metadata'].get('filepath', '')):
                                            st.image(img['metadata']['filepath'])
                                        st.markdown(f"**Caption:** {img['metadata'].get('caption', 'No caption')}")
                                        st.markdown(f"*Summary: {img['metadata'].get('summary', 'No summary available')}*")
                                        st.divider()
                    
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
            else:
                st.warning("Please enter a question.")
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("### Chat History")
            for i, chat in enumerate(reversed(st.session_state.chat_history[:-1]), 1):
                with st.expander(f"Q: {chat['question']}", expanded=False):
                    st.markdown(chat['answer'])
    else:
        st.info("Please initialize the system using the button in the sidebar.")

if __name__ == "__main__":
    main() 