import os
import uuid
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from data_extraction import extract_pdf_elements, display_saved_image, PDFParser
from data_summarize import summarize_texts, summarize_tables, summarize_images
from dotenv import load_dotenv
import json
from functools import lru_cache
import streamlit as st

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Define paths
persist_directory = "./chroma_db"
file_path = 'data/attention_paper.pdf'
output_path = "./pdf_extracted_content/"


class MultimodalRetriever:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize Pinecone with environment
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        
        # Use a single index with namespaces
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "multimodal-store")
        
        try:
            # Try to get existing index
            self.index = pc.Index(self.index_name)
            print(f"Found existing {self.index_name} index")
        except Exception as e:
            print(f"Creating new {self.index_name} index...")
            # Create new index if it doesn't exist
            self.index = pc.create_index(
                name=self.index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        
        # Initialize vector store with namespaces and index name
        self.vectorstore = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings,
            text_key="text",
            index_name=self.index_name
        )
        
        # Initialize document stores
        self.text_store = InMemoryStore()
        self.table_store = InMemoryStore()
        self.image_store = InMemoryStore()
        
        # Improve text splitting for better context
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Smaller chunks for more precise retrieval
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize LLM for response generation
        self.llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0.7)
        
        # Update retrieval filters to use 'type' instead of 'section_type'
        self.retrieval_filters = {
            "text": lambda x: x.get("type") == "text",
            "equation": lambda x: x.get("type") == "equation",
            "technical": lambda x: any(term in x.get("content", "").lower() 
                                    for term in ["algorithm", "formula", "equation", "implementation"])
        }

    def initialize_vectorstores(self):
        """Initialize vector stores with content"""
        try:
            # Initialize parser and extract content
            parser = PDFParser()
            print("Extracting PDF elements...")
            result = parser.extract_elements(file_path)
            
            # Generate summaries
            print("Generating summaries...")
            text_summaries = summarize_texts(result['texts'])
            table_summaries = summarize_tables(result['tables'])
            image_summaries = summarize_images(result['images'])
            
            # Prepare documents for vector store
            print("Preparing documents...")
            text_docs = self.prepare_text_documents(result['texts'], text_summaries)
            table_docs = self.prepare_table_documents(result['tables'], table_summaries)
            image_docs = self.prepare_image_documents(result['images'], image_summaries)
            
            # Add to vector store
            print("Adding to vector store...")
            self.add_to_vectorstore(text_docs, table_docs, image_docs)
            
            print("Vector stores initialized successfully!")
            return True
            
        except Exception as e:
            print(f"Error initializing vector stores: {e}")
            return False

    def prepare_text_documents(self, texts, summaries):
        """Prepare text documents with summaries"""
        docs = []
        for text, summary in zip(texts, summaries):
            try:
                doc_id = str(uuid.uuid4())
                doc = Document(
                    page_content=text['content'],
                    metadata={
                        'id': doc_id,
                        'type': text['metadata']['type'],
                        'page': text['metadata']['page'],
                        'font_info': text['metadata']['font_info'],
                        'summary': summary
                    }
                )
                docs.append(doc)
            except Exception as e:
                print(f"Error preparing text document: {e}")
        return docs

    def prepare_table_documents(self, tables, summaries):
        """Prepare table documents with summaries"""
        docs = []
        for table, summary in zip(tables, summaries):
            try:
                doc_id = str(uuid.uuid4())
                content = table.get('html', '') or str(table.get('data', ''))
                doc = Document(
                    page_content=content,
                    metadata={
                        'id': doc_id,
                        'type': 'table',
                        'page': table['metadata']['page'],
                        'caption': table['metadata'].get('caption', ''),
                        'summary': summary
                    }
                )
                docs.append(doc)
            except Exception as e:
                print(f"Error preparing table document: {e}")
        return docs

    def prepare_image_documents(self, images, summaries):
        """Prepare image documents with summaries"""
        docs = []
        for image, summary in zip(images, summaries):
            try:
                doc_id = str(uuid.uuid4())
                doc = Document(
                    page_content=summary,  # Use summary as content
                    metadata={
                        'id': doc_id,
                        'type': 'image',
                        'page': image['metadata']['page'],
                        'caption': image['metadata'].get('caption', ''),
                        'filepath': image['filepath'],
                        'summary': summary
                    }
                )
                docs.append(doc)
            except Exception as e:
                print(f"Error preparing image document: {e}")
        return docs

    def add_to_vectorstore(self, text_docs, table_docs, image_docs):
        """Add documents to vector store"""
        try:
            # Clear existing vectors
            print("Clearing existing vectors...")
            for namespace in ["text", "table", "image"]:
                try:
                    self.index.delete(delete_all=True, namespace=namespace)
                    print(f"Cleared vectors from {namespace} namespace")
                except Exception as e:
                    print(f"No existing vectors in {namespace} namespace")
            
            # Add documents in batches with verification
            batch_size = 100
            for namespace, docs in [
                ("text", text_docs),
                ("table", table_docs),
                ("image", image_docs)
            ]:
                if docs:
                    print(f"Adding {len(docs)} documents to {namespace} namespace...")
                    for i in range(0, len(docs), batch_size):
                        batch = docs[i:i + batch_size]
                        self.vectorstore.add_documents(batch, namespace=namespace)
                        print(f"Added batch {i//batch_size + 1} to {namespace} namespace")
            
            # Store originals
            print("Storing original documents...")
            if text_docs:
                self.text_store.mset(list(zip(text_docs, text_docs)))
            if table_docs:
                self.table_store.mset(list(zip(table_docs, table_docs)))
            if image_docs:
                self.image_store.mset(list(zip(image_docs, image_docs)))
            
        except Exception as e:
            print(f"Error adding to vector store: {str(e)}")
            raise
    
    @lru_cache(maxsize=100)
    def retrieve(self, query, k=2):
        """Retrieve relevant content with caching"""
        try:
            results = {
                "texts": self.vectorstore.similarity_search(
                    query, k=k, namespace="text",
                    filter={"type": {"$in": ["text", "title", "heading", "paragraph"]}}
                ),
                "tables": self.vectorstore.similarity_search(
                    query, k=k, namespace="table"
                ),
                "images": self.vectorstore.similarity_search(
                    query, k=k, namespace="image"
                )
            }
            
            # Display retrieved images using Streamlit
            print("\nRetrieved Images:")
            for img_doc in results["images"]:
                print(f"\nImage Summary: {img_doc.metadata['summary']}")
                print(f"Caption: {img_doc.metadata['caption']}")
                if os.path.exists(img_doc.metadata['filepath']):
                    img = display_saved_image(img_doc.metadata['filepath'])
                    if img:
                        # Use st.container to avoid threading warnings
                        with st.container():
                            st.image(img, caption=img_doc.metadata['caption'])
            
            return results
            
        except Exception as e:
            print(f"Error during retrieval: {str(e)}")
            return {
                "texts": [],
                "tables": [],
                "images": []
            }
    
    def generate_response(self, query, retrieved_content):
        """Generate response using retrieved content and summaries"""
        try:
            prompt_template = """You are an AI assistant helping with questions about a research paper on transformers.
            Use the following retrieved content to answer the question. Include relevant information from texts, tables, 
            and images if available.

            Retrieved Text Sections:
            {text_sections}
            
            Text Summaries:
            {text_summaries}

            Retrieved Tables:
            {tables}
            
            Table Summaries:
            {table_summaries}

            Retrieved Images:
            {images}
            
            Image Summaries:
            {image_summaries}

            Question: {query}

            Provide a comprehensive answer using the retrieved information. If referring to images or tables, 
            be specific about which ones you're referencing. Use the summaries to provide context but rely on 
            the full content for detailed information.
            """
            
            # Format retrieved content with summaries - with safe access and type checking
            text_sections = "\n".join([
                f"- [{doc.metadata.get('type', 'text')}] {doc.page_content}" 
                for doc in retrieved_content.get("texts", [])
            ])
            text_summaries = "\n".join([
                f"- [{doc.metadata.get('type', 'text')}] {doc.metadata.get('summary', '')}" 
                for doc in retrieved_content.get("texts", [])
            ])
            
            tables = "\n".join([
                f"- {doc.page_content}" for doc in retrieved_content.get("tables", [])
            ])
            table_summaries = "\n".join([
                f"- {doc.metadata.get('summary', '')}" for doc in retrieved_content.get("tables", [])
            ])
            
            images = "\n".join([
                f"- {doc.page_content}" for doc in retrieved_content.get("images", [])
            ])
            image_summaries = "\n".join([
                f"- {doc.metadata.get('summary', '')}" for doc in retrieved_content.get("images", [])
            ])
            
            # Create prompt
            prompt = ChatPromptTemplate.from_template(prompt_template)
            
            # Generate response
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({
                "text_sections": text_sections or "No relevant text sections found.",
                "text_summaries": text_summaries or "No text summaries available.",
                "tables": tables or "No relevant tables found.",
                "table_summaries": table_summaries or "No table summaries available.",
                "images": images or "No relevant images found.",
                "image_summaries": image_summaries or "No image summaries available.",
                "query": query
            })
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while generating the response. Please try again."

def main():
    # Initialize retriever
    retriever = MultimodalRetriever()
    
    # Initialize vector stores
    print("Initializing vector stores...")
    retriever.initialize_vectorstores()
    
    # Interactive query loop
    while True:
        query = input("\nEnter your question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
            
        print("\nRetrieving relevant content...")
        retrieved_content = retriever.retrieve(query)
        
        # Print detailed retrieval results
        print("\nRetrieved Text Sections:")
        for doc in retrieved_content["texts"]:
            print(f"\nPage {doc.metadata['page']}:")
            print(f"Summary: {doc.metadata['summary']}")
            print(f"Content: {doc.page_content[:200]}...")
            
        print("\nRetrieved Tables:")
        for doc in retrieved_content["tables"]:
            print(f"\nPage {doc.metadata['page']}:")
            print(f"Caption: {doc.metadata['caption']}")
            print(f"Summary: {doc.metadata['summary']}")
        
        print("\nGenerating response...")
        response = retriever.generate_response(query, retrieved_content)
        
        print("\nResponse:")
        print(response)
        print("-" * 80)

if __name__ == "__main__":
    main()
