import os
import uuid
try:
    from langchain_chroma import Chroma
except ImportError:
    logger.warning("langchain_chroma not found, falling back to legacy import")
    from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from data_extraction import extract_pdf_elements, display_saved_image
from data_summarize import summarize_texts, summarize_tables, summarize_images
from dotenv import load_dotenv
import json
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import time
from time import perf_counter
import logging
from dataclasses import dataclass
from config import settings

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Define paths
persist_directory = "./chroma_db"
file_path = 'data/attention_paper.pdf'
output_path = "./pdf_extracted_content/"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class RetrievalMetrics:
    text_time: float
    table_time: float
    image_time: float
    total_time: float

class MultimodalRetriever:
    """A retriever for handling multimodal content from research papers.
    
    This class manages the retrieval of text, tables, and images from a research paper,
    using vector stores for similarity search and maintaining metadata about the content.
    
    Attributes:
        persist_directory (str): Directory for vector store persistence
        output_path (str): Path for extracted content
        batch_size (int): Size of batches for processing
        
    Methods:
        retrieve: Retrieve relevant content for a query
        initialize_vectorstores: Set up vector stores with content
        cleanup: Clean up resources
    """
    def __init__(self, persist_directory: str = "./chroma_db", 
                 output_path: str = "./pdf_extracted_content/",
                 batch_size: int = 5):
        """Initialize the retriever with configurable settings"""
        # Load environment variables
        load_dotenv()
        
        # Store configuration
        self.persist_directory = persist_directory
        self.output_path = output_path
        self.batch_size = batch_size
        
        # Initialize components
        self._initialize_components()
        
        # Register cleanup on exit
        import atexit
        atexit.register(self.cleanup)
    
    def _initialize_components(self):
        """Initialize embeddings, stores and other components"""
        # Debug print (remove after testing)
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and api_key.startswith("sk-"):
            logger.info(f"API key loaded successfully: {api_key[:10]}...")
        else:
            logger.warning(f"Warning: API key not found or invalid format")
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
        
        # Initialize stores
        self.text_vectorstore = Chroma(
            collection_name="text_store",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        self.table_vectorstore = Chroma(
            collection_name="table_store",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        self.image_vectorstore = Chroma(
            collection_name="image_store",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        # Initialize document stores
        self.text_store = InMemoryStore()
        self.table_store = InMemoryStore()
        self.image_store = InMemoryStore()
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize LLM
        self.llm = ChatOpenAI(model=settings.MODEL_NAME, temperature=0.7)
        
        # Set up retrieval filters
        self.retrieval_filters = {
            "text": lambda x: x["type"] == "text",
            "equation": lambda x: "equation" in x["section_type"].lower(),
            "technical": lambda x: any(term in x["content"].lower() 
                                    for term in ["algorithm", "formula", "equation", "implementation"])
        }

    def cleanup(self):
        """Clean up resources"""
        try:
            logger.info("Cleaning up vector stores...")
            try:
                if hasattr(self, 'text_vectorstore'):
                    self.text_vectorstore.delete_collection()
            except Exception as e:
                logger.warning(f"Error cleaning text store: {str(e)}")
            
            try:
                if hasattr(self, 'table_vectorstore'):
                    self.table_vectorstore.delete_collection()
            except Exception as e:
                logger.warning(f"Error cleaning table store: {str(e)}")
            
            try:
                if hasattr(self, 'image_vectorstore'):
                    self.image_vectorstore.delete_collection()
            except Exception as e:
                logger.warning(f"Error cleaning image store: {str(e)}")
            
            # Clear document stores
            if hasattr(self, 'text_store'):
                self.text_store.clear()
            if hasattr(self, 'table_store'):
                self.table_store.clear()
            if hasattr(self, 'image_store'):
                self.image_store.clear()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def prepare_text_documents(self, texts, summaries):
        """Prepare text documents with their summaries"""
        documents = []
        doc_ids = []
        
        for text, summary in zip(texts, summaries):
            doc_id = str(uuid.uuid4())
            doc_ids.append(doc_id)
            
            metadata = {
                "doc_id": doc_id,
                "type": "text",
                "section_type": text["metadata"]["type"],
                "page": text["metadata"]["page"],
                "summary": summary
            }
            
            chunks = self.text_splitter.split_text(text["content"])
            for chunk in chunks:
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata=metadata
                    )
                )
        
        return documents, doc_ids, texts
    
    def prepare_table_documents(self, tables, summaries):
        """Prepare table documents with their summaries"""
        documents = []
        doc_ids = []
        
        for table, summary in zip(tables, summaries):
            doc_id = str(uuid.uuid4())
            doc_ids.append(doc_id)
            
            metadata = {
                "doc_id": doc_id,
                "type": "table",
                "page": table["metadata"]["page"],
                "caption": table["metadata"].get("caption", "No caption"),
                "summary": summary
            }
            
            documents.append(
                Document(
                    page_content=table["content"]["text"],
                    metadata=metadata
                )
            )
        
        return documents, doc_ids, tables
    
    def prepare_image_documents(self, images, summaries):
        """Prepare image documents with their summaries"""
        documents = []
        doc_ids = []
        
        for image, summary in zip(images, summaries):
            doc_id = str(uuid.uuid4())
            doc_ids.append(doc_id)
            
            metadata = {
                "doc_id": doc_id,
                "type": "image",
                "page": image["metadata"]["page"],
                "caption": image["metadata"].get("caption", "No caption"),
                "filename": image["metadata"]["filename"],
                "filepath": os.path.join(self.output_path, image["metadata"]["filename"]),
                "summary": summary
            }
            
            content = f"""Image caption: {image['metadata'].get('caption', 'No caption')}
                         Summary: {summary}"""
            
            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata
                )
            )
        
        return documents, doc_ids, images
    
    def add_documents_with_timeout(self, vectorstore, docs, timeout=300):
        """Add documents to vectorstore with timeout"""
        with ThreadPoolExecutor() as executor:
            future = executor.submit(vectorstore.add_documents, docs)
            try:
                future.result(timeout=timeout)  # 5 minutes timeout
                return True
            except TimeoutError:
                print(f"Operation timed out after {timeout} seconds")
                return False

    def initialize_vectorstores(self):
        """Initialize vector stores with content and summaries"""
        try:
            # Cleanup existing stores
            self.cleanup()
            
            # Extract content
            logger.info("Extracting PDF elements...")
            texts, images, tables = extract_pdf_elements(settings.pdf_path)
            if not texts and not images and not tables:
                logger.error("No content extracted from PDF")
                return False
            
            # Limit data for testing
            texts = texts[:20]  # Process only first 20 texts for testing
            print(f"Using first {len(texts)} text sections for testing...")
            
            print("Generating summaries...")
            text_summaries = summarize_texts(texts)
            table_summaries = summarize_tables(tables)
            image_summaries = summarize_images([img['content'] for img in images])
            
            # Prepare documents
            print("Preparing documents...")
            text_docs, text_ids, text_originals = self.prepare_text_documents(texts, text_summaries)
            table_docs, table_ids, table_originals = self.prepare_table_documents(tables, table_summaries)
            image_docs, image_ids, image_originals = self.prepare_image_documents(images, image_summaries)
            
            print(f"Processing {len(texts)} texts, {len(tables)} tables, {len(images)} images...")
            
            # Add documents to vector stores with progress updates
            print("Adding documents to vector stores...")
            if not self.add_documents_with_timeout(self.text_vectorstore, text_docs):
                raise Exception("Failed to add documents to text vector store")
            if not self.add_documents_with_timeout(self.table_vectorstore, table_docs):
                raise Exception("Failed to add documents to table vector store")
            if not self.add_documents_with_timeout(self.image_vectorstore, image_docs):
                raise Exception("Failed to add documents to image vector store")
            
            # Store original documents
            print("Storing original documents...")
            self.text_store.mset(list(zip(text_ids, text_originals)))
            self.table_store.mset(list(zip(table_ids, table_originals)))
            self.image_store.mset(list(zip(image_ids, image_originals)))
            
            print("Vector stores initialized successfully!")
            return True
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            return False
    
    def retrieve(self, query: str, k: int = 2) -> dict:
        """Enhanced retrieval with better error handling"""
        metrics = RetrievalMetrics(0, 0, 0, 0)
        start_time = perf_counter()
        
        results = {
            "texts": [],
            "tables": [],
            "images": [],
            "metrics": metrics
        }
        
        try:
            # Measure text retrieval
            text_start = perf_counter()
            try:
                results["texts"] = self.text_vectorstore.similarity_search(query, k=k)
            except Exception as e:
                logger.error(f"Text retrieval failed: {str(e)}")
            finally:
                metrics.text_time = perf_counter() - text_start
            
            # Measure table retrieval
            table_start = perf_counter()
            try:
                results["tables"] = self.table_vectorstore.similarity_search(query, k=k)
            except Exception as e:
                logger.error(f"Table retrieval failed: {str(e)}")
            finally:
                metrics.table_time = perf_counter() - table_start
            
            # Measure image retrieval
            image_start = perf_counter()
            try:
                results["images"] = self.image_vectorstore.similarity_search(query, k=k)
            except Exception as e:
                logger.error(f"Image retrieval failed: {str(e)}")
            finally:
                metrics.image_time = perf_counter() - image_start
            
            metrics.total_time = perf_counter() - start_time
            logger.info(f"Retrieval metrics: {metrics}")
            
            return results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            raise
    
    def generate_response(self, query, retrieved_content):
        """Generate response using retrieved content and summaries"""
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
        
        # Format retrieved content with summaries
        text_sections = "\n".join([f"- {doc.page_content}" for doc in retrieved_content["texts"]])
        text_summaries = "\n".join([f"- {doc.metadata['summary']}" for doc in retrieved_content["texts"]])
        
        tables = "\n".join([f"- {doc.page_content}" for doc in retrieved_content["tables"]])
        table_summaries = "\n".join([f"- {doc.metadata['summary']}" for doc in retrieved_content["tables"]])
        
        images = "\n".join([f"- {doc.page_content}" for doc in retrieved_content["images"]])
        image_summaries = "\n".join([f"- {doc.metadata['summary']}" for doc in retrieved_content["images"]])
        
        # Create prompt
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # Generate response
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({
            "text_sections": text_sections,
            "text_summaries": text_summaries,
            "tables": tables,
            "table_summaries": table_summaries,
            "images": images,
            "image_summaries": image_summaries,
            "query": query
        })
        
        return response

    def analyze_image_elements(self, image_doc):
        """Detailed analysis of image components"""
        return {
            "type": image_doc.metadata.get("type", "unknown"),
            "components": self._extract_image_components(image_doc),
            "relationships": self._analyze_relationships(image_doc),
            "key_points": self._identify_key_points(image_doc)
        }

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
