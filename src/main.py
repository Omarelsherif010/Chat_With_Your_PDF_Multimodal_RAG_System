import os
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from base64 import b64decode, b64encode
from dotenv import load_dotenv
import json
from retrieval import MultimodalRetriever
import logging
from pydantic import BaseModel, Field
from config import settings
import atexit
import signal
from dataclasses import dataclass, asdict
import time
from typing import Dict, List, Optional, Any

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Add proper logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class QueryInput(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    return_sources: bool = Field(default=False)

@dataclass
class SystemMetrics:
    query_count: int = 0
    avg_response_time: float = 0
    error_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

class MultimodalRAG:
    def __init__(self):
        self.chain = None
        self.chain_with_sources = None
        # Initialize retriever with settings
        self.retriever = MultimodalRetriever(
            persist_directory=settings.PERSIST_DIRECTORY,
            output_path=settings.OUTPUT_PATH,
            batch_size=settings.BATCH_SIZE
        )
        
        # Register shutdown handlers
        atexit.register(self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        signal.signal(signal.SIGINT, self.shutdown)
        
        # Add error handling wrapper
        def error_handler(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Error in {func.__name__}: {str(e)}")
                    # Log error and potentially retry or recover
                    return None
            return wrapper
            
        # Apply to key methods
        self.query = error_handler(self.query)
        self.initialize = error_handler(self.initialize)
        
        self.metrics = SystemMetrics()
    
    def initialize(self):
        """Initialize the system"""
        try:
            # First create chains
            self.create_chain()
            
            # Then initialize retriever
            success = self.retriever.initialize_vectorstores()
            if not success:
                logger.error("Failed to initialize vector stores")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            return False
    
    def parse_retrieved_content(self, retrieved_docs):
        """Parse retrieved content into text and images"""
        texts = []
        images = []
        tables = []
        
        # Process text documents
        for doc in retrieved_docs["texts"]:
            texts.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
            
        # Process table documents
        for doc in retrieved_docs["tables"]:
            tables.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
            
        # Process image documents
        for doc in retrieved_docs["images"]:
            if os.path.exists(doc.metadata['filepath']):
                try:
                    # Read the base64 content
                    with open(doc.metadata['filepath'], 'r') as img_file:
                        base64_content = img_file.read()
                    
                    # Add proper data URI prefix for images
                    img_content = f"data:image/png;base64,{base64_content}"
                    
                    images.append({
                        "content": img_content,
                        "metadata": doc.metadata
                    })
                except Exception as e:
                    print(f"Warning: Could not process image {doc.metadata['filepath']}: {str(e)}")
        
        return {
            "texts": texts,
            "tables": tables,
            "images": images
        }
    
    def build_prompt(self, inputs: dict) -> str:
        """Build prompt from inputs"""
        context = inputs["context"]
        question = inputs["question"]
        
        # Format context sections
        text_sections = "\n".join([f"Text: {doc.page_content}" for doc in context["texts"]])
        table_sections = "\n".join([f"Table: {doc.page_content}" for doc in context["tables"]])
        image_sections = "\n".join([f"Image: {doc.page_content}" for doc in context["images"]])
        
        prompt = f"""Use the following retrieved information to answer the question.
        
        Context:
        {text_sections}
        
        Tables:
        {table_sections}
        
        Images:
        {image_sections}
        
        Question: {question}
        
        Answer the question based on the provided context. If referring to images or tables, 
        be specific about which ones you're referencing. If you don't find relevant information, 
        say so."""
        
        return prompt
    
    def _format_text_sections(self, texts):
        """Format text sections with better structure"""
        return "\n\n".join([
            f"Section {i+1} (Page {doc['metadata']['page']}):\n"
            f"Type: {doc['metadata']['section_type']}\n"
            f"Content: {doc['content']}\n"
            f"Summary: {doc['metadata']['summary']}"
            for i, doc in enumerate(texts)
        ])
    
    def _format_technical_details(self, texts):
        """Format technical details from texts"""
        technical_sections = [
            doc for doc in texts 
            if any(term in doc['content'].lower() 
                  for term in ['algorithm', 'implementation', 'computation', 'procedure'])
        ]
        
        if not technical_sections:
            return "No technical details found."
            
        return "\n\n".join([
            f"Technical Detail {i+1}:\n{doc['content']}"
            for i, doc in enumerate(technical_sections)
        ])

    def _format_equations(self, texts):
        """Format equations from texts"""
        equations = [
            doc for doc in texts
            if any(symbol in doc['content'] 
                  for symbol in ['=', '∑', '∏', '∫', '√', '≈', '≠', '≤', '≥'])
        ]
        
        if not equations:
            return "No equations found."
            
        return "\n\n".join([
            f"Equation {i+1}:\n{doc['content']}"
            for i, doc in enumerate(equations)
        ])

    def _format_tables(self, tables):
        """Format tables with better structure"""
        if not tables:
            return "No tables found."
            
        return "\n\n".join([
            f"Table {i+1} (Page {doc['metadata']['page']}):\n"
            f"Caption: {doc['metadata'].get('caption', 'No caption')}\n"
            f"Content: {doc['content']}\n"
            f"Summary: {doc['metadata']['summary']}"
            for i, doc in enumerate(tables)
        ])

    def _format_code_examples(self, texts):
        """Extract and format code implementations"""
        code_sections = [
            doc for doc in texts
            if any(marker in doc['content'].lower() 
                  for marker in ['implementation:', 'code:', 'algorithm:', 'pseudo-code:'])
        ]
        
        if not code_sections:
            return "No code examples found."
        
        return "\n\n".join([
            f"```python\n{doc['content']}\n```"
            for doc in code_sections
        ])

    def _format_complexity(self, texts):
        """Extract and format complexity information"""
        complexity_sections = [
            doc for doc in texts
            if any(term in doc['content'].lower() 
                  for term in ['complexity', 'computational cost', 'time complexity', 'space complexity'])
        ]
        return self._format_section("Complexity Analysis", complexity_sections)

    def create_chain(self):
        """Create the RAG chain"""
        # Create the base chain
        prompt_template = """Use the following retrieved information to answer the question.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer the question based on the provided context. If referring to images or tables, 
        be specific about which ones you're referencing. If you don't find relevant information, 
        say so."""
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # Create the base chain
        self.chain = (
            {"context": lambda x: self.retriever.retrieve(x["question"]), 
             "question": RunnablePassthrough()}
            | prompt
            | ChatOpenAI(model=settings.MODEL_NAME, max_tokens=1000)
            | StrOutputParser()
        )
        
        # Create chain with sources
        self.chain_with_sources = (
            {"question": RunnablePassthrough()}
            | {
                "context": lambda x: self.retriever.retrieve(x["question"]),
                "response": lambda x: (
                    prompt.invoke({"context": x["context"], "question": x["question"]})
                    | ChatOpenAI(model=settings.MODEL_NAME, max_tokens=1000)
                    | StrOutputParser()
                ),
                "sources": lambda x: x["context"]
            }
        )
    
    def query(self, question: str, return_sources: bool = False):
        """Query the RAG system"""
        try:
            # Validate input
            input_data = QueryInput(
                question=question,
                return_sources=return_sources
            )
            
            # Update metrics
            start_time = time.time()
            self.metrics.query_count += 1
            
            try:
                if return_sources:
                    result = self.chain_with_sources.invoke(input_data.question)
                else:
                    result = self.chain.invoke(input_data.question)
                    
                # Update response time metric
                query_time = time.time() - start_time
                self.metrics.avg_response_time = (
                    (self.metrics.avg_response_time * (self.metrics.query_count - 1) + query_time)
                    / self.metrics.query_count
                )
                
                return result
                
            except Exception as e:
                self.metrics.error_count += 1
                logger.error(f"Query processing error: {str(e)}")
                raise
                
        except ValueError as e:
            logger.error(f"Invalid input: {str(e)}")
            raise

    def evaluate_response(self, response: str, context: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Evaluate response quality with metrics"""
        return {
            "completeness": self._check_completeness(response, context),
            "technical_depth": self._check_technical_depth(response),
            "visual_reference": self._check_visual_references(response, context),
            "mathematical_clarity": self._check_mathematical_clarity(response),
            "source_citation": self._check_source_citations(response, context)
        }

    def _check_completeness(self, response: str, context: Dict[str, List[Any]]) -> float:
        """Check if response covers all relevant points"""
        key_points = set()
        for doc in context["texts"]:
            terms = set(doc.page_content.lower().split())
            key_points.update(terms)
        
        if not key_points:
            return 0.0
        
        response_terms = set(response.lower().split())
        coverage = len(key_points.intersection(response_terms)) / len(key_points)
        return min(coverage * 100, 100)

    def _check_technical_depth(self, response):
        """Check technical depth of response"""
        technical_terms = {
            'algorithm', 'computation', 'implementation', 'procedure',
            'complexity', 'efficiency', 'optimization', 'calculation'
        }
        response_terms = set(response.lower().split())
        return len(technical_terms.intersection(response_terms))

    def _check_visual_references(self, response, context):
        """Check if response references visual elements"""
        has_images = bool(context["images"])
        mentions_figures = 'figure' in response.lower() or 'diagram' in response.lower()
        return has_images and mentions_figures

    def _check_mathematical_clarity(self, response):
        """Check mathematical clarity"""
        math_symbols = {'=', '∑', '∏', '∫', '√', '≈', '≠', '≤', '≥'}
        return any(symbol in response for symbol in math_symbols)

    def _check_source_citations(self, response, context):
        """Check if response cites sources"""
        has_citations = any(
            f"page {doc['metadata']['page']}" in response.lower()
            for doc in context["texts"]
        )
        return has_citations

    def display_metrics(self, response, context):
        metrics = self.evaluate_response(response, context)
        print("\nResponse Quality Metrics:")
        print(f"✓ Completeness: {metrics['completeness']:.1f}%")
        print(f"✓ Technical Depth: {metrics['technical_depth']} terms")
        print(f"✓ Visual References: {'Yes' if metrics['visual_reference'] else 'No'}")
        print(f"✓ Mathematical Clarity: {'Yes' if metrics['mathematical_clarity'] else 'No'}")
        print(f"✓ Source Citations: {'Yes' if metrics['source_citation'] else 'No'}")

    def handle_follow_up(self, previous_context):
        """Handle follow-up questions with context"""
        while True:
            follow_up = input("\nAny follow-up questions? (or 'continue'): ")
            if follow_up.lower() == 'continue':
                break
            
            # Merge previous and new context
            new_context = self.retriever.retrieve(follow_up)
            merged_context = self._merge_contexts(previous_context, new_context)
            
            response = self.generate_response(follow_up, merged_context)
            print("\nFollow-up Response:")
            print(response)

    def format_citations(self, sources):
        """Format sources in academic citation style"""
        citations = []
        for src in sources:
            if src.metadata.get('type') == 'text':
                citations.append(f"[{len(citations)+1}] Page {src.metadata['page']}: {src.metadata['section_type']}")
        return "\n\nReferences:\n" + "\n".join(citations)

    def health_check(self) -> dict:
        """System health check"""
        status = {
            "vectorstores": {},
            "models": {},
            "storage": {}
        }
        
        # Check vector stores
        for store_name in ["text", "table", "image"]:
            try:
                store = getattr(self.retriever, f"{store_name}_vectorstore")
                count = len(store.get())
                status["vectorstores"][store_name] = {
                    "status": "healthy",
                    "document_count": count
                }
            except Exception as e:
                status["vectorstores"][store_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Check model availability
        try:
            self.llm.invoke("test")
            status["models"]["llm"] = "healthy"
        except Exception as e:
            status["models"]["llm"] = f"error: {str(e)}"
            
        return status

    def shutdown(self, *args):
        """Graceful shutdown"""
        logger.info("Shutting down RAG system...")
        try:
            # Cleanup vector stores
            self.retriever.cleanup()
            # Close any open files
            # Save any pending data
            logger.info("Shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")

    def collect_metrics(self):
        """Return system metrics"""
        return asdict(self.metrics)

def main():
    # Initialize RAG system
    rag = MultimodalRAG()
    print("Initializing RAG system...")
    
    # Add initialization check
    try:
        rag.initialize()
        rag.create_chain()
    except Exception as e:
        print(f"Failed to initialize RAG system: {str(e)}")
        return
    
    print("\nInitialization complete! Ready for queries.")
    
    # Interactive query loop
    while True:
        try:
            query = input("\nEnter your question (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
            
            # Get response
            result = rag.query(query, return_sources=True)
            
            # Display response with enhancements
            print("\nResponse:")
            print(result["response"])
            
            # Display quality metrics
            rag.display_metrics(result["response"], result["sources"])
            
            # Show citations
            print(rag.format_citations(result["sources"]))
            
            # Handle follow-up questions
            rag.handle_follow_up(result["sources"])
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            print("Please try another question or type 'quit' to exit.")

if __name__ == "__main__":
    main()
