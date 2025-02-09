import os
import uuid
import logging
from typing import Dict, List, Tuple, Optional, Any
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import json
from functools import lru_cache, wraps
from pathlib import Path
from PIL import Image
import io
import base64
import torch
from transformers import CLIPProcessor, CLIPModel
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
import numpy as np
from time import sleep
import time
from tqdm import tqdm
from dataclasses import dataclass, field
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('retrieval.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DocumentMetadata(BaseModel):
    """Validation model for document metadata"""
    doc_id: str = Field(..., description="Unique identifier for the document")
    type: str = Field(..., description="Type of content (text/image)")
    page_num: int = Field(..., ge=1, description="Page number")
    section_title: Optional[str] = Field(None, description="Section title or heading")
    references: Optional[List[str]] = Field(default_factory=list, description="Referenced sections or citations")
    
    @field_validator('type')
    def validate_type(cls, v):
        if v not in ['text', 'image']:
            raise ValueError('Type must be either "text" or "image"')
        return v

class ImageMetadata(DocumentMetadata):
    """Additional validation for image metadata"""
    image_num: int = Field(..., ge=1, description="Image number")
    mime_type: str = Field(..., description="MIME type of the image")
    caption: Optional[str] = Field(None, description="Image caption")
    image_data_path: str = Field(..., description="Path to stored image data")

class RetrieverConfig(BaseSettings):
    """Configuration for MultimodalRetriever"""
    batch_size: int = 100
    clip_dimension: int = 512
    text_chunk_size: int = 500
    text_chunk_overlap: int = 100
    temperature: float = 0.7
    llm_model: str = "gpt-4o-mini-2024-07-18"
    clip_model: str = "openai/clip-vit-base-patch32"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    text_index_name: str = "text-store"
    image_index_name: str = "image-store"
    log_file: str = "retrieval.log"
    image_store_dir: str = "image_store"

    class Config:
        env_prefix = "RETRIEVER_"

def rate_limit(calls: int = 10, period: float = 1.0):
    """Rate limiting decorator"""
    def decorator(func):
        last_reset = 0.0
        calls_made = 0
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal last_reset, calls_made
            now = time.time()
            
            if now - last_reset > period:
                calls_made = 0
                last_reset = now
            
            if calls_made >= calls:
                sleep_time = period - (now - last_reset)
                if sleep_time > 0:
                    sleep(sleep_time)
                calls_made = 0
                last_reset = time.time()
            
            calls_made += 1
            return func(*args, **kwargs)
        return wrapper
    return decorator

@dataclass
class RetrievalMetrics:
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    query_times: List[float] = field(default_factory=list)
    errors: Dict[str, int] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)

class MultimodalRetriever:
    def __init__(self):
        """Initialize the multimodal retriever with both text and image models"""
        load_dotenv()
        
        # Initialize configuration
        self.config = RetrieverConfig()
        
        # Initialize CLIP for image embeddings
        logger.info("Initializing CLIP model...")
        self.clip_model = CLIPModel.from_pretrained(self.config.clip_model)
        self.clip_processor = CLIPProcessor.from_pretrained(self.config.clip_model)
        
        # Initialize Pinecone
        logger.info("Initializing Pinecone...")
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # Initialize embeddings
        self.text_embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        
        # Set up indices
        self.text_index_name = self.config.text_index_name
        self.image_index_name = self.config.image_index_name
        
        # Initialize text index (1536 dimensions)
        self.text_index = self.create_index(pc, self.text_index_name, 1536)
        
        # Initialize image index (512 dimensions)
        self.image_index = self.create_index(pc, self.image_index_name, 512)
        
        # Initialize vector stores
        self.text_vectorstore = PineconeVectorStore(
            index=self.text_index,
            embedding=self.text_embeddings,
            text_key="text",
            index_name=self.text_index_name
        )
        
        # Initialize stores
        self.text_store = InMemoryStore()
        self.image_store = InMemoryStore()
        
        # Text processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.text_chunk_size,
            chunk_overlap=self.config.text_chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # LLM for response generation
        self.llm = ChatOpenAI(
            model=self.config.llm_model,
            temperature=self.config.temperature
        )
        
        # Add image data store
        self.image_data_store = {}  # Store image data separately
        
        # Create directory for image data if it doesn't exist
        self.image_store_dir = Path(self.config.image_store_dir)
        self.image_store_dir.mkdir(exist_ok=True)
        
        # Add batch size configuration
        self.batch_size = self.config.batch_size
        
        # Add CLIP dimensions
        self.clip_dimension = self.config.clip_dimension
        
        # Initialize metrics
        self.metrics = RetrievalMetrics()

    def create_index(self, pc: Pinecone, name: str, dimension: int) -> Any:
        """Create Pinecone index with error handling"""
        try:
            index = pc.Index(name)
            logger.info(f"Found existing {name} index")
            return index
        except Exception as e:
            logger.info(f"Creating new {name} index...")
            try:
                index = pc.create_index(
                    name=name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=self.config.pinecone_cloud,
                        region=self.config.pinecone_region
                    )
                )
                # Wait for index to be ready
                while not pc.describe_index(name).status['ready']:
                    logger.info("Waiting for index to be ready...")
                    time.sleep(1)
                return pc.Index(name)
            except Exception as e:
                logger.error(f"Failed to create index {name}: {e}")
                raise

    @rate_limit(calls=10, period=1.0)
    def get_clip_embedding(self, image_base64: str) -> np.ndarray:
        """Get CLIP embedding for an image"""
        try:
            # Decode base64 image
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            # Process image with CLIP
            inputs = self.clip_processor(images=image, return_tensors="pt")
            
            # Get image features
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            
            # Convert to numpy and normalize
            embedding = image_features.numpy()[0]
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding.tolist()  # Convert to list for Pinecone
            
        except Exception as e:
            logger.error(f"Error getting CLIP embedding: {e}")
            # Return zero vector of correct dimension as fallback
            return [0.0] * self.clip_dimension

    def prepare_text_documents(self, pages: List[Dict], summaries: Optional[Dict] = None) -> Tuple[List[Document], List[str], List[Dict]]:
        """Prepare text documents with enhanced metadata and summaries"""
        documents = []
        doc_ids = []
        originals = []
        
        for page in pages:
            doc_id = str(uuid.uuid4())
            doc_ids.append(doc_id)
            originals.append(page)
            
            # Find corresponding summary
            summary = None
            if summaries:
                for page_summary in summaries['page_summaries']:
                    if page_summary['page_num'] == page['page_num']:
                        summary = page_summary['summary']
                        break
            
            # Extract non-table content
            content = page['content']
            table_sections = []
            non_table_sections = []
            current_section = []
            in_table = False
            
            for line in content.split('\n'):
                if line.startswith("# Table") or "|---" in line:
                    if current_section and not in_table:
                        non_table_sections.append('\n'.join(current_section))
                    current_section = [line]
                    in_table = True
                elif in_table:
                    if line.strip() and ("|" in line or line.startswith("#")):
                        current_section.append(line)
                    else:
                        table_sections.append('\n'.join(current_section))
                        current_section = [line] if line.strip() else []
                        in_table = False
                else:
                    current_section.append(line)
            
            # Add any remaining content
            if current_section:
                if in_table:
                    table_sections.append('\n'.join(current_section))
                else:
                    non_table_sections.append('\n'.join(current_section))
            
            # Process non-table content
            for section in non_table_sections:
                if section.strip():  # Skip empty sections
                    chunks = self.text_splitter.split_text(section)
                    for chunk in chunks:
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                "doc_id": doc_id,
                                "type": "text",
                                "page_num": page['page_num'],
                                "summary": summary,
                                "full_content": section
                            }
                        )
                        documents.append(doc)
        
        return documents, doc_ids, originals

    def prepare_image_documents(self, images: List[Dict], summaries: Optional[Dict] = None) -> Tuple[List[Document], List[str], List[np.ndarray]]:
        """Prepare image documents with their embeddings"""
        documents = []
        doc_ids = []
        embeddings = []
        
        # Create images directory if it doesn't exist
        image_dir = Path("static/images")  # Changed to static/images for web access
        image_dir.mkdir(parents=True, exist_ok=True)
        
        for image in images:
            try:
                doc_id = str(uuid.uuid4())
                doc_ids.append(doc_id)
                
                # Get base64 image data and validate
                if 'base64' not in image:
                    logger.error(f"No base64 data found for image {image.get('image_num')}")
                    continue
                    
                # Save image to local directory
                try:
                    # Add validation for base64 data
                    base64_data = image['base64']
                    if not base64_data:
                        logger.error(f"Empty base64 data for image {image.get('image_num')}")
                        continue
                        
                    # Try to decode base64 data
                    try:
                        image_data = base64.b64decode(base64_data)
                    except Exception as e:
                        logger.error(f"Invalid base64 data for image {image.get('image_num')}: {e}")
                        continue
                    
                    # Validate decoded image data
                    try:
                        # Verify it's a valid image
                        test_image = Image.open(io.BytesIO(image_data))
                        test_image.verify()  # Verify it's a valid image file
                        
                        # Save with correct extension based on image format
                        extension = test_image.format.lower() if test_image.format else 'png'
                        image_filename = f"{doc_id}.{extension}"
                        image_path = image_dir / image_filename
                        
                        # Save the actual image file
                        with open(image_path, 'wb') as f:
                            f.write(image_data)
                        
                        logger.info(f"Successfully saved image {image.get('image_num')} to {image_path}")
                        
                    except Exception as e:
                        logger.error(f"Invalid image data for image {image.get('image_num')}: {e}")
                        continue
                    
                    # Find corresponding summary
                    summary = None
                    if summaries:
                        for image_summary in summaries.get('image_summaries', []):
                            if (image_summary['page_num'] == image['page_num'] and 
                                image_summary['image_num'] == image['image_num']):
                                summary = image_summary['summary']
                                break
                    
                    # Create document with absolute path to local image
                    doc = Document(
                        page_content=f"Image {image.get('image_num', 'N/A')} on Page {image.get('page_num', 'N/A')}\n \
                        Type: image/{extension}\n \
                        Summary: {summary or 'No summary available'}",
                        metadata={
                            "doc_id": doc_id,
                            "type": "image",
                            "page_num": image.get('page_num', 0),
                            "image_num": image.get('image_num', 0),
                            "mime_type": f"image/{extension}",
                            "summary": summary,
                            "image_path": str(image_path.absolute())  # Store absolute path
                        }
                    )
                    documents.append(doc)
                    
                    # Get image embedding using the verified image
                    pil_image = Image.open(image_path)  # Use saved image for embedding
                    embedding = self.get_clip_embedding_for_image(pil_image)
                    embeddings.append(embedding)
                    
                except Exception as e:
                    logger.error(f"Error saving image {image.get('image_num')}: {e}")
                    continue
                
            except Exception as e:
                logger.error(f"Error processing image {image.get('image_num')}: {e}")
                continue
        
        return documents, doc_ids, embeddings

    def prepare_table_documents(self, pages: List[Dict], summaries: Optional[Dict] = None) -> Tuple[List[Document], List[str], List[Dict]]:
        """Prepare table documents with their summaries"""
        documents = []
        doc_ids = []
        originals = []
        
        for page in pages:
            # Find corresponding summary
            summary = None
            if summaries:
                for page_summary in summaries['page_summaries']:
                    if page_summary['page_num'] == page['page_num']:
                        summary = page_summary['summary']
                        break
            
            # Extract tables and their captions
            lines = page['content'].split('\n')
            table_content = []
            current_table = []
            title = ""  # Initialize with empty string instead of None
            caption = []
            in_table = False
            
            for line in lines:
                if line.startswith("# Table"):
                    if current_table:  # Save previous table if exists
                        table_content.append({
                            "title": title or "Untitled Table",  # Provide default if empty
                            "caption": '\n'.join(caption) or "No caption available",  # Provide default if empty
                            "content": '\n'.join(current_table)
                        })
                    title = line
                    caption = []
                    current_table = [line]
                    in_table = False
                elif "|---" in line or (in_table and "|" in line):
                    current_table.append(line)
                    in_table = True
                elif in_table and not line.strip():
                    in_table = False
                    table_content.append({
                        "title": title or "Untitled Table",  # Provide default if empty
                        "caption": '\n'.join(caption) or "No caption available",  # Provide default if empty
                        "content": '\n'.join(current_table)
                    })
                    title = ""  # Reset to empty string
                    caption = []
                    current_table = []
                elif title and not in_table:
                    caption.append(line)
            
            # Add last table if exists
            if current_table:
                table_content.append({
                    "title": title or "Untitled Table",  # Provide default if empty
                    "caption": '\n'.join(caption) or "No caption available",  # Provide default if empty
                    "content": '\n'.join(current_table)
                })
            
            # Create documents for each table
            for table in table_content:
                if table["content"].strip():
                    doc_id = str(uuid.uuid4())
                    doc_ids.append(doc_id)
                    originals.append(page)
                    
                    doc = Document(
                        page_content=table["content"],
                        metadata={
                            "doc_id": doc_id,
                            "type": "table",
                            "page_num": page['page_num'],
                            "summary": summary or "No summary available",  # Provide default if None
                            "title": table["title"],  # Now guaranteed to be a string
                            "caption": table["caption"],  # Now guaranteed to be a string
                            "full_content": page['content']
                        }
                    )
                    documents.append(doc)
        
        return documents, doc_ids, originals

    def hybrid_search(self, query: str, k: int = 2) -> List[Document]:
        """Combine vector and keyword search"""
        try:
            # Get vector search results
            vector_results = self.text_vectorstore.similarity_search(
                query, k=k, namespace="text"
            )
            
            # Get keyword search results (simple implementation)
            keyword_results = [
                doc for doc in vector_results
                if any(word.lower() in doc.page_content.lower() 
                      for word in query.split())
            ]
            
            # Combine results with deduplication
            combined_results = list({
                doc.metadata['doc_id']: doc 
                for doc in vector_results + keyword_results
            }.values())
            
            return combined_results[:k]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []

    def is_image_related_query(self, query: str) -> bool:
        """Determine if query is specifically asking about images or visual elements"""
        # Keywords that strongly indicate image-related queries
        image_keywords = [
            'show me', 'display', 'draw', 'illustrate', 'visualize',
            'what does * look like', 'how is * drawn',
            'figure', 'diagram', 'picture', 'image',
            'architecture diagram', 'model architecture',
            'visual representation', 'graphical',
            'what is shown in', 'explain the figure'
        ]
        
        query_lower = query.lower()
        
        # Check for exact phrases
        for keyword in image_keywords:
            if '*' in keyword:
                # Handle wildcard patterns
                parts = keyword.split('*')
                if parts[0] in query_lower and parts[1] in query_lower:
                    return True
            elif keyword in query_lower:
                return True
        
        # Check for specific question patterns
        if any(pattern in query_lower for pattern in [
            'how does the model',  # Only if asking about structure
            'what is the structure',
            'how is the architecture',
            'can you explain the diagram',
            'describe the figure'
        ]):
            return True
        
        return False

    def analyze_query_needs(self, query: str) -> Dict[str, bool]:
        """Analyze query to determine which types of content are needed"""
        query_lower = query.lower()
        
        # Check for table/metrics needs
        needs_table = any(word in query_lower for word in [
            'score', 'result', 'metric', 'table', 'performance', 'number',
            'accuracy', 'bleu', 'percentage', 'statistics', 'compare'
        ])
        
        # Check for image needs
        needs_image = self.is_image_related_query(query)
        
        # Always need text for context and understanding
        return {
            "needs_text": True,  # Always True as we need text for context
            "needs_table": needs_table,
            "needs_image": needs_image
        }

    def retrieve(self, query: str, k: int = 4) -> Dict[str, List[Document]]:
        """Retrieve relevant content from text, tables, and images based on query needs"""
        try:
            # Analyze query needs
            content_needs = self.analyze_query_needs(query)
            retrieved_content = {"texts": [], "tables": [], "images": []}
            
            # Always get text results, but adjust k based on other content needs
            k_text = k if not (content_needs["needs_table"] or content_needs["needs_image"]) else 2
            text_results = self.text_vectorstore.similarity_search(
                query, 
                k=k_text,
                filter={"type": "text"}
            )
            retrieved_content["texts"] = text_results
            logger.info(f"Retrieved {len(text_results)} text documents")
            
            # Get table results if needed
            if content_needs["needs_table"]:
                table_results = self.text_vectorstore.similarity_search(
                    query,
                    k=2,
                    filter={"type": "table"}
                )
                # Filter tables by relevance
                filtered_tables = []
                query_terms = set(query.lower().split())
                for doc in table_results:
                    table_text = (doc.metadata.get('title', '') + ' ' + 
                                doc.metadata.get('caption', '') + ' ' + 
                                doc.page_content).lower()
                    if any(term in table_text for term in query_terms):
                        filtered_tables.append(doc)
                
                retrieved_content["tables"] = filtered_tables
                logger.info(f"Retrieved {len(filtered_tables)} relevant table documents")
            
            # Get image results if needed
            if content_needs["needs_image"]:
                try:
                    query_embedding = self.get_clip_embedding_for_text(query)
                    image_search = self.image_index.query(
                        vector=query_embedding,
                        top_k=2,
                        include_metadata=True
                    )
                    
                    image_results = []
                    for match in image_search.matches:
                        image_results.append(
                            Document(
                                page_content=match.metadata.get("content", ""),
                                metadata={
                                    k: v for k, v in match.metadata.items() 
                                    if k != "content"
                                }
                            )
                        )
                    retrieved_content["images"] = image_results
                    logger.info(f"Retrieved {len(image_results)} image documents")
                except Exception as e:
                    logger.error(f"Error retrieving images: {e}")
            
            # Log retrieval decisions
            logger.info(f"Content needs - Text: Always included, "
                       f"Table: {content_needs['needs_table']}, "
                       f"Image: {content_needs['needs_image']}")
            
            return retrieved_content

        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return {"texts": [], "tables": [], "images": []}

    @rate_limit(calls=10, period=1.0)
    def get_clip_embedding_for_image(self, image: Image.Image) -> np.ndarray:
        """Get CLIP embedding for a PIL image"""
        try:
            # Process image with CLIP
            inputs = self.clip_processor(images=image, return_tensors="pt")
            
            # Get image features
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            
            # Convert to numpy and normalize
            embedding = image_features.numpy()[0]
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding.tolist()  # Convert to list for Pinecone
            
        except Exception as e:
            logger.error(f"Error getting CLIP embedding: {e}")
            # Return zero vector of correct dimension as fallback
            return [0.0] * self.clip_dimension

    @rate_limit(calls=10, period=1.0)
    def get_clip_embedding_for_text(self, text: str) -> np.ndarray:
        """Get CLIP embedding for text"""
        try:
            # Process text with CLIP
            inputs = self.clip_processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=77)
            
            # Get text features
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
            
            # Convert to numpy and normalize
            embedding = text_features.numpy()[0]
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error getting CLIP text embedding: {e}")
            return [0.0] * self.clip_dimension

    def get_image_data(self, image_data_path: str) -> Dict:
        """Retrieve image data from storage"""
        try:
            with open(image_data_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error retrieving image data: {e}")
            return None

    def generate_response(self, query: str, retrieved_content: Dict) -> str:
        """Generate response using retrieved content"""
        try:
            prompt_template = """You are an AI assistant helping with questions about a research paper on the Transformer architecture.
            Use the following retrieved content to answer the question accurately and precisely.
            
            Retrieved Text Sections:
            {text_sections}
            
            Retrieved Tables:
            {table_sections}
            
            Retrieved Images:
            {image_sections}
            
            Question: {query}
            
            Instructions:
            1. Use the retrieved content to provide a detailed and accurate response.
            2. If the question is about images and there are retrieved images, describe what the images show.
            3. If the question asks about specific metrics or numbers, cite them from the tables and pages if available.
            4. If no relevant information is found in the retrieved content, say so clearly.
            5. Base your response only on the retrieved content, do not make up information.
            """
            
            # Prepare sections for prompt
            text_sections = "\n\n".join([
                f"Page {doc.metadata['page_num']}:\n{doc.page_content}"
                for doc in retrieved_content["texts"]
            ])
            
            table_sections = "\n\n".join([
                f"Table (Page {doc.metadata['page_num']}):\n{doc.page_content}"
                for doc in retrieved_content["tables"]
            ])
            
            image_sections = "\n\n".join([
                f"Image {doc.metadata.get('image_num', 'N/A')} (Page {doc.metadata.get('page_num', 'N/A')}):\n"
                f"Summary: {doc.metadata.get('summary', 'No summary available')}\n"
                f"Content: {doc.page_content}"
                for doc in retrieved_content["images"]
            ])
            
            # Create prompt
            prompt = ChatPromptTemplate.from_template(prompt_template)
            
            # Create chain
            chain = prompt | self.llm | StrOutputParser()
            
            # Generate response
            response = chain.invoke({
                "text_sections": text_sections or "No relevant text sections found.",
                "table_sections": table_sections or "No relevant tables found.",
                "image_sections": image_sections or "No relevant images found.",
                "query": query
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error generating the response. Please try rephrasing your question."

    def load_and_store_content(self, json_path: str, summaries_path: str = None) -> bool:
        """Load content from JSON and summaries, then store in vector database"""
        try:
            # Load parsed content
            logger.info(f"Loading content from {json_path}")
            with open(json_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            # Load summaries if provided
            summaries = None
            if summaries_path:
                logger.info(f"Loading summaries from {summaries_path}")
                with open(summaries_path, 'r', encoding='utf-8') as f:
                    summaries = json.load(f)

            # Prepare documents
            logger.info("Preparing documents...")
            text_docs, text_ids, text_originals = self.prepare_text_documents(content['pages'], summaries)
            table_docs, table_ids, table_originals = self.prepare_table_documents(content['pages'], summaries)
            image_docs, image_ids, image_embeddings = self.prepare_image_documents(content['images'], summaries)
            
            # Clear existing vectors
            logger.info("Clearing existing vectors...")
            try:
                self.text_index.delete(delete_all=True)
                logger.info("Cleared text vectors")
            except Exception as e:
                logger.info("No existing text vectors")
            
            try:
                self.image_index.delete(delete_all=True)
                logger.info("Cleared image vectors")
            except Exception as e:
                logger.info("No existing image vectors")
            
            # Add text documents with progress bar
            logger.info("Adding text documents...")
            with tqdm(total=len(text_docs), desc="Adding text documents") as pbar:
                for i in range(0, len(text_docs), self.batch_size):
                    batch = text_docs[i:i + self.batch_size]
                    self.text_vectorstore.add_documents(batch)
                    pbar.update(len(batch))
            
            # Add table documents
            logger.info("Adding table documents...")
            with tqdm(total=len(table_docs), desc="Adding table documents") as pbar:
                for i in range(0, len(table_docs), self.batch_size):
                    batch = table_docs[i:i + self.batch_size]
                    self.text_vectorstore.add_documents(batch)
                    pbar.update(len(batch))
            
            # Add image documents
            if image_docs and image_embeddings:
                logger.info("Adding image documents...")
                for i in range(0, len(image_docs), self.batch_size):
                    batch_docs = image_docs[i:i + self.batch_size]
                    batch_embeddings = image_embeddings[i:i + self.batch_size]
                    vectors = []
                    
                    for doc, embedding in zip(batch_docs, batch_embeddings):
                        metadata = {
                            "doc_id": doc.metadata["doc_id"],
                            "type": "image",
                            "page_num": doc.metadata["page_num"],
                            "image_num": doc.metadata["image_num"],
                            "mime_type": doc.metadata["mime_type"],
                            "summary": doc.metadata.get("summary"),
                            "image_path": doc.metadata["image_path"],
                            "content": doc.page_content
                        }
                        
                        vectors.append({
                            "id": doc.metadata["doc_id"],
                            "values": embedding,
                            "metadata": metadata
                        })
                    
                    # Upsert vectors to image index
                    if vectors:
                        self.image_index.upsert(vectors=vectors)
                        logger.info(f"Added batch of {len(vectors)} image vectors")
            
            # Store original documents
            logger.info("Storing original documents...")
            self.text_store.mset(list(zip(text_ids, text_originals)))
            self.table_store = InMemoryStore()  # Add table store
            self.table_store.mset(list(zip(table_ids, table_originals)))
            
            logger.info("Content stored successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error storing content: {str(e)}")
            raise

    def cleanup(self):
        """Enhanced cleanup with memory management"""
        try:
            # Clear CLIP resources
            if hasattr(self, 'clip_model'):
                del self.clip_model
            if hasattr(self, 'clip_processor'):
                del self.clip_processor
            
            # Clear vector stores
            if hasattr(self, 'text_vectorstore'):
                del self.text_vectorstore
            
            # Clear indices
            if hasattr(self, 'text_index'):
                del self.text_index
            if hasattr(self, 'image_index'):
                del self.image_index
            
            # Clear memory stores - remove close() calls
            if hasattr(self, 'text_store'):
                del self.text_store
            if hasattr(self, 'image_store'):
                del self.image_store
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def check_health(self) -> bool:
        """Check system health"""
        try:
            # Check CLIP model
            if not hasattr(self, 'clip_model') or not hasattr(self, 'clip_processor'):
                logger.error("CLIP model not initialized")
                return False
            
            # Check indices
            try:
                self.text_index.describe_index_stats()
                self.image_index.describe_index_stats()
            except Exception as e:
                logger.error(f"Index health check failed: {e}")
                return False
            
            # Check stores
            if not self.text_store or not self.image_store:
                logger.error("Document stores not initialized")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def display_image(self, image_path: str) -> None:
        """Display an image using PIL"""
        try:
            path = Path(image_path)
            if not path.exists():
                print(f"Image file not found: {image_path}")
                return
            
            image = Image.open(path)
            image.show()
        except Exception as e:
            logger.error(f"Error displaying image: {e}")
            print(f"Could not display image from {image_path}")

def main():
    try:
        with MultimodalRetriever() as retriever:
            # Load and store content
            json_path = "llama_parse_output/llama_parse_output_4.json"
            summaries_path = "llama_parse_summary/summaries_5.json"
            retriever.load_and_store_content(json_path, summaries_path)
            
            # Interactive query loop
            while True:
                try:
                    query = input("\nEnter your question (or 'quit' to exit): ")
                    if query.lower() == 'quit':
                        break
                    
                    logger.info(f"Processing query: {query}")
                    
                    print("\nRetrieving relevant content...")
                    retrieved_content = retriever.retrieve(query)
                    
                    # Print retrieved content
                    print("\nRetrieved Text Content:")
                    print("-" * 80)
                    for doc in retrieved_content["texts"]:
                        print(f"\nPage {doc.metadata['page_num']}:")
                        print(f"Summary: {doc.metadata.get('summary', 'No summary available')}")
                        print(f"Content: {doc.page_content}")
                        print("-" * 40)
                    
                    # Print retrieved tables
                    if retrieved_content["tables"]:
                        print("\nRetrieved Tables:")
                        print("-" * 80)
                        for doc in retrieved_content["tables"]:
                            print(f"\n{doc.metadata.get('title', 'Table')}")
                            print(f"Page {doc.metadata['page_num']}:")
                            print(doc.page_content)
                            print("-" * 40)
                    
                    # Print and display retrieved images
                    if retrieved_content["images"]:
                        print("\nRetrieved Images:")
                        print("-" * 80)
                        for doc in retrieved_content["images"]:
                            print(f"\nImage {doc.metadata.get('image_num', 'N/A')} (Page {doc.metadata.get('page_num', 'N/A')}):")
                            print(f"Summary: {doc.metadata.get('summary', 'No summary available')}")
                            print(f"Content: {doc.page_content}")
                            print("-" * 40)
                            
                            # Display the image if path exists
                            image_path = doc.metadata.get('image_path')
                            if image_path and os.path.exists(image_path):
                                print(f"Displaying image from {image_path}")
                                retriever.display_image(image_path)
                            else:
                                print("Image file not found")
                    
                    print("\nGenerating response...")
                    response = retriever.generate_response(query, retrieved_content)
                    
                    print("\nResponse:")
                    print(response)
                    print("-" * 80)
                    
                except Exception as e:
                    logger.error(f"Error processing query: {e}")
                    print("\nI apologize, but I encountered an error. Please try again.")
                    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print("An error occurred initializing the system. Please check the logs.")

if __name__ == "__main__":
    main()
