import os
import logging
from pathlib import Path
from typing import Dict, Optional, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
import json
import torch
import numpy as np
import re

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

class CachedRetriever:
    """A retriever that uses existing embeddings without re-embedding content"""
    
    def __init__(self):
        """Initialize the cached retriever"""
        load_dotenv()
        
        # Initialize Pinecone
        logger.info("Connecting to Pinecone...")
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # Connect to existing indices
        self.text_index = pc.Index("text-store")
        self.image_index = pc.Index("image-store")
        
        # Initialize embeddings for queries
        self.text_embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        
        # Initialize CLIP for image queries
        self.clip_processor = None
        self.clip_model = None
        
        # Initialize vector store for text retrieval
        self.text_vectorstore = PineconeVectorStore(
            index=self.text_index,
            embedding=self.text_embeddings,
            text_key="text"
        )
        
        # Initialize LLM for response generation
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.7
        )
        
        # Load metadata and summaries
        self.load_metadata()
    
    def load_metadata(self):
        """Load metadata and summaries from files"""
        try:
            # Load summaries
            summaries_path = Path("llama_parse_summary/summaries_5.json")
            if summaries_path.exists():
                with open(summaries_path, 'r', encoding='utf-8') as f:
                    self.summaries = json.load(f)
                logger.info("Loaded summaries successfully")
            else:
                logger.warning("Summaries file not found")
                self.summaries = {}
                
            # Load original content for reference
            content_path = Path("llama_parse_output/llama_parse_output_4.json")
            if content_path.exists():
                with open(content_path, 'r', encoding='utf-8') as f:
                    self.content = json.load(f)
                logger.info("Loaded original content successfully")
            else:
                logger.warning("Original content file not found")
                self.content = {}
                
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            raise

    def is_image_related_query(self, query: str) -> bool:
        """Determine if query is specifically asking about images"""
        image_keywords = [
            'show me', 'display', 'draw', 'illustrate', 'visualize',
            'what does * look like', 'how is * drawn',
            'figure', 'diagram', 'picture', 'image',
            'architecture diagram', 'model architecture',
            'visual representation', 'graphical'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in image_keywords)

    def analyze_query_needs(self, query: str, retrieved_text: List = None) -> Dict[str, bool]:
        """Analyze query and retrieved text to determine which types of content are needed"""
        query_lower = query.lower()
        
        # Check query for table needs
        needs_table = any(word in query_lower for word in [
            'score', 'result', 'metric', 'table', 'performance',
            'number', 'accuracy', 'bleu', 'percentage', 'statistics'
        ])
        
        # Check for image needs in query
        needs_image = self.is_image_related_query(query)
        
        # Check for figure references in retrieved text
        if retrieved_text:
            for doc in retrieved_text:
                text_lower = doc.page_content.lower()
                if any(ref in text_lower for ref in ['figure', 'fig.', 'fig ']):
                    needs_image = True
                    break
        
        return {
            "needs_text": True,
            "needs_table": needs_table,
            "needs_image": needs_image
        }

    def extract_figure_numbers(self, text: str) -> List[int]:
        """Extract figure numbers from text"""
        figure_patterns = [
            r'figure\s*(\d+)',
            r'fig\.\s*(\d+)',
            r'fig\s*(\d+)'
        ]
        
        figure_numbers = []
        text_lower = text.lower()
        
        for pattern in figure_patterns:
            matches = re.finditer(pattern, text_lower)
            figure_numbers.extend(int(match.group(1)) for match in matches)
        
        return sorted(list(set(figure_numbers)))  # Remove duplicates and sort

    def get_clip_embedding(self, query: str) -> List[float]:
        """Get CLIP embedding for text query"""
        try:
            if self.clip_processor is None:
                from transformers import CLIPProcessor, CLIPModel
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            
            # Process text with CLIP
            inputs = self.clip_processor(text=query, return_tensors="pt", padding=True)
            
            # Get text features
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
            
            # Convert to numpy and normalize
            embedding = text_features.numpy()[0]
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error getting CLIP embedding: {e}")
            return [0.0] * 512  # Return zero vector of correct dimension

    def retrieve(self, query: str, k: int = 4) -> Dict:
        """Retrieve relevant content from existing embeddings"""
        try:
            # First get text results
            text_results = self.text_vectorstore.similarity_search(
                query,
                k=k,
                filter={"type": "text"}
            )
            
            # Analyze needs including retrieved text
            content_needs = self.analyze_query_needs(query, text_results)
            retrieved_content = {"texts": [], "tables": [], "images": []}
            
            # Store text results
            k_text = k if not (content_needs["needs_table"] or content_needs["needs_image"]) else 2
            retrieved_content["texts"] = text_results[:k_text]
            logger.info(f"Retrieved {len(retrieved_content['texts'])} text documents")
            
            # Get table results if needed
            if content_needs["needs_table"]:
                table_results = self.text_vectorstore.similarity_search(
                    query,
                    k=2,
                    filter={"type": "table"}
                )
                retrieved_content["tables"] = table_results
                logger.info(f"Retrieved {len(table_results)} table documents")
            
            # Get image results if needed
            if content_needs["needs_image"]:
                try:
                    # Get CLIP embedding for image search
                    query_embedding = self.get_clip_embedding(query)
                    
                    # Search image index
                    image_results = self.image_index.query(
                        vector=query_embedding,
                        top_k=2,
                        include_metadata=True
                    )
                    
                    # Convert to documents
                    image_docs = []
                    for match in image_results.matches:
                        image_docs.append({
                            "metadata": match.metadata,
                            "score": match.score
                        })
                    
                    # Check for specific figure references in text
                    figure_numbers = []
                    for doc in text_results:
                        figure_numbers.extend(self.extract_figure_numbers(doc.page_content))
                    
                    if figure_numbers:
                        logger.info(f"Found references to figures: {figure_numbers}")
                        # Add specifically referenced images
                        for img in image_docs:
                            img_num = img["metadata"].get("image_num")
                            if img_num and img_num in figure_numbers:
                                retrieved_content["images"].append(img)
                        
                        # If no specific figures found, add all retrieved images
                        if not retrieved_content["images"]:
                            retrieved_content["images"] = image_docs
                    else:
                        # No specific figures referenced, add all retrieved images
                        retrieved_content["images"] = image_docs
                    
                    logger.info(f"Retrieved {len(retrieved_content['images'])} image documents")
                    
                except Exception as e:
                    logger.error(f"Error retrieving images: {e}")
            
            return retrieved_content
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return {"texts": [], "tables": [], "images": []}

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
            3. If the question asks about specific metrics or numbers, cite them from the tables.
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
                f"Image (Page {img['metadata'].get('page_num', 'N/A')}):\n"
                f"Path: {img['metadata'].get('image_path', 'N/A')}\n"
                f"Content: {img['metadata'].get('content', 'No content available')}"
                for img in retrieved_content["images"]
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
            return "I apologize, but I encountered an error generating the response."

def main():
    """Interactive demo of the cached retriever"""
    try:
        retriever = CachedRetriever()
        print("\nCached Retriever initialized successfully!")
        
        while True:
            query = input("\nEnter your question (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
            
            print("\nRetrieving content...")
            retrieved_content = retriever.retrieve(query)
            
            print("\nGenerating response...")
            response = retriever.generate_response(query, retrieved_content)
            
            print("\nResponse:")
            print("-" * 80)
            print(response)
            print("-" * 80)
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print("\nAn error occurred. Please check the logs.")

if __name__ == "__main__":
    main() 