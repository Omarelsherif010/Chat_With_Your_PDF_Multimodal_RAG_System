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
from data_extraction import extract_pdf_elements, display_saved_image
from data_summarize import summarize_texts, summarize_tables, summarize_images
from dotenv import load_dotenv
import json
from functools import lru_cache

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
        
        # Add metadata filters for better retrieval
        self.retrieval_filters = {
            "text": lambda x: x["type"] == "text",
            "equation": lambda x: "equation" in x["section_type"].lower(),
            "technical": lambda x: any(term in x["content"].lower() 
                                    for term in ["algorithm", "formula", "equation", "implementation"])
        }

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
                "filepath": os.path.join(output_path, image["metadata"]["filename"]),
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
    
    def initialize_vectorstores(self):
        """Initialize vector stores with content and summaries"""
        try:
            # Extract content
            print("Extracting PDF elements...")
            texts, images, tables = extract_pdf_elements(file_path)
            
            # Limit to first 20 texts for development
            texts = texts[:20]
            print(f"Using first {len(texts)} text sections for development...")
            
            # Generate summaries
            print("Generating summaries...")
            text_summaries = summarize_texts(texts)
            table_summaries = summarize_tables(tables)
            image_summaries = summarize_images([img['content'] for img in images])
            
            # Prepare documents
            print("Preparing documents...")
            text_docs, text_ids, text_originals = self.prepare_text_documents(texts, text_summaries)
            table_docs, table_ids, table_originals = self.prepare_table_documents(tables, table_summaries)
            image_docs, image_ids, image_originals = self.prepare_image_documents(images, image_summaries)
            
            # Clear existing vectors if any
            print("Clearing existing vectors...")
            try:
                # Try to delete vectors from each namespace separately
                for namespace in ["text", "table", "image"]:
                    try:
                        self.index.delete(delete_all=True, namespace=namespace)
                        print(f"Cleared vectors from {namespace} namespace")
                    except Exception as e:
                        print(f"No existing vectors in {namespace} namespace")
            except Exception as e:
                print("No existing vectors to clear")
            
            # Add documents to vector store with batching and namespaces
            batch_size = 100
            
            print("Adding documents to text namespace...")
            for i in range(0, len(text_docs), batch_size):
                batch = text_docs[i:i + batch_size]
                self.vectorstore.add_documents(batch, namespace="text")
            
            print("Adding documents to table namespace...")
            for i in range(0, len(table_docs), batch_size):
                batch = table_docs[i:i + batch_size]
                self.vectorstore.add_documents(batch, namespace="table")
            
            print("Adding documents to image namespace...")
            for i in range(0, len(image_docs), batch_size):
                batch = image_docs[i:i + batch_size]
                self.vectorstore.add_documents(batch, namespace="image")
            
            # Store original documents
            print("Storing original documents...")
            self.text_store.mset(list(zip(text_ids, text_originals)))
            self.table_store.mset(list(zip(table_ids, table_originals)))
            self.image_store.mset(list(zip(image_ids, image_originals)))
            
            print("Vector stores initialized successfully!")
            return True
            
        except Exception as e:
            print(f"Error in initialize_vectorstores: {str(e)}")
            raise e
    
    @lru_cache(maxsize=100)
    def retrieve(self, query, k=2):
        """Retrieve relevant content with caching"""
        try:
            results = {
                "texts": self.vectorstore.similarity_search(
                    query, k=k, namespace="text"
                ),
                "tables": self.vectorstore.similarity_search(
                    query, k=k, namespace="table"
                ),
                "images": self.vectorstore.similarity_search(
                    query, k=k, namespace="image"
                )
            }
            
            # Display retrieved images
            print("\nRetrieved Images:")
            for img_doc in results["images"]:
                print(f"\nImage Summary: {img_doc.metadata['summary']}")
                print(f"Caption: {img_doc.metadata['caption']}")
                if os.path.exists(img_doc.metadata['filepath']):
                    display_saved_image(img_doc.metadata['filepath'])
            
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
