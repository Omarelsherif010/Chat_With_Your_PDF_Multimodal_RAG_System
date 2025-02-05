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

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "false"

class MultimodalRAG:
    def __init__(self):
        self.retriever = MultimodalRetriever()
        
    def initialize(self):
        """Initialize the retriever and vector stores"""
        self.retriever.initialize_vectorstores()
    
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
                with open(doc.metadata['filepath'], 'rb') as img_file:
                    img_content = b64encode(img_file.read()).decode()
                    images.append({
                        "content": img_content,
                        "metadata": doc.metadata
                    })
        
        return {
            "texts": texts,
            "tables": tables,
            "images": images
        }
    
    def build_prompt(self, kwargs):
        """Build prompt with context and question"""
        context = kwargs["context"]
        question = kwargs["question"]
        
        # Build context sections
        text_context = "\n".join([
            f"Text {i+1}:\n{doc['content']}\nSummary: {doc['metadata']['summary']}"
            for i, doc in enumerate(context["texts"])
        ])
        
        table_context = "\n".join([
            f"Table {i+1}:\n{doc['content']}\nSummary: {doc['metadata']['summary']}"
            for i, doc in enumerate(context["tables"])
        ])
        
        # Construct base prompt
        prompt_content = [
            {
                "type": "text",
                "text": f"""Answer the question based on the following context from a research paper.
                
                Text Sections:
                {text_context}
                
                Tables:
                {table_context}
                
                Question: {question}
                
                Provide a detailed answer using the provided context. Reference specific sections when relevant.
                """
            }
        ]
        
        # Add images to prompt if present
        for i, img in enumerate(context["images"]):
            prompt_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img['content']}"
                }
            })
            prompt_content.append({
                "type": "text",
                "text": f"Image {i+1} Caption: {img['metadata']['caption']}\nSummary: {img['metadata']['summary']}"
            })
        
        return ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a helpful AI assistant analyzing a research paper."),
            HumanMessage(content=prompt_content)
        ])
    
    def create_chain(self):
        """Create the RAG chain"""
        # Basic chain
        self.chain = (
            {
                "context": self.retriever.retrieve | RunnableLambda(self.parse_retrieved_content),
                "question": RunnablePassthrough()
            }
            | RunnableLambda(self.build_prompt)
            | ChatOpenAI(model="gpt-4o-mini", max_tokens=1000)
            | StrOutputParser()
        )
        
        # Chain with sources
        self.chain_with_sources = {
            "context": self.retriever.retrieve | RunnableLambda(self.parse_retrieved_content),
            "question": RunnablePassthrough()
        } | RunnablePassthrough().assign(
            response=(
                RunnableLambda(self.build_prompt)
                | ChatOpenAI(model="gpt-4o-mini", max_tokens=1000)
                | StrOutputParser()
            )
        )
    
    def query(self, question: str, return_sources: bool = False):
        """Query the RAG system"""
        if return_sources:
            result = self.chain_with_sources.invoke(question)
            return {
                "response": result["response"],
                "sources": result["context"]
            }
        else:
            return self.chain.invoke(question)

def main():
    # Initialize RAG system
    rag = MultimodalRAG()
    print("Initializing RAG system...")
    rag.initialize()
    rag.create_chain()
    
    # Interactive query loop
    while True:
        query = input("\nEnter your question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        
        show_sources = input("Show sources? (y/n): ").lower() == 'y'
        
        print("\nGenerating response...")
        result = rag.query(query, return_sources=show_sources)
        
        if show_sources:
            print("\nResponse:")
            print(result["response"])
            print("\nSources used:")
            print(json.dumps(result["sources"], indent=2))
        else:
            print("\nResponse:")
            print(result)
        
        print("-" * 80)

if __name__ == "__main__":
    main()
