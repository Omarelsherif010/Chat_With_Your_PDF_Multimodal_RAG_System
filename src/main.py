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
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


class MultimodalRAG:
    def __init__(self):
        self.retriever = MultimodalRetriever()
        
    def initialize(self):
        """Initialize the retriever and vector stores"""
        try:
            # Check if API keys are available
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OpenAI API key not found. Please provide it in the sidebar.")
            if not os.getenv("GROQ_API_KEY"):
                raise ValueError("Groq API key not found. Please provide it in the sidebar.")
            
            print("Initializing RAG system...")
            success = self.retriever.initialize_vectorstores()
            if not success:
                raise Exception("Failed to initialize vector stores")
            
            print("Creating chain...")
            self.create_chain()
            print("Initialization complete!")
            return True
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise e
    
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
    
    def build_prompt(self, kwargs):
        """Build prompt with context and question"""
        context = kwargs["context"]
        question = kwargs["question"]
        
        # Enhanced prompt structure
        prompt_content = [
            SystemMessage(content="""You are an expert AI assistant analyzing research papers. When answering:
            1. Start with a clear, concise explanation
            2. Include relevant mathematical formulas when available
            3. Reference specific sections from the paper
            4. Explain technical implementation details
            5. Connect concepts to the visual diagrams when relevant
            6. Use markdown formatting for better readability"""),
            HumanMessage(content=[{
                "type": "text",
                "text": f"""Answer the question based on the following context from the research paper.
                
                Text Sections:
                {self._format_text_sections(context["texts"])}
                
                Technical Details:
                {self._format_technical_details(context["texts"])}
                
                Mathematical Formulas:
                {self._format_equations(context["texts"])}
                
                Tables:
                {self._format_tables(context["tables"])}
                
                Question: {question}"""
            }])
        ]
        
        # Add images with better context
        if context["images"]:
            for i, img in enumerate(context["images"]):
                prompt_content[-1].content.extend([
                    {
                        "type": "image_url",
                        "image_url": {"url": img["content"]}
                    },
                    {
                        "type": "text",
                        "text": f"""
                        Figure {i+1}:
                        Caption: {img['metadata']['caption']}
                        Summary: {img['metadata']['summary']}
                        Key Elements to Notice:
                        - Mathematical relationships shown
                        - Architecture components
                        - Data flow and transformations
                        """
                    }
                ])
        
        return prompt_content
    
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
        """Format technical details from text sections"""
        technical_sections = [
            doc for doc in texts
            if any(term in doc['content'].lower() 
                  for term in ["algorithm", "implementation", "method", "technique"])
        ]
        return "\n\n".join([
            f"Technical Detail (Page {doc['metadata']['page']}):\n{doc['content']}"
            for doc in technical_sections
        ]) or "No technical details found."
    
    def _format_equations(self, texts):
        """Format equations from text sections"""
        equation_sections = [
            doc for doc in texts
            if any(symbol in doc['content'] for symbol in ["=", "∑", "∫", "×", "÷"])
        ]
        return "\n\n".join([
            f"Equation (Page {doc['metadata']['page']}):\n{doc['content']}"
            for doc in equation_sections
        ]) or "No equations found."
    
    def _format_tables(self, tables):
        """Format tables with better structure"""
        return "\n\n".join([
            f"Table (Page {doc['metadata']['page']}):\n"
            f"Content: {doc['content']}\n"
            f"Summary: {doc['metadata']['summary']}"
            for doc in tables
        ]) if tables else "No tables found."
    
    def create_chain(self):
        """Create the RAG chain"""
        # Basic chain
        self.chain = (
            {
                "context": self.retriever.retrieve | RunnableLambda(self.parse_retrieved_content),
                "question": RunnablePassthrough()
            }
            | RunnableLambda(self.build_prompt)
            | ChatOpenAI(model="gpt-4o-mini-2024-07-18", max_tokens=1000)
            | StrOutputParser()
        )
        
        # Chain with sources
        self.chain_with_sources = {
            "context": self.retriever.retrieve | RunnableLambda(self.parse_retrieved_content),
            "question": RunnablePassthrough()
        } | RunnablePassthrough().assign(
            response=(
                RunnableLambda(self.build_prompt)
                | ChatOpenAI(model="gpt-4o-mini-2024-07-18", max_tokens=1000)
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

    def evaluate_response(self, response, context):
        """Evaluate response quality"""
        metrics = {
            "completeness": self._check_completeness(response, context),
            "technical_depth": self._check_technical_depth(response),
            "visual_reference": self._check_visual_references(response, context),
            "mathematical_clarity": self._check_mathematical_clarity(response),
            "source_citation": self._check_source_citations(response, context)
        }
        return metrics

def main():
    # Initialize RAG system
    rag = MultimodalRAG()
    print("Initializing RAG system...")
    rag.initialize()
    
    # Interactive query loop
    print("\nSystem ready for queries!")
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
