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
        # Basic chain
        self.chain = (
            {
                "context": self.retriever.retrieve | RunnableLambda(self.parse_retrieved_content),
                "question": RunnablePassthrough()
            }
            | RunnableLambda(self.build_prompt)
            | ChatOpenAI(model="gpt-4-turbo-preview", max_tokens=1000)
            | StrOutputParser()
        )
        
        # Chain with sources
        self.chain_with_sources = {
            "context": self.retriever.retrieve | RunnableLambda(self.parse_retrieved_content),
            "question": RunnablePassthrough()
        } | RunnablePassthrough().assign(
            response=(
                RunnableLambda(self.build_prompt)
                | ChatOpenAI(model="gpt-4-turbo-preview", max_tokens=1000)
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

    def _check_completeness(self, response, context):
        """Check if response covers all relevant points"""
        key_points = set()
        for doc in context["texts"]:
            # Extract key terms from content
            terms = set(doc['content'].lower().split())
            key_points.update(terms)
        
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
