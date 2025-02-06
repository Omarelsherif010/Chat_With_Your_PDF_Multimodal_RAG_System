import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from data_extraction import PDFParser
from dotenv import load_dotenv
import time
from tqdm import tqdm

# Load environment variables
load_dotenv()

def summarize_texts(texts):
    """Summarize text sections with rate limiting"""
    print("Summarizing text sections...")
    
    # Initialize model
    model = ChatGroq(
        temperature=0.5,
        model="llama-3.1-8b-instant",
        max_retries=3
    )
    
    # Create summarization chain
    prompt = ChatPromptTemplate.from_template("""
    Summarize this text section from a research paper.
    Be concise but preserve key technical details.
    
    Text: {text}
    """)
    summarize_chain = prompt | model | StrOutputParser()
    
    summaries = []
    for text in tqdm(texts, desc="Processing texts"):
        try:
            summary = summarize_chain.invoke({"text": text['content']})
            summaries.append(summary)
            time.sleep(0.5)  # Rate limiting
        except Exception as e:
            print(f"Error summarizing text: {e}")
            summaries.append("Error generating summary")
    
    return summaries

def summarize_tables(tables):
    """Summarize tables with improved context"""
    print("Summarizing tables...")
    
    model = ChatGroq(
        temperature=0.5,
        model="llama-3.1-8b-instant",
        max_retries=3
    )
    
    prompt = ChatPromptTemplate.from_template("""
    Summarize this table from a research paper.
    Include key metrics and findings.
    
    Table Content: {content}
    Caption: {caption}
    """)
    summarize_chain = prompt | model | StrOutputParser()
    
    summaries = []
    for table in tqdm(tables, desc="Processing tables"):
        try:
            summary = summarize_chain.invoke({
                "content": table.get('data', table.get('content', '')),
                "caption": table['metadata'].get('caption', 'No caption')
            })
            summaries.append(summary)
            time.sleep(0.5)
        except Exception as e:
            print(f"Error summarizing table: {e}")
            summaries.append("Error generating summary")
    
    return summaries

def summarize_images(images):
    """Summarize images using GPT-4V"""
    print("Summarizing images...")
    
    model = ChatOpenAI(model="gpt-4o-mini-2024-07-18", max_tokens=500)
    
    summaries = []
    for image in tqdm(images, desc="Processing images"):
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this figure from a research paper. Focus on technical details and relationships shown."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"file://{image['filepath']}"
                            }
                        }
                    ]
                }
            ]
            
            response = model.invoke(messages)
            summaries.append(response.content)
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error summarizing image: {e}")
            summaries.append("Error generating summary")
    
    return summaries

def main():
    # Initialize parser
    parser = PDFParser()
    pdf_file = "data/attention_paper.pdf"
    
    # Extract elements
    print(f"Processing PDF: {pdf_file}")
    result = parser.extract_elements(pdf_file)
    
    # Generate summaries
    text_summaries = summarize_texts(result['texts'])
    table_summaries = summarize_tables(result['tables'])
    image_summaries = summarize_images(result['images'])
    
    # Print results
    print("\nSummaries Generated:")
    print(f"Texts: {len(text_summaries)}")
    print(f"Tables: {len(table_summaries)}")
    print(f"Images: {len(image_summaries)}")

if __name__ == "__main__":
    main()
