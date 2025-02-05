import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from data_extraction import extract_pdf_elements
from dotenv import load_dotenv
import time
from tqdm import tqdm  # For progress bar


# Load environment variables from .env file
load_dotenv()

# Set up environment variables for API keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")

# Add this to disable LangSmith tracing completely
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Define paths
file_path = 'data/attention_paper.pdf'

# Prompt for summarization
prompt_text = """
You are an assistant tasked with summarizing tables and text.
Give a concise summary of the table or text.

Respond only with the summary, no additional comment.
Do not start your message by saying "Here is a summary" or anything like that.
Just give the summary as it is.

Table or text chunk: {element}
"""
prompt = ChatPromptTemplate.from_template(prompt_text)

# Initialize the model
model = ChatGroq(
    temperature=0.5, 
    model="llama-3.1-8b-instant",
    max_retries=3  # Add retries for transient errors
)
summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

def process_with_rate_limit(items, process_fn, batch_size=5, delay=2):
    """Process items in batches with rate limiting"""
    results = []
    for i in tqdm(range(0, len(items), batch_size)):
        batch = items[i:i + batch_size]
        try:
            batch_results = process_fn(batch)
            results.extend(batch_results)
            if i + batch_size < len(items):  # Don't sleep after the last batch
                time.sleep(delay)
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            time.sleep(delay * 2)  # Wait longer on error
    return results

def summarize_texts(texts):
    """Summarize text sections with rate limiting."""
    text_contents = [text['content'] for text in texts]
    
    def process_batch(batch):
        return summarize_chain.batch(batch, {"max_concurrency": 1})
    
    return process_with_rate_limit(text_contents, process_batch)

def summarize_tables(tables):
    """Summarize table sections with rate limiting."""
    table_contents = [table['content']['text'] for table in tables]
    
    def process_batch(batch):
        return summarize_chain.batch(batch, {"max_concurrency": 1})
    
    return process_with_rate_limit(table_contents, process_batch)

def summarize_images(images):
    """Summarize images with rate limiting."""
    def process_batch(batch):
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": """Describe the image in detail. For context, 
                                     this image is part of a research paper explaining 
                                     the transformers architecture. Be specific about 
                                     graphs, such as bar plots."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image}"}
                        }
                    ]
                }
            ]
            for image in batch
        ]
        
        chain = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=500) | StrOutputParser()
        return [chain.invoke(msg) for msg in messages]
    
    return process_with_rate_limit(images, process_batch, batch_size=1)

def main():
    # Extract elements from the PDF
    texts, images, tables = extract_pdf_elements(file_path)

    # Summarize texts, tables, and images
    # print("\nProcessing Text Summaries...")
    # text_summaries = summarize_texts(texts)
    # print("\nProcessing Table Summaries...")
    # table_summaries = summarize_tables(tables)
    
    print("\nProcessing Image Summaries...")
    image_summaries = summarize_images([img['content'] for img in images])

    # Print summaries
    # print("\nText Summaries:")
    # for summary in text_summaries:
    #     print(summary)

    # print("\nTable Summaries:")
    # for summary in table_summaries:
    #     print(summary)

    print("\nImage Summaries:")
    for i, summary in enumerate(image_summaries):
        print(f"\nImage {i+1}:")
        print(summary)
        print("-" * 80)  # Separator line

if __name__ == "__main__":
    main()
