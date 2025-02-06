import os
import fitz  # PyMuPDF
import pandas as pd
from PIL import Image
import io
import re
import base64
from dotenv import load_dotenv
import gc
from typing import Tuple, List, Dict, Any
import logging

# # Load environment variables from .env file
load_dotenv()

# Set up environment variables for API keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")

# Define paths
output_path = "./pdf_extracted_content/"
file_path = 'data/attention_paper.pdf'

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

def save_image_base64(image, filename):
    """Convert PIL Image to base64 and save to file"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Save raw base64 string without data URI prefix
    with open(filename, 'w') as f:
        f.write(img_str)
    return img_str

def extract_pdf_elements(file_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    # Add batch processing for large documents
    BATCH_SIZE = 5  # Process 5 pages at a time
    
    doc = fitz.open(file_path)
    total_pages = len(doc)
    
    all_texts = []
    all_images = []
    all_tables = []
    
    for start_page in range(0, total_pages, BATCH_SIZE):
        end_page = min(start_page + BATCH_SIZE, total_pages)
        # Process pages in batches
        batch_texts, batch_images, batch_tables = process_page_batch(doc, start_page, end_page)
        
        all_texts.extend(batch_texts)
        all_images.extend(batch_images)
        all_tables.extend(batch_tables)
        
        # Force garbage collection after each batch
        gc.collect()
    
    return all_texts, all_images, all_tables

def detect_tables(page):
    """Detect tables using layout analysis"""
    tables = []
    words = page.get_text("words")
    
    # Simple table detection based on word alignment
    # This is a basic implementation - you might want to enhance it
    table_candidates = []
    for word in words:
        if word[4].lower().startswith(("table")):
            table_candidates.append(word)
    
    # Extract tabular data near table captions
    for candidate in table_candidates:
        table_data = extract_tabular_data(page, candidate[0:4])
        if table_data:
            tables.append(table_data)
    
    return tables

def extract_tabular_data(page, bbox):
    """Extract tabular data from a given region"""
    # Basic table extraction - can be enhanced
    table_text = page.get_text("text", clip=bbox)
    # Convert text to structured data (simplified)
    return {'text': table_text, 'bbox': bbox}

def find_nearby_caption(texts, page_num, img_index):
    """Find image caption by looking for nearby text starting with 'Figure'"""
    nearby_texts = [t for t in texts if t['metadata']['page'] == page_num 
                   and t['metadata']['type'] == 'caption'
                   and t['content'].lower().startswith('figure')]
    
    return nearby_texts[0]['content'] if nearby_texts else None

def display_saved_image(base64_file):
    """Display an image from a saved base64 file"""
    with open(base64_file, 'r') as f:
        base64_str = f.read()
    
    # Decode base64 string back to image
    img_bytes = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_bytes))
    
    # Display image
    img.show()  # This will open the image in your default image viewer

def extract_equations(page):
    """Extract mathematical equations from the page"""
    equations = []
    blocks = page.get_text("dict")["blocks"]
    
    for block in blocks:
        if block["type"] == 0:  # Text block
            text = " ".join([span["text"] for line in block.get("lines", []) 
                           for span in line.get("spans", [])])
            
            # Look for equation patterns
            if any(marker in text for marker in ["=", "∑", "∏", "∫", "√"]):
                equations.append({
                    'content': text,
                    'metadata': {
                        'type': 'equation',
                        'position': block.get("bbox", [0,0,0,0])
                    }
                })
    
    return equations

def process_page_batch(doc: fitz.Document, 
                      start_page: int, 
                      end_page: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Process PDF pages in batches with type hints"""
    texts = []
    images = []
    tables = []
    processed_images = set()
    
    for page_num in range(start_page, end_page):
        page = doc[page_num]
        # Add actual page processing:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] == 0:  # Text block
                text_content = extract_text_from_block(block)
                if text_content:
                    texts.append(text_content)
                    
        # Process images
        image_list = page.get_images(full=True)
        for img in image_list:
            if img[0] not in processed_images:
                image_data = process_image(doc, img, page_num)
                if image_data:
                    images.append(image_data)
                    processed_images.add(img[0])
                    
        # Process tables
        tables.extend(detect_tables(page))
        
    return texts, images, tables

def extract_text_from_block(block):
    """Extract text from a PDF block"""
    try:
        text = " ".join([span["text"] for line in block.get("lines", []) 
                        for span in line.get("spans", [])])
        return {
            'content': text,
            'metadata': {
                'type': 'text',
                'page': block.get("page", 0),
                'position': block.get("bbox", [0,0,0,0])
            }
        }
    except Exception as e:
        logger.error(f"Error extracting text from block: {str(e)}")
        return None

def main():
    texts, images, tables = extract_pdf_elements(file_path)
    
    print(f"Extracted {len(texts)} text sections, {len(images)} images, and {len(tables)} tables")
    
    # Print detailed information
    print("\nText sections by type:")
    text_types = {}
    for text in texts:
        text_type = text['metadata']['type']
        text_types[text_type] = text_types.get(text_type, 0) + 1
    for text_type, count in text_types.items():
        print(f"- {text_type}: {count}")
    
    print("\nSample text (title):")
    title = next((t for t in texts if t['metadata']['type'] == 'title'), None)
    if title:
        print(title['content'][:100] + "...")
    
    print("\nImages saved:")
    for img in images:
        print(f"- {img['metadata']['filename']}: {img['metadata']['caption'][:100] if img['metadata']['caption'] else 'No caption'}")
    
    print("\nDisplaying saved images:")
    for img in images:
        image_path = os.path.join(output_path, img['metadata']['filename'])
        print(f"Opening: {img['metadata']['filename']}")
        display_saved_image(image_path)
    
    return texts, images, tables

if __name__ == "__main__":
    texts, images, tables = main()
