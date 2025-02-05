import os
import fitz  # PyMuPDF
import pandas as pd
from PIL import Image
import io
import re
import base64
from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

# # Set up environment variables for API keys
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
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
    
    with open(filename, 'w') as f:
        f.write(img_str)
    return img_str

def extract_pdf_elements(file_path):
    # Initialize PDF document
    doc = fitz.open(file_path)
    
    texts = []
    images = []
    tables = []
    processed_images = set()  # Track processed images
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Extract text with more structure
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            # Check if it's a text block and has the expected structure
            if block["type"] == 0 and "lines" in block:  # Text block
                text_spans = []
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if "text" in span:
                            text_spans.append(span["text"])
                
                text_content = " ".join(text_spans).strip()
                if text_content:
                    # Get font properties from first span if available
                    first_span = block["lines"][0]["spans"][0] if block["lines"] and block["lines"][0]["spans"] else None
                    font_size = first_span["size"] if first_span and "size" in first_span else 0
                    is_bold = first_span["flags"] & 2 ** 4 if first_span and "flags" in first_span else False
                    y_position = block["bbox"][1] if "bbox" in block else 0
                    
                    text_type = "paragraph"
                    if font_size > 12 and is_bold:
                        text_type = "heading"
                    elif y_position < 100 and page_num == 0:
                        text_type = "title"
                    elif text_content.lower().startswith(("table", "figure")):
                        text_type = "caption"
                    
                    texts.append({
                        'content': text_content,
                        'metadata': {
                            'page': page_num + 1,
                            'type': text_type,
                            'font_size': font_size,
                            'position': block.get("bbox", [0,0,0,0])
                        }
                    })
        
        # Improved table detection
        tables_on_page = detect_tables(page)
        for table in tables_on_page:
            tables.append({
                'content': table,
                'metadata': {
                    'page': page_num + 1,
                    'position': table['bbox'] if 'bbox' in table else None
                }
            })
        
        # Extract images with captions and save them
        # Get both normal images and mask images
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            
            # Skip if we've already processed this image
            if xref in processed_images:
                continue
                
            try:
                base_image = doc.extract_image(xref)
                if base_image:
                    image_bytes = base_image["image"]
                    
                    # Find nearby caption
                    caption = find_nearby_caption(texts, page_num + 1, img_index)
                    
                    # Convert to PIL Image
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Skip very small images (likely icons or decorations)
                    if image.size[0] < 50 or image.size[1] < 50:
                        continue
                    
                    # Save image as base64
                    image_filename = f"image_page{page_num + 1}_idx{img_index}.b64"
                    image_path = os.path.join(output_path, image_filename)
                    base64_str = save_image_base64(image, image_path)
                    
                    images.append({
                        'content': base64_str,
                        'metadata': {
                            'page': page_num + 1,
                            'index': img_index,
                            'size': image.size,
                            'caption': caption,
                            'filename': image_filename
                        }
                    })
                    
                    processed_images.add(xref)
                    
            except Exception as e:
                print(f"Warning: Could not extract image on page {page_num + 1}, index {img_index}: {str(e)}")
                continue
    
    # Remove duplicate images based on content
    unique_images = []
    seen_contents = set()
    
    for img in images:
        if img['content'] not in seen_contents:
            seen_contents.add(img['content'])
            unique_images.append(img)
    
    return texts, unique_images, tables

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
