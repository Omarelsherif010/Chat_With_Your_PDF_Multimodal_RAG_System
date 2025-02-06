import os
import fitz  # PyMuPDF
import pandas as pd
from PIL import Image
import io
import re
import base64
from dotenv import load_dotenv
import camelot
import numpy as np

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
        tables_on_page = extract_tables(page)
        for table in tables_on_page:
            tables.append(table)
        
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

def extract_tables(page):
    """Extract tables from a page using camelot"""
    tables = []
    doc = fitz.open(file_path)  # We need this for caption detection
    
    try:
        # Extract tables from the current page
        page_num = page.number + 1
        # Try both lattice and stream modes
        table_list = []
        try:
            # First try lattice mode for tables with visible lines
            lattice_tables = camelot.read_pdf(
                file_path,
                pages=str(page_num),
                flavor='lattice',
                line_scale=40  # Adjust line detection sensitivity
            )
            if len(lattice_tables) > 0:
                table_list.extend(lattice_tables)
        except Exception as e:
            print(f"Lattice mode failed on page {page_num}: {str(e)}")
            
        if not table_list:  # Only try stream if lattice found nothing
            try:
                # Then try stream mode for tables without visible lines
                stream_tables = camelot.read_pdf(
                    file_path,
                    pages=str(page_num),
                    flavor='stream',
                    edge_tol=500  # Increase tolerance for table edges
                )
                table_list.extend(stream_tables)
            except Exception as e:
                print(f"Stream mode failed on page {page_num}: {str(e)}")
        
        # Process each table found
        for table_idx, table in enumerate(table_list):
            try:
                # Get DataFrame
                df = table.df
                
                # Clean and format the table
                df = clean_dataframe(df)
                
                # Skip empty or tiny tables
                if df.empty or df.size < 4:
                    continue
                    
                # Skip tables with low accuracy or high whitespace
                if table.accuracy < 80 or table.whitespace > 50:
                    continue
                
                # Convert DataFrame to formatted text
                table_text = format_dataframe(df)
                
                # Get table bounds from camelot
                bbox = table._bbox
                
                # Create table data structure
                table_data = {
                    'content': df.values.tolist(),  # Store as list of lists
                    'text': table_text,
                    'bbox': bbox,
                    'metadata': {
                        'page': page_num,
                        'type': 'table',
                        'rows': len(df),
                        'columns': len(df.columns),
                        'headers': df.columns.tolist(),
                        'accuracy': table.accuracy,
                        'whitespace': table.whitespace
                    }
                }
                
                # Look for caption
                caption = find_table_caption(doc[page_num-1], bbox)
                if caption:
                    table_data['metadata']['caption'] = caption
                
                tables.append(table_data)
                
            except Exception as e:
                print(f"Error processing table {table_idx} on page {page_num}: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error extracting tables from page {page.number + 1}: {str(e)}")
    
    return tables

def clean_dataframe(df):
    """Clean and format the DataFrame"""
    # Remove empty rows and columns
    df = df.dropna(how='all').dropna(axis=1, how='all')
    
    # Replace NaN with empty string
    df = df.fillna('')
    
    # Clean cell values - using map instead of deprecated applymap
    df = df.map(lambda x: str(x).strip())
    
    # If first row looks like headers, use it
    if len(df) > 0 and df.iloc[0].notna().all():  # Fixed ambiguous truth value
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)
    
    return df

def format_dataframe(df):
    """Convert DataFrame to formatted text"""
    # Calculate column widths
    col_widths = [max(df[col].astype(str).map(len).max(), len(str(col))) 
                  for col in df.columns]
    
    # Format headers
    headers = [str(col).ljust(width) for col, width in zip(df.columns, col_widths)]
    rows = [" | ".join(headers)]
    
    # Add separator
    separator = "-" * len(rows[0])
    rows.append(separator)
    
    # Format data rows
    for _, row in df.iterrows():
        formatted_row = [str(val).ljust(width) 
                        for val, width in zip(row, col_widths)]
        rows.append(" | ".join(formatted_row))
    
    return "\n".join(rows)

def get_table_bbox(page, table_idx):
    """Get approximate bounding box for table"""
    # This is a rough estimation - you might need to adjust based on your PDFs
    page_height = page.rect.height
    page_width = page.rect.width
    
    # Divide page into sections based on number of tables found
    section_height = page_height / 4  # Assume max 4 tables per page
    top = section_height * table_idx
    bottom = top + section_height
    
    return [0, top, page_width, bottom]

def find_table_caption(page, table_bbox):
    """Find caption near table location"""
    caption_text = ""
    blocks = page.get_text("dict")["blocks"]
    
    # Look for text blocks near the table
    for block in blocks:
        if block["type"] == 0:  # Text block
            text = " ".join([span["text"] for line in block.get("lines", []) 
                           for span in line.get("spans", [])])
            
            # Look for table captions or references
            if any(text.lower().startswith(prefix) for prefix in ["table", "tbl", "tab."]):
                # Check if text is near table
                if is_near_bbox(block["bbox"], table_bbox, threshold=50):
                    caption_text = text
                    break
    
    return caption_text or "No caption"

def is_near_bbox(bbox1, bbox2, threshold=50):
    """Check if two bounding boxes are near each other"""
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    
    # Check vertical and horizontal distance
    vertical_dist = min(abs(y2 - y3), abs(y1 - y4))
    horizontal_dist = min(abs(x2 - x3), abs(x1 - x4))
    
    return vertical_dist < threshold and horizontal_dist < threshold

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
