import os
import json
from llama_parse import LlamaParse
from PIL import Image
import io
import base64
from dotenv import load_dotenv
import hashlib
from pathlib import Path
import fitz  # PyMuPDF
from tqdm import tqdm
from datetime import datetime

# Load environment variables
load_dotenv()

# Define paths
output_path = "./pdf_extracted_content/"
file_path = 'data/attention_paper.pdf'

# Create necessary directories
os.makedirs(output_path, exist_ok=True)
os.makedirs("./parse_cache/", exist_ok=True)

class PDFParser:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('LLAMA_CLOUD_API_KEY')
        if not self.api_key:
            raise ValueError("Llama API key must be provided")

        # Create output directories
        self.output_dir = Path("./pdf_extracted_content")
        self.cache_dir = Path("./parse_cache")
        self.output_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)

        # Define parsing instructions for research papers
        self.parsing_instruction = """
        Research Paper Parsing Instructions:

        Extract the following elements:
        1. Text Content:
           - Title, Abstract, Section headers
           - Main body text
           - Figure and table captions
           - References

        2. Tables:
           - Full table content
           - Table headers
           - Table captions
           - Table footnotes

        3. Images/Figures:
           - Figure content
           - Figure captions
           - Equations (as images)
           - Diagrams and charts

        4. Metadata:
           - Page numbers
           - Section types
           - Font information
           - Layout structure

        Maintain relationships between:
        - Figures and their captions
        - Tables and their captions
        - References in text to figures/tables
        - Section hierarchy

        Output Format:
        - Text: Structured with headers and paragraphs
        - Tables: HTML format with headers
        - Images: High-quality extraction with captions
        """

    def get_file_hash(self, file_path):
        """Generate hash of file content for caching"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()

    def load_cache(self, file_path):
        """Load cached parse result if available"""
        try:
            file_hash = self.get_file_hash(file_path)
            cache_file = self.cache_dir / f"{file_hash}.json"
            
            if not cache_file.exists():
                return None
                
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}")
            return None

    def save_cache(self, file_path, parse_result):
        """Save parse result to cache"""
        try:
            file_hash = self.get_file_hash(file_path)
            cache_file = self.cache_dir / f"{file_hash}.json"
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(parse_result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving cache: {e}")

    def extract_elements(self, file_path):
        """Extract elements from PDF using LlamaParse"""
        try:
            # Check cache first
            cached_result = self.load_cache(file_path)
            if cached_result:
                print("Using cached parse result")
                return cached_result

            print("Processing PDF elements...")
            # Process and structure the results
            result = {
                'texts': [],
                'images': [],
                'tables': []
            }

            # Get raw document from the wrapper function
            raw_doc = self.get_raw_document(file_path)
            if not raw_doc:
                return self._fallback_extraction(file_path)

            for element in tqdm(raw_doc.elements, desc="Processing elements"):
                try:
                    if element.type == "text":
                        result['texts'].append({
                            'content': element.text,
                            'metadata': {
                                'page': element.metadata.page_number,
                                'type': self._determine_text_type(element),
                                'font_info': self._extract_font_info(element)
                            }
                        })
                    elif element.type == "table":
                        result['tables'].append({
                            'content': element.table.data,
                            'html': element.table.html,
                            'metadata': {
                                'page': element.metadata.page_number,
                                'caption': self._find_caption(element, "table"),
                                'headers': element.table.header if hasattr(element.table, 'header') else None
                            }
                        })
                    elif element.type == "image":
                        image_data = self._process_image(element)
                        if image_data:
                            result['images'].append(image_data)
                except Exception as e:
                    print(f"Error processing element: {e}")
                    continue

            # Save to cache
            self.save_cache(file_path, result)
            
            return result

        except Exception as e:
            print(f"Error extracting PDF elements: {e}")
            return self._fallback_extraction(file_path)

    def get_raw_document(self, file_path):
        """Get raw document from LlamaParse"""
        try:
            parser = LlamaParse(
                api_key=self.api_key,
                result_type="markdown",
                verbose=True,
                language="en",
                output_tables_as_HTML=True,
                ocr_languages=['eng'],
                enable_table_detection=True,
                max_retries=3,
                parsing_instruction=self.parsing_instruction
            )
            return parser.parse_file(file_path)
        except Exception as e:
            print(f"Error getting raw document: {e}")
            return None

    def _determine_text_type(self, element):
        """Determine type of text element"""
        text = element.text.lower().strip()
        metadata = element.metadata
        
        if metadata.is_title or (metadata.font_size and metadata.font_size > 14):
            return "title"
        elif metadata.is_header or text.startswith(('abstract', 'introduction', 'conclusion')):
            return "heading"
        elif text.startswith(('figure', 'fig.', 'table')):
            return "caption"
        elif metadata.is_footer:
            return "footer"
        else:
            return "paragraph"

    def _extract_font_info(self, element):
        """Extract font information from element"""
        return {
            'size': getattr(element.metadata, 'font_size', None),
            'name': getattr(element.metadata, 'font_name', None),
            'is_bold': getattr(element.metadata, 'is_bold', False),
            'is_italic': getattr(element.metadata, 'is_italic', False)
        }

    def _find_caption(self, element, elem_type):
        """Find caption for table or figure"""
        caption = element.metadata.get('caption', '')
        if not caption and hasattr(element, 'nearby_text'):
            for text in element.nearby_text:
                if text.lower().startswith((elem_type, f'{elem_type}.')):
                    caption = text
                    break
        return caption or f'{elem_type.title()} on page {element.metadata.page_number}'

    def _process_image(self, element):
        """Process and save image element"""
        try:
            image_data = element.image.data
            if not image_data:
                return None

            # Convert to PIL Image for processing
            img = Image.open(io.BytesIO(image_data))
            
            # Skip small or low-quality images
            if img.size[0] < 50 or img.size[1] < 50:
                return None

            # Generate filename and save path
            filename = f"image_page{element.metadata.page_number}_{hash(str(image_data))[:8]}.png"
            filepath = self.output_dir / filename

            # Save processed image
            img.save(filepath, format='PNG', optimize=True)

            return {
                'filepath': str(filepath),
                'metadata': {
                    'page': element.metadata.page_number,
                    'caption': self._find_caption(element, "figure"),
                    'size': img.size,
                    'format': img.format
                }
            }

        except Exception as e:
            print(f"Error processing image: {e}")
            return None

    def _fallback_extraction(self, file_path):
        """Fallback PDF extraction using PyMuPDF"""
        print("Using fallback PDF extraction...")
        doc = fitz.open(file_path)
        texts = []
        images = []
        tables = []
        
        # Track image positions for caption matching
        image_positions = []
        
        # First pass: extract text and track potential captions
        captions = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if block["type"] == 0:  # Text block
                    text = " ".join(span["text"] for line in block.get("lines", []) 
                                  for span in line.get("spans", []))
                    
                    # Track potential captions
                    if text.lower().startswith(('figure', 'fig.', 'table')):
                        captions.append({
                            'text': text,
                            'bbox': block["bbox"],
                            'page': page_num
                        })
                    
                    # Add text block
                    if text.strip():
                        texts.append({
                            'content': text,
                            'metadata': {
                                'page': page_num + 1,
                                'type': determine_block_type(block),
                                'font_size': get_block_font_size(block),
                                'position': block["bbox"],
                                'is_bold': is_block_bold(block)
                            }
                        })
        
        # Second pass: extract and process images
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract images with better quality
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    if base_image:
                        image_data = base_image["image"]
                        image = Image.open(io.BytesIO(image_data))
                        
                        # Skip small or low-quality images
                        if not is_valid_image(image):
                            continue
                        
                        # Find nearest caption
                        caption = find_nearest_caption(
                            page_num,
                            page.get_image_bbox(xref),
                            captions
                        )
                        
                        image_filename = f"image_page{page_num + 1}_idx{img_index}.b64"
                        image_path = os.path.join(output_path, image_filename)
                        
                        # Process and save image
                        img_processed = process_image(image)
                        if img_processed:
                            with open(image_path, 'wb') as f:
                                f.write(base64.b64encode(img_processed))
                            
                            images.append({
                                'content': base64.b64encode(img_processed).decode(),
                                'metadata': {
                                    'page': page_num + 1,
                                    'index': img_index,
                                    'size': image.size,
                                    'caption': caption,
                                    'filename': image_filename,
                                    'filepath': image_path,
                                    'format': image.format or 'unknown',
                                    'dpi': base_image.get("dpi", (72,72))
                                }
                            })
                            
                except Exception as img_error:
                    print(f"Error extracting image: {str(img_error)}")
                    continue
        
        print(f"Fallback extraction found: {len(texts)} texts, {len(images)} images")
        return texts, images, tables

    def save_as_markdown(self, result, file_path):
        """Save parsed results as markdown file"""
        try:
            # Create markdown output directory
            markdown_dir = Path("./markdown_output")
            markdown_dir.mkdir(exist_ok=True)
            
            # Generate markdown filename from PDF name
            pdf_name = Path(file_path).stem
            markdown_path = markdown_dir / f"{pdf_name}_parsed.md"
            
            markdown_content = []
            
            # Add title
            markdown_content.append(f"# {pdf_name}\n\n")
            
            # Add timestamp
            markdown_content.append(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            
            # Add table of contents
            markdown_content.append("## Table of Contents\n")
            markdown_content.append("1. [Text Content](#text-content)\n")
            markdown_content.append("2. [Tables](#tables)\n")
            markdown_content.append("3. [Images](#images)\n\n")
            
            # Add text content
            markdown_content.append("## Text Content\n\n")
            if result['texts']:
                for text in result['texts']:
                    text_type = text['metadata'].get('type', 'text')
                    page = text['metadata'].get('page', 'Unknown')
                    
                    # Format based on text type
                    if text_type == 'title':
                        markdown_content.append(f"### {text['content']}\n")
                    elif text_type == 'heading':
                        markdown_content.append(f"#### {text['content']}\n")
                    else:
                        markdown_content.append(f"{text['content']}\n")
                    markdown_content.append(f"*[Page {page}]*\n\n")
            else:
                markdown_content.append("No text content extracted.\n\n")
            
            # Add tables
            markdown_content.append("## Tables\n\n")
            if result['tables']:
                for i, table in enumerate(result['tables'], 1):
                    page = table['metadata'].get('page', 'Unknown')
                    caption = table['metadata'].get('caption', 'No caption')
                    
                    markdown_content.append(f"### Table {i}\n")
                    markdown_content.append(f"**Caption:** {caption}\n")
                    markdown_content.append(f"*[Page {page}]*\n\n")
                    
                    # Add table content
                    if 'html' in table:
                        markdown_content.append(table['html'])
                    else:
                        markdown_content.append("```\n")
                        markdown_content.append(format_table(table['content']))
                        markdown_content.append("\n```\n")
                    markdown_content.append("\n\n")
            else:
                markdown_content.append("No tables extracted.\n\n")
            
            # Add images
            markdown_content.append("## Images\n\n")
            if result['images']:
                for i, image in enumerate(result['images'], 1):
                    page = image['metadata'].get('page', 'Unknown')
                    caption = image['metadata'].get('caption', 'No caption')
                    filepath = image['filepath']
                    
                    markdown_content.append(f"### Figure {i}\n")
                    markdown_content.append(f"**Caption:** {caption}\n")
                    markdown_content.append(f"*[Page {page}]*\n\n")
                    
                    # Add image reference
                    if os.path.exists(filepath):
                        # Calculate relative path
                        rel_path = os.path.relpath(filepath, markdown_dir)
                        markdown_content.append(f"![Figure {i}]({rel_path})\n\n")
                    else:
                        markdown_content.append("*Image file not found*\n\n")
            else:
                markdown_content.append("No images extracted.\n\n")
            
            # Write to file
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(''.join(markdown_content))
            
            print(f"Markdown file saved to: {markdown_path}")
            return markdown_path
            
        except Exception as e:
            print(f"Error saving markdown file: {e}")
            return None

def is_valid_image(image):
    """Check if image meets quality criteria"""
    try:
        # Size checks
        if image.size[0] < 50 or image.size[1] < 50:
            return False
            
        # Basic quality checks
        if image.mode == '1':  # Binary image
            return False
            
        # Check for mostly empty or solid color
        if image.mode in ('RGB', 'RGBA'):
            colors = image.getcolors(maxcolors=256)
            if colors is not None and len(colors) < 3:
                return False
        
        return True
        
    except Exception:
        return False

def find_nearest_caption(page_num, image_bbox, captions):
    """Find the nearest caption to an image"""
    nearest_caption = None
    min_distance = float('inf')
    
    for caption in captions:
        if abs(caption['page'] - page_num) <= 1:  # Check same page and adjacent pages
            distance = bbox_distance(image_bbox, caption['bbox'])
            if distance < min_distance:
                min_distance = distance
                nearest_caption = caption['text']
    
    return nearest_caption or f'Image on page {page_num + 1}'

def process_parsed_elements(elements):
    """Process parsed elements into texts, images, and tables"""
    texts = []
    images = []
    tables = []
    
    # Track elements for deduplication
    seen_images = set()
    seen_tables = set()
    
    for element in elements:
        try:
            if isinstance(element, dict):
                # Handle cached elements
                elem_type = element['type']
                content = element['content']
                metadata = element['metadata']
            else:
                # Handle direct LlamaParse elements
                elem_type = element.type
                metadata = element.metadata
                content = element
            
            if elem_type == "text":
                # Process text element
                text_content = content.get('text', '') if isinstance(content, dict) else content.text
                text_content = text_content.strip()
                if not text_content:
                    continue
                
                texts.append({
                    'content': text_content,
                    'metadata': {
                        'page': metadata.get('page_number', 1),
                        'type': determine_text_type(element),
                        'font_size': metadata.get('font_size', 0),
                        'position': metadata.get('bbox', [0,0,0,0]),
                        'font_name': metadata.get('font_name', 'unknown'),
                        'is_bold': metadata.get('is_bold', False),
                        'is_italic': metadata.get('is_italic', False)
                    }
                })
            
            elif elem_type == "table":
                # Skip duplicate tables
                table_hash = hash(str(content))
                if table_hash in seen_tables:
                    continue
                seen_tables.add(table_hash)
                
                # Enhanced table processing
                table_data = clean_table_data(content)
                if not table_data:  # Skip empty tables
                    continue
                
                tables.append({
                    'content': table_data,
                    'text': format_table(content),
                    'bbox': metadata.get('bbox', [0,0,0,0]),
                    'metadata': {
                        'page': metadata.get('page_number', 1),
                        'type': 'table',
                        'rows': len(table_data),
                        'columns': len(table_data[0]) if table_data else 0,
                        'headers': metadata.get('table_header', []),
                        'caption': extract_table_caption(element),
                        'confidence': metadata.get('confidence', 1.0)
                    }
                })
            
            elif elem_type == "image":
                try:
                    # Get image data
                    if isinstance(content, dict):
                        image_data = base64.b64decode(content['image_data']) if content.get('image_data') else None
                    else:
                        image_data = content.image.data if hasattr(content, 'image') else None
                    
                    if not image_data:
                        continue
                    
                    # Generate unique filename
                    image_filename = f"image_page{metadata.get('page_number', 1)}_idx{len(images)}.b64"
                    image_path = os.path.join(output_path, image_filename)
                    
                    # Process and save image
                    img = Image.open(io.BytesIO(image_data))
                    img_processed = process_image(img)
                    
                    # Save processed image
                    with open(image_path, 'wb') as f:
                        f.write(base64.b64encode(img_processed))
                    
                    images.append({
                        'content': base64.b64encode(img_processed).decode(),
                        'metadata': {
                            'page': metadata.get('page_number', 1),
                            'index': len(images),
                            'size': img.size,
                            'caption': extract_image_caption(element),
                            'filename': image_filename,
                            'filepath': image_path,  # Add filepath for retrieval
                            'format': img.format or 'unknown',
                            'confidence': metadata.get('confidence', 1.0)
                        }
                    })
                except Exception as e:
                    print(f"Error processing image: {str(e)}")
                    continue
            
        except Exception as e:
            print(f"Error processing element {elem_type}: {str(e)}")
            continue
    
    return texts, images, tables

def clean_table_data(data):
    """Clean and validate table data"""
    if not data:
        return []
    
    # Remove empty rows and columns
    cleaned_data = [
        [str(cell).strip() for cell in row if str(cell).strip()]
        for row in data
        if any(str(cell).strip() for cell in row)
    ]
    
    return cleaned_data if cleaned_data else []

def extract_table_caption(element):
    """Extract table caption from metadata or nearby text"""
    caption = element.metadata.get('caption', '')
    if not caption:
        # Try to find caption in nearby text elements
        if hasattr(element, 'nearby_text'):
            for text in element.nearby_text:
                if text.lower().startswith('table'):
                    caption = text
                    break
    return caption or 'No caption'

def extract_image_caption(element):
    """Extract image caption from metadata or nearby text"""
    caption = element.metadata.get('caption', '')
    if not caption:
        # Try to find caption in nearby text elements
        if hasattr(element, 'nearby_text'):
            for text in element.nearby_text:
                if text.lower().startswith(('figure', 'fig.')):
                    caption = text
                    break
    return caption or 'No caption'

def process_image(image):
    """Process image data before saving"""
    try:
        # Convert to RGB if needed
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        
        # Optimize image
        buffered = io.BytesIO()
        image.save(buffered, format='PNG', optimize=True)
        return buffered.getvalue()
        
    except Exception as e:
        print(f"Error processing image data: {str(e)}")
        return None

def determine_text_type(element):
    """Determine the type of text based on metadata"""
    if element.metadata.is_header:
        return "heading"
    elif element.metadata.is_title:
        return "title"
    elif element.metadata.is_caption:
        return "caption"
    else:
        return "paragraph"

def format_table(table):
    """Format table data into a string representation"""
    if not table:
        return ""
    
    # Calculate column widths
    col_widths = []
    for col in range(len(table[0])):
        col_data = [str(row[col]) for row in table]
        col_widths.append(max(len(str(x)) for x in col_data))
    
    # Format the table
    lines = []
    
    # Add headers if available
    if len(table) > 0 and len(table[0]) > 0:
        header_line = " | ".join(str(h).ljust(w) for h, w in zip(table[0], col_widths))
        lines.append(header_line)
        lines.append("-" * len(header_line))
    
    # Add data rows
    for row in table:
        line = " | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths))
        lines.append(line)
    
    return "\n".join(lines)

def determine_block_type(block):
    """Determine the type of text block based on its properties"""
    try:
        # Get the first span's properties
        spans = block.get("lines", [{}])[0].get("spans", [{}])
        if not spans:
            return "paragraph"
            
        first_span = spans[0]
        font_size = first_span.get("size", 0)
        flags = first_span.get("flags", 0)
        text = first_span.get("text", "").strip().lower()
        
        # Check for title (usually large font at start of document)
        if font_size > 14:
            return "title"
            
        # Check for heading (bold or larger font)
        if font_size > 12 or (flags & 2**4):  # 2^4 is bold flag
            return "heading"
            
        # Check for caption
        if text.startswith(("figure", "fig.", "table")):
            return "caption"
            
        # Default to paragraph
        return "paragraph"
        
    except Exception as e:
        print(f"Error determining block type: {str(e)}")
        return "paragraph"

def get_block_font_size(block):
    """Extract font size from block"""
    try:
        return block.get("lines", [{}])[0].get("spans", [{}])[0].get("size", 0)
    except:
        return 0

def is_block_bold(block):
    """Check if block contains bold text"""
    try:
        flags = block.get("lines", [{}])[0].get("spans", [{}])[0].get("flags", 0)
        return bool(flags & 2**4)  # 2^4 is bold flag
    except:
        return False

def bbox_distance(bbox1, bbox2):
    """Calculate distance between two bounding boxes"""
    x1_center = (bbox1[0] + bbox1[2]) / 2
    y1_center = (bbox1[1] + bbox1[3]) / 2
    x2_center = (bbox2[0] + bbox2[2]) / 2
    y2_center = (bbox2[1] + bbox2[3]) / 2
    
    return ((x1_center - x2_center) ** 2 + (y1_center - y2_center) ** 2) ** 0.5

def extract_pdf_elements(file_path):
    """Wrapper function to extract elements from PDF"""
    try:
        parser = PDFParser()
        
        # First try LlamaParse
        print("Attempting LlamaParse extraction...")
        try:
            # Initialize LlamaParse with specific options for research papers
            llama_parser = LlamaParse(
                api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
                result_type="markdown",
                verbose=True,
                language="en",
                output_tables_as_HTML=True,
                ocr_languages=['eng'],
                enable_table_detection=True,
                max_retries=3,
                parsing_instruction="Extract all text, tables, and figures with their captions and relationships."
            )
            
            # Parse with LlamaParse and save raw output
            print("Parsing with LlamaParse...")
            raw_doc = llama_parser.parse_file(file_path)
            
            # Save raw LlamaParse output as markdown
            markdown_dir = Path("./llama_parse_output")
            markdown_dir.mkdir(exist_ok=True)
            
            raw_markdown_path = markdown_dir / f"{Path(file_path).stem}_llama_raw.md"
            with open(raw_markdown_path, 'w', encoding='utf-8') as f:
                f.write(f"# Raw LlamaParse Output for {Path(file_path).name}\n\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("## Document Elements\n\n")
                for elem in raw_doc.elements:
                    f.write(f"### Element Type: {elem.type}\n")
                    f.write(f"Page: {getattr(elem.metadata, 'page_number', 'Unknown')}\n")
                    if elem.type == "text":
                        f.write(f"```text\n{elem.text}\n```\n\n")
                    elif elem.type == "table":
                        f.write(f"```html\n{elem.table.html}\n```\n\n")
                    elif elem.type == "image":
                        f.write("*[Image Data Present]*\n\n")
                    f.write("---\n\n")
            
            print(f"Raw LlamaParse output saved to: {raw_markdown_path}")
            
            # Process the elements
            result = parser.extract_elements(file_path)
            
            # Save processed output as markdown
            if not isinstance(result, tuple):
                processed_markdown_path = markdown_dir / f"{Path(file_path).stem}_processed.md"
                parser.save_as_markdown(result, processed_markdown_path)
                print(f"Processed output saved to: {processed_markdown_path}")
                
                # Return structured content
                return result['texts'], result['images'], result['tables']
            else:
                return result
                
        except Exception as llama_error:
            print(f"LlamaParse failed: {str(llama_error)}")
            print("Falling back to PyMuPDF extraction...")
            return parser._fallback_extraction(file_path)
            
    except Exception as e:
        print(f"Error in extract_pdf_elements: {e}")
        return [], [], []

def display_saved_image(base64_file):
    """Display an image from a saved base64 file"""
    try:
        # Read file in binary mode
        with open(base64_file, 'rb') as f:
            base64_data = f.read()
        
        # Decode base64 string back to image
        img_bytes = base64.b64decode(base64_data)
        img = Image.open(io.BytesIO(img_bytes))
        
        # For Streamlit compatibility, return the image object
        return img
        
    except Exception as e:
        print(f"Error displaying image {base64_file}: {str(e)}")
        return None

def main():
    texts, images, tables = extract_pdf_elements(file_path)
    
    print(f"Extracted {len(texts)} text sections, {len(images)} images, and {len(tables)} tables")
    
    # Print sample information
    if texts:
        print("\nSample text:")
        print(texts[0]['content'][:200])
    
    if tables:
        print("\nSample table:")
        print(tables[0]['text'])
    
    if images:
        print("\nSample image metadata:")
        print(images[0]['metadata'])
    
    return texts, images, tables

if __name__ == "__main__":
    main()
