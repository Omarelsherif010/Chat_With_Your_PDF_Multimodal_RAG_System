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

        # Update parsing instructions to focus more on tables
        self.parsing_instruction = """
        Table Extraction Instructions:

        1. Tables:
           - Extract all tables with their complete content
           - Preserve table structure and formatting
           - Include table headers and column names
           - Capture table captions and footnotes
           - Maintain cell alignments and spans
           - Extract numerical data with proper formatting
           - Preserve any in-table formatting (bold, italic, etc.)

        2. Table Context:
           - Identify table references in the text
           - Extract table titles and numbering
           - Capture table descriptions from surrounding text
           - Note any dependencies between tables

        3. Additional Elements:
           - Extract text content and section headers
           - Capture figures and their captions
           - Maintain document structure

        Output Format:
        - Tables: Structured HTML format with preserved formatting
        - Table metadata: Include page numbers, captions, and reference information
        - Text: Maintain paragraph structure
        - Images: Extract with captions
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

    def get_raw_document(self, file_path):
        """Get raw document from LlamaParse with enhanced table settings"""
        try:
            parser = LlamaParse(
                api_key=self.api_key,
                result_type="markdown",
                verbose=True,
                language="en",
                output_tables_as_HTML=True,  # Ensure HTML table output
                enable_table_detection=True,  # Explicitly enable table detection
                table_detection_mode="high_quality",  # Use high quality table detection
                ocr_languages=['eng'],
                max_retries=3,
                parsing_instruction=self.parsing_instruction
            )
            return parser.parse_file(file_path)
        except Exception as e:
            print(f"Error getting raw document: {e}")
            return None

    def _process_table(self, element):
        """Enhanced table processing"""
        try:
            if not hasattr(element, 'table') or not element.table:
                return None

            # Extract table HTML and data
            table_data = {
                'html': element.table.html if hasattr(element.table, 'html') else None,
                'data': element.table.data if hasattr(element.table, 'data') else None,
                'metadata': {
                    'page': element.metadata.page_number,
                    'caption': self._find_caption(element, "table"),
                    'headers': element.table.header if hasattr(element.table, 'header') else None,
                    'rows': len(element.table.data) if hasattr(element.table, 'data') else 0,
                    'columns': len(element.table.data[0]) if hasattr(element.table, 'data') and element.table.data else 0,
                    'bbox': getattr(element.metadata, 'bbox', None)
                }
            }

            # Validate table data
            if not table_data['data'] and not table_data['html']:
                return None

            return table_data

        except Exception as e:
            print(f"Error processing table: {e}")
            return None

    def _fallback_table_extraction(self, page):
        """Extract tables using PyMuPDF fallback"""
        tables = []
        try:
            # Find tables on page
            tab = page.find_tables()
            if not tab.tables:
                return []
                
            for idx, table in enumerate(tab.tables):
                try:
                    # Extract table data
                    data = []
                    for row in table.extract():
                        data.append([str(cell).strip() for cell in row])
                    
                    # Find potential caption
                    caption = self._find_table_caption(page, table.bbox)
                    
                    # Create table entry
                    table_data = {
                        'data': data,
                        'html': self._convert_to_html_table(data),
                        'metadata': {
                            'page': page.number + 1,
                            'caption': caption,
                            'headers': data[0] if data else [],
                            'rows': len(data),
                            'columns': len(data[0]) if data else 0,
                            'bbox': list(table.bbox)
                        }
                    }
                    tables.append(table_data)
                    
                except Exception as e:
                    print(f"Error extracting table {idx}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error in fallback table extraction: {e}")
        
        return tables

    def _convert_to_html_table(self, data):
        """Convert table data to HTML format"""
        if not data:
            return ""
        
        html = ["<table border='1'>"]
        
        # Add header row
        html.append("<thead><tr>")
        for header in data[0]:
            html.append(f"<th>{header}</th>")
        html.append("</tr></thead>")
        
        # Add data rows
        html.append("<tbody>")
        for row in data[1:]:
            html.append("<tr>")
            for cell in row:
                html.append(f"<td>{cell}</td>")
            html.append("</tr>")
        html.append("</tbody>")
        
        html.append("</table>")
        return "\n".join(html)

    def _find_table_caption(self, page, table_bbox):
        """Find table caption near the table"""
        try:
            # Get text blocks near the table
            blocks = page.get_text("dict")["blocks"]
            table_y = table_bbox[1]  # Top of table
            
            # Look for captions above and below table
            potential_captions = []
            for block in blocks:
                if block["type"] == 0:  # Text block
                    text = " ".join(span["text"] for line in block["lines"] 
                                  for span in line["spans"]).strip()
                    
                    if text.lower().startswith(("table", "tab.")):
                        # Calculate distance to table
                        block_y = block["bbox"][1]
                        distance = abs(block_y - table_y)
                        potential_captions.append((text, distance))
            
            # Return closest caption if found
            if potential_captions:
                potential_captions.sort(key=lambda x: x[1])
                return potential_captions[0][0]
                
        except Exception as e:
            print(f"Error finding table caption: {e}")
        
        return "No caption found"

    def _extract_tables_from_page(self, page, doc):
        """Extract tables from a single page using both methods"""
        tables = []
        
        # Try LlamaParse first
        try:
            llama_tables = [elem for elem in doc.elements 
                          if elem.metadata.page_number == page.number + 1 
                          and elem.type == "table"]
            
            for table_elem in llama_tables:
                table_data = self._process_table(table_elem)
                if table_data:
                    tables.append(table_data)
                    
        except Exception as e:
            print(f"Error in LlamaParse table extraction: {e}")
        
        # If no tables found, try PyMuPDF
        if not tables:
            tables.extend(self._fallback_table_extraction(page))
        
        return tables

    def extract_elements(self, file_path):
        """Extract elements from PDF using LlamaParse with fallback to PyMuPDF"""
        try:
            # Check cache first
            cached_result = self.load_cache(file_path)
            if cached_result:
                print("Using cached parse result")
                self._print_extraction_results(cached_result)
                return cached_result

            print("Processing PDF elements...")
            result = {
                'texts': [],
                'images': [],
                'tables': []
            }

            # Try LlamaParse first
            raw_doc = self.get_raw_document(file_path)
            
            # Open with PyMuPDF for fallback and additional extraction
            doc = fitz.open(file_path)
            
            # Process each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract tables (using both methods)
                page_tables = self._extract_tables_from_page(page, raw_doc if raw_doc else None)
                result['tables'].extend(page_tables)
                
                # Extract other elements...
                # (rest of existing extraction code)

            # Save to cache
            self.save_cache(file_path, result)
            
            # Print extraction results
            self._print_extraction_results(result)
            
            return result

        except Exception as e:
            print(f"Error extracting PDF elements: {e}")
            return self._fallback_extraction(file_path)

    def _print_extraction_results(self, result):
        """Print the extraction results in a formatted way"""
        print("\n=== LlamaParse Extraction Results ===\n")
        
        # Print text statistics
        print("Text Elements:")
        text_types = {}
        for text in result['texts']:
            text_type = text['metadata']['type']
            text_types[text_type] = text_types.get(text_type, 0) + 1
        for text_type, count in text_types.items():
            print(f"  - {text_type.title()}: {count}")
        
        # Print table information
        print("\nTables:")
        if result['tables']:
            for i, table in enumerate(result['tables'], 1):
                page = table['metadata'].get('page', 'Unknown')
                caption = table['metadata'].get('caption', 'No caption')
                rows = len(table['data']) if table.get('data') else 0
                cols = len(table['data'][0]) if table.get('data') and table['data'] else 0
                print(f"  {i}. Table on page {page}")
                print(f"     Rows: {rows}, Columns: {cols}")
                print(f"     Caption: {caption[:100]}{'...' if len(caption) > 100 else ''}")
        else:
            print("  No tables found")
        
        # Print image information
        print("\nImages:")
        if result['images']:
            for i, image in enumerate(result['images'], 1):
                page = image['metadata'].get('page', 'Unknown')
                caption = image['metadata'].get('caption', 'No caption')
                print(f"  {i}. Image on page {page}")
                print(f"     Caption: {caption[:100]}{'...' if len(caption) > 100 else ''}")
        else:
            print("  No images found")
        
        print("\n=====================================\n")

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
        try:
            doc = fitz.open(file_path)
            result = {
                'texts': [],
                'images': [],
                'tables': []
            }
            
            # Track image positions for caption matching
            captions = []
            
            # First pass: extract text and track potential captions
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
                            result['texts'].append({
                                'content': text,
                                'metadata': {
                                    'page': page_num + 1,
                                    'type': self._determine_fallback_text_type(block),
                                    'font_info': self._extract_fallback_font_info(block)
                                }
                            })
            
            # Second pass: extract images using improved PyMuPDF method
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get list of images on the page
                images = page.get_images(full=True)
                
                # Process each image
                for img_index, img in enumerate(images):
                    try:
                        xref = img[0]  # xref number
                        
                        # Basic image info
                        base_image = doc.extract_image(xref)
                        if not base_image:
                            continue
                            
                        # Get image properties
                        pix = fitz.Pixmap(doc, xref)
                        
                        # Skip small, low-quality, or mask images
                        if self._should_skip_image(pix):
                            pix = None
                            continue
                            
                        # Convert pixmap if necessary
                        if pix.n >= 4:  # CMYK or RGBA
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        
                        # Generate unique filename
                        filename = f"image_page{page_num + 1}_idx{img_index}.png"
                        filepath = self.output_dir / filename
                        
                        # Save image
                        pix.save(filepath)
                        pix = None  # Free memory
                        
                        # Get image rectangle on page
                        image_rect = page.get_image_bbox(xref)
                        
                        # Find nearest caption
                        caption = self._find_nearest_caption(
                            page_num,
                            image_rect,
                            captions
                        )
                        
                        # Add to results
                        result['images'].append({
                            'filepath': str(filepath),
                            'metadata': {
                                'page': page_num + 1,
                                'caption': caption,
                                'size': (base_image['width'], base_image['height']),
                                'format': base_image['ext'],
                                'colorspace': base_image['colorspace'],
                                'bbox': image_rect
                            }
                        })
                        
                    except Exception as img_error:
                        print(f"Error extracting image on page {page_num + 1}: {str(img_error)}")
                        continue
            
            print(f"Fallback extraction found: {len(result['texts'])} texts, {len(result['images'])} images")
            return result
            
        except Exception as e:
            print(f"Error in fallback extraction: {e}")
            return {'texts': [], 'images': [], 'tables': []}

    def _should_skip_image(self, pix):
        """Determine if an image should be skipped based on quality criteria"""
        try:
            # Skip if too small
            if pix.width < 50 or pix.height < 50:
                return True
                
            # Skip if it's a mask
            if pix.n == 1 and pix.is_alpha:
                return True
                
            # Skip if image appears to be a solid color or nearly empty
            if pix.n in (1, 3):  # Grayscale or RGB
                # Sample pixels to check for variation
                samples = []
                for x in range(0, pix.width, max(1, pix.width // 10)):
                    for y in range(0, pix.height, max(1, pix.height // 10)):
                        samples.append(pix.pixel(x, y))
                
                # If almost all pixels are the same, skip
                unique_samples = set(samples)
                if len(unique_samples) < 3:
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error checking image quality: {e}")
            return True

    def _find_nearest_caption(self, page_num, image_bbox, captions):
        """Find nearest caption to an image using improved distance calculation"""
        nearest_caption = None
        min_distance = float('inf')
        
        # Convert bbox to rectangle for easier calculations
        img_rect = fitz.Rect(image_bbox)
        
        for caption in captions:
            if abs(caption['page'] - page_num) <= 1:  # Check same page and adjacent pages
                cap_rect = fitz.Rect(caption['bbox'])
                
                # Calculate distance between rectangles
                distance = self._rect_distance(img_rect, cap_rect)
                
                # Prefer captions below images
                if cap_rect.y0 >= img_rect.y1:  # Caption is below image
                    distance *= 0.8  # Give preference to captions below images
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_caption = caption['text']
        
        return nearest_caption or f'Image on page {page_num + 1}'

    def _rect_distance(self, rect1, rect2):
        """Calculate distance between two rectangles"""
        # If rectangles intersect, distance is 0
        if rect1.intersects(rect2):
            return 0
            
        # Calculate closest points
        x1 = max(rect1.x0, min(rect2.x0, rect1.x1))
        y1 = max(rect1.y0, min(rect2.y0, rect1.y1))
        x2 = max(rect2.x0, min(x1, rect2.x1))
        y2 = max(rect2.y0, min(y1, rect2.y1))
        
        # Return Euclidean distance
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def _determine_fallback_text_type(self, block):
        """Determine text type from PyMuPDF block"""
        try:
            # Get first span's properties
            span = block["lines"][0]["spans"][0]
            font_size = span.get("size", 0)
            flags = span.get("flags", 0)
            text = span.get("text", "").strip().lower()
            
            # Check for title
            if font_size > 14:
                return "title"
            # Check for heading
            elif font_size > 12 or (flags & 16):  # 16 is bold flag
                return "heading"
            # Check for caption
            elif text.startswith(("figure", "fig.", "table")):
                return "caption"
            else:
                return "paragraph"
        except:
            return "paragraph"

    def _extract_fallback_font_info(self, block):
        """Extract font info from PyMuPDF block"""
        try:
            span = block["lines"][0]["spans"][0]
            return {
                'size': span.get("size", None),
                'name': span.get("font", None),
                'is_bold': bool(span.get("flags", 0) & 16),
                'is_italic': bool(span.get("flags", 0) & 1)
            }
        except:
            return {
                'size': None,
                'name': None,
                'is_bold': False,
                'is_italic': False
            }

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
                    text_type = text['metadata']['type']
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
                        markdown_content.append(format_table(table['data']))
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
        col_data = [str(x) for x in table[col]]
        col_widths.append(max(len(str(x)) for x in col_data)
    
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
    parser = PDFParser()
    pdf_file = "data/attention_paper.pdf"
    
    print(f"Processing PDF: {pdf_file}")
    result = parser.extract_elements(pdf_file)
    
    # Access tables
    for table in result['tables']:
        print(f"\nTable on page {table['metadata']['page']}:")
        print(f"Caption: {table['metadata']['caption']}")
        print(f"Dimensions: {table['metadata']['rows']}x{table['metadata']['columns']}")
        print(f"HTML: {table['html']}")
        print(f"Data: {table['data']}")

if __name__ == "__main__":
    main()
