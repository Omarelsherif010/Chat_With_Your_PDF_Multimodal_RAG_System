import os
from typing import Optional, Dict, List
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from pathlib import Path
import fitz  # PyMuPDF
import base64
import json
from datetime import datetime

def extract_images_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Extract images from PDF and convert them to base64.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List[Dict]: List of dictionaries containing image information
    """
    images = []
    doc = fitz.open(pdf_path)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images()
        
        for img_idx, img in enumerate(image_list):
            try:
                # Get image data
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Convert to base64
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                
                # Store image information
                images.append({
                    'page_num': page_num + 1,
                    'image_num': img_idx + 1,
                    'base64': base64_image,
                    'mime_type': base_image["ext"],
                    'size': len(image_bytes)
                })
                
            except Exception as e:
                print(f"Warning: Could not extract image {img_idx + 1} from page {page_num + 1}: {e}")
    
    return images

def extract_text_from_pdf_llama_parse(
    file_path: str,
    api_key: Optional[str] = None,
    result_path: Optional[str] = None
) -> str:
    """
    Extract text and images from a PDF file and save as markdown and JSON.
    
    Args:
        file_path: Path to the PDF file
        api_key: Llama Parse API key (optional, defaults to environment variable)
        result_path: Path to save the markdown output (optional)
        
    Returns:
        str: Extracted text in markdown format
    """
    # Load environment variables
    load_dotenv()
    
    # Get API key from environment variable if not provided
    if api_key is None:
        api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        if not api_key:
            raise ValueError("Llama Parse API key not found. Please provide it or set LLAMA_CLOUD_API_KEY environment variable.")

    try:
        # Get total number of pages
        doc = fitz.open(file_path)
        total_pages = len(doc)
        doc.close()
        
        print(f"Processing PDF with {total_pages} pages...")

        # Extract images first
        print("Extracting images...")
        images = extract_images_from_pdf(file_path)
        print(f"Found {len(images)} images")

        # Initialize Llama Parse with all pages configuration
        parser = LlamaParse(
            api_key=api_key,
            result_type="markdown",
            language="en",
            first_page=1,
            last_page=total_pages,
            include_page_breaks=True,
            verbose=True,
        )
        
        # Set up file extractor
        file_extractor = {".pdf": parser}
        
        # Use SimpleDirectoryReader to parse the file
        reader = SimpleDirectoryReader(
            input_files=[file_path],
            file_extractor=file_extractor
        )
        
        print("Extracting content...")
        # Load and extract content
        documents = reader.load_data()
        print(f"Found {len(documents)} documents")
        
        if not documents:
            raise Exception("No content extracted from PDF")
        
        # Prepare output data
        output_data = {
            'metadata': {
                'source': str(Path(file_path).name),
                'total_pages': total_pages,
                'extracted_pages': len(documents),
                'total_images': len(images),
                'extraction_date': datetime.now().isoformat(),
            },
            'pages': [],
            'images': images
        }
        
        # Process each page
        print("\nProcessing pages:")
        for i, doc in enumerate(documents, 1):
            print(f"- Processing page {i}/{len(documents)}")
            page_content = doc.text.strip()
            content_preview = page_content[:100] + "..." if len(page_content) > 100 else page_content
            print(f"  Content preview: {content_preview}")
            
            # Add page to output data
            output_data['pages'].append({
                'page_num': i,
                'content': page_content
            })
        
        if result_path:
            # Create output directory
            output_dir = Path(result_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save JSON data
            json_path = output_dir / f"{Path(result_path).stem}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"JSON data saved to: {json_path}")
            
            # Generate and save markdown
            markdown_content = [
                f"# {output_data['metadata']['source']}\n",
                "## Metadata\n",
                f"- Total Pages: {output_data['metadata']['total_pages']}",
                f"- Extracted Pages: {output_data['metadata']['extracted_pages']}",
                f"- Total Images: {output_data['metadata']['total_images']}",
                f"- Extraction Date: {output_data['metadata']['extraction_date']}",
                "\n---\n"
            ]
            
            # Add pages
            for page in output_data['pages']:
                markdown_content.extend([
                    f"\n## Page {page['page_num']}\n",
                    page['content'],
                    "\n---\n"
                ])
            
            # Add images
            if images:
                markdown_content.append("\n## Images\n")
                for img in images:
                    markdown_content.extend([
                        f"\n### Image {img['image_num']} (Page {img['page_num']})\n",
                        f"![Image {img['image_num']}](data:image/{img['mime_type']};base64,{img['base64']})\n",
                        "\n---\n"
                    ])
            
            # Save markdown
            with open(result_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(markdown_content))
            print(f"Markdown content saved to: {result_path}")
            
        return output_data

    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        raise

def main():
    try:
        pdf_path = os.path.join("data", "attention_paper.pdf")
        output_path = "llama_parse_output/llama_parse_output_4.md"
        
        content = extract_text_from_pdf_llama_parse(
            file_path=pdf_path,
            result_path=output_path
        )
        print("Successfully extracted content")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
