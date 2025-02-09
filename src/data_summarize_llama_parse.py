import os
from typing import Dict, List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import time
from tqdm import tqdm
import json
from pathlib import Path

def load_parsed_content(json_path: str) -> Dict:
    """Load parsed content from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_with_rate_limit(items: List, process_fn, batch_size: int = 5, delay: int = 2):
    """Process items in batches with rate limiting."""
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

class ContentSummarizer:
    def __init__(self, model_name: str = "llama-3.1-8b-instant", temperature: float = 0.5):
        """Initialize the summarizer with specified model."""
        load_dotenv()
        
        # Set up environment variables
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        
        # Disable LangSmith tracing
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        
        # Initialize the model
        self.model = ChatGroq(
            temperature=temperature,
            model=model_name,
            max_retries=3
        )
        
        # Set up prompts
        self.text_prompt = ChatPromptTemplate.from_template("""
            You are an assistant tasked with summarizing academic text.
            Give a concise summary of the following text chunk.
            Focus on key points and maintain academic tone.
            
            Text: {content}
            """)
        
        self.image_prompt = ChatPromptTemplate.from_template("""
            You are an assistant tasked with describing academic figures.
            This image is from a research paper about transformer architecture.
            Describe the key elements and significance of the figure.
            
            Image caption or context: {caption}
            """)
        
        # Set up chains
        self.text_chain = {"content": lambda x: x} | self.text_prompt | self.model | StrOutputParser()

    def summarize_pages(self, pages: List[Dict]) -> List[Dict]:
        """Summarize text content from pages."""
        print("\nSummarizing pages...")
        
        def process_batch(batch):
            contents = [page['content'] for page in batch]
            summaries = self.text_chain.batch(contents, {"max_concurrency": 1})
            return [{"page_num": page['page_num'], "summary": summary} 
                   for page, summary in zip(batch, summaries)]
        
        return process_with_rate_limit(pages, process_batch)

    def summarize_images(self, images: List[Dict]) -> List[Dict]:
        """Summarize images using GPT-4o-mini."""
        print("\nSummarizing images...")
        

        if not self.openai_api_key:
            print("Warning: OpenAI API key not found. Skipping image summarization.")
            return []
        
        vision_model = ChatOpenAI(model="gpt-4o-mini-2024-07-18", max_tokens=500)
        
        def process_batch(batch):
            results = []
            for img in batch:
                try:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Describe this academic figure in detail."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/{img['mime_type']};base64,{img['base64']}"
                                    }
                                }
                            ]
                        }
                    ]
                    summary = vision_model.invoke(messages).content
                    results.append({
                        "page_num": img['page_num'],
                        "image_num": img['image_num'],
                        "summary": summary
                    })
                except Exception as e:
                    print(f"Error processing image {img['image_num']}: {e}")
            return results
        
        return process_with_rate_limit(images, process_batch, batch_size=1)

    def save_summaries(self, summaries: Dict, output_path: str):
        """Save summaries to markdown and JSON files."""
        # Create output directory
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        json_path = output_dir / f"{Path(output_path).stem}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)
        
        # Generate markdown
        markdown_content = [
            "# Document Summaries\n",
            "## Page Summaries\n"
        ]
        
        for page in summaries['page_summaries']:
            markdown_content.extend([
                f"### Page {page['page_num']}\n",
                page['summary'],
                "\n---\n"
            ])
        
        if summaries['image_summaries']:
            markdown_content.append("\n## Image Summaries\n")
            for img in summaries['image_summaries']:
                markdown_content.extend([
                    f"### Image {img['image_num']} (Page {img['page_num']})\n",
                    img['summary'],
                    "\n---\n"
                ])
        
        # Save markdown
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(markdown_content))
        
        print(f"Summaries saved to:\n- {output_path}\n- {json_path}")

def main():
    # Load parsed content
    input_json = "llama_parse_output/llama_parse_output_4.json"
    output_path = "llama_parse_summary/summaries_5.md"
    

    try:
        # Load parsed content
        print(f"Loading parsed content from {input_json}")
        content = load_parsed_content(input_json)
        
        # Initialize summarizer
        summarizer = ContentSummarizer()
        
        # Generate summaries
        page_summaries = summarizer.summarize_pages(content['pages'])
        image_summaries = summarizer.summarize_images(content['images'])
        
        # Combine summaries
        summaries = {
            'metadata': content['metadata'],
            'page_summaries': page_summaries,
            'image_summaries': image_summaries
        }
        
        # Save summaries
        summarizer.save_summaries(summaries, output_path)
        print("Summarization completed successfully")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
