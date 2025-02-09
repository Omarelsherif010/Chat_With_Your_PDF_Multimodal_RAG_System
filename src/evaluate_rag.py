import json
import logging
from pathlib import Path
from typing import Dict, List
from retrieval_llama_parse import MultimodalRetriever
from dataclasses import dataclass
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from rouge_score import rouge_scorer
import numpy as np
from cached_retriever import CachedRetriever

# Set up logging
logging.basicConfig(
    level=logging.INFO,

    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt')
except Exception as e:
    logger.warning(f"Error downloading NLTK data: {e}")

@dataclass
class EvaluationMetrics:
    """Metrics for evaluating RAG responses"""
    relevance_score: float  # 0-1: How relevant is the response to the question
    completeness_score: float  # 0-1: How complete is the response
    accuracy_score: float  # 0-1: How accurate is the information
    source_score: float  # 0-1: How well it uses retrieved sources
    image_handling_score: float = 0.0  # 0-1: How well it handles images (if applicable)

@dataclass
class ExtendedEvaluationMetrics(EvaluationMetrics):
    """Extended metrics including BLEU and ROUGE scores"""
    bleu_score: float = 0.0  # 0-1: BLEU score for response quality
    rouge_l_score: float = 0.0  # 0-1: ROUGE-L score for response quality
    context_relevance: float = 0.0  # 0-1: How well the response uses retrieved context
    factual_accuracy: float = 0.0  # 0-1: Factual accuracy based on ground truth

@dataclass
class EvaluationResult:
    """Results of a single question evaluation"""
    question: str
    response: str
    retrieved_content: Dict
    metrics: EvaluationMetrics
    timestamp: datetime = datetime.now()

class RAGEvaluator:
    def __init__(self):
        """Initialize the RAG evaluator with enhanced metrics"""
        self.test_questions = [
            {
                "question": "How is the scaled dot product attention calculated?",
                "expected_elements": [
                    "dot product of query and key",
                    "scaling factor √dk",
                    "softmax function",
                    "multiplication with values"
                ]
            },
            {
                "question": "What is the BLEU score of the model in English to German translation EN-DE?",
                "expected_elements": [
                    "BLEU score value",
                    "EN-DE translation",
                    "comparison with other models"
                ]
            },
            {
                "question": "How long were the base and big models trained?",
                "expected_elements": [
                    "base model training time",
                    "big model training time",
                    "training details"
                ]
            },
            {
                "question": "Which optimizer was used when training the models?",
                "expected_elements": [
                    "optimizer name",
                    "optimizer parameters",
                    "training details"
                ]
            },
            {
                "question": "Show me a picture that shows the difference between Scaled Dot-Product Attention and Multi-Head Attention.",
                "expected_elements": [
                    "image retrieval",
                    "image display",
                    "explanation of differences",
                    "visual components description"
                ]
            }
        ]
        
        self.results: List[EvaluationResult] = []
        
        # Add ground truth answers
        self.ground_truth = {
            "How is the scaled dot product attention calculated?": """
            The scaled dot product attention is calculated as:
            1. Computing dot product of query (Q) with key (K)
            2. Scaling by factor 1/√dk where dk is dimension of keys
            3. Applying softmax function
            4. Multiplying result with values (V)
            Formula: Attention(Q,K,V) = softmax(QK^T/√dk)V
            """,
            
            "What is the BLEU score of the model in English to German translation EN-DE?": """
            The model achieves a BLEU score of 28.4 on the WMT 2014 English-to-German translation task (EN-DE).
            The big transformer model outperforms the best previously reported models by more than 2.0 BLEU points.
            """,
            
            "How long were the base and big models trained?": """
            The base model was trained for 100,000 steps or 12 hours on 8 P100 GPUs.
            The big model was trained for 300,000 steps or 3.5 days on 8 P100 GPUs.
            """,
            
            "Which optimizer was used when training the models?": """
            The Adam optimizer was used with:
            - β1 = 0.9
            - β2 = 0.98
            - ε = 10^-9
            Learning rate varied over training according to formula with warmup_steps=4000.
            """,
            
            "Show me a picture that shows the difference between Scaled Dot-Product Attention and Multi-Head Attention.": """
            The figure shows:
            1. Scaled Dot-Product Attention with Q, K, V inputs
            2. Multi-Head Attention with multiple parallel attention heads
            3. Concatenation and linear projection in Multi-Head Attention
            The diagram clearly illustrates the architectural differences between the two attention mechanisms.
            """
        }
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        # Initialize BLEU smoothing
        self.smoothing = SmoothingFunction().method1
        
        # Initialize the cached retriever
        self.retriever = CachedRetriever()
        logger.info("Initialized CachedRetriever")
        
    def calculate_bleu_score(self, reference: str, candidate: str) -> float:
        """Calculate BLEU score between reference and candidate texts"""
        try:
            # Tokenize reference and candidate
            reference_tokens = nltk.word_tokenize(reference.lower())
            candidate_tokens = nltk.word_tokenize(candidate.lower())
            
            # Calculate BLEU score with smoothing
            return sentence_bleu([reference_tokens], candidate_tokens, 
                               smoothing_function=self.smoothing)
        except Exception as e:
            logger.error(f"Error calculating BLEU score: {e}")
            return 0.0
            
    def calculate_rouge_score(self, reference: str, candidate: str) -> float:
        """Calculate ROUGE-L score between reference and candidate texts"""
        try:
            scores = self.rouge_scorer.score(reference, candidate)
            return scores['rougeL'].fmeasure
        except Exception as e:
            logger.error(f"Error calculating ROUGE score: {e}")
            return 0.0
            
    def evaluate_context_relevance(self, response: str, retrieved_content: Dict) -> float:
        """Evaluate how well the response uses the retrieved context"""
        try:
            # Extract key terms from retrieved content
            context_terms = set()
            for doc in retrieved_content["texts"] + retrieved_content["tables"]:
                context_terms.update(nltk.word_tokenize(doc.page_content.lower()))
            
            # Extract terms from response
            response_terms = set(nltk.word_tokenize(response.lower()))
            
            # Calculate overlap
            if not context_terms:
                return 0.0
                
            overlap = len(context_terms.intersection(response_terms))
            return min(1.0, overlap / len(context_terms))
            
        except Exception as e:
            logger.error(f"Error evaluating context relevance: {e}")
            return 0.0
            
    def evaluate_factual_accuracy(self, question: str, response: str) -> float:
        """Evaluate factual accuracy using ground truth"""
        try:
            if question not in self.ground_truth:
                return 0.5  # Default score when no ground truth available
                
            # Calculate BLEU and ROUGE scores against ground truth
            bleu = self.calculate_bleu_score(self.ground_truth[question], response)
            rouge = self.calculate_rouge_score(self.ground_truth[question], response)
            
            # Combine scores
            return (bleu + rouge) / 2
            
        except Exception as e:
            logger.error(f"Error evaluating factual accuracy: {e}")
            return 0.0

    def evaluate_response(self, question: Dict, response: str, retrieved_content: Dict) -> ExtendedEvaluationMetrics:
        """Enhanced evaluation with additional metrics"""
        # Calculate base metrics directly instead of using super()
        found_elements = sum(1 for elem in question["expected_elements"] 
                            if elem.lower() in response.lower())
        completeness = found_elements / len(question["expected_elements"])
        
        # Check source usage
        sources_used = (
            bool(retrieved_content["texts"]) or 
            bool(retrieved_content["tables"]) or 
            bool(retrieved_content["images"])
        )
        
        # Check image handling for image-related questions
        image_score = 0.0
        if "image" in question["question"].lower():
            image_score = float(
                bool(retrieved_content["images"]) and 
                "image" in response.lower() and
                any(Path(img.metadata["image_path"]).exists() 
                    for img in retrieved_content["images"])
            )
        
        # Calculate base metrics
        base_metrics = {
            "relevance_score": 0.8 if any(elem.lower() in response.lower() 
                                        for elem in question["expected_elements"]) else 0.2,
            "completeness_score": completeness,
            "accuracy_score": 0.8 if sources_used else 0.3,  # Simplified accuracy based on source usage
            "source_score": float(sources_used),
            "image_handling_score": image_score
        }
        
        # Calculate additional metrics
        bleu_score = self.calculate_bleu_score(
            self.ground_truth.get(question["question"], ""),
            response
        )
        
        rouge_score = self.calculate_rouge_score(
            self.ground_truth.get(question["question"], ""),
            response
        )
        
        context_relevance = self.evaluate_context_relevance(response, retrieved_content)
        factual_accuracy = self.evaluate_factual_accuracy(question["question"], response)
        
        return ExtendedEvaluationMetrics(
            **base_metrics,  # Unpack base metrics
            bleu_score=bleu_score,
            rouge_l_score=rouge_score,
            context_relevance=context_relevance,
            factual_accuracy=factual_accuracy
        )
    
    def run_evaluation(self) -> List[EvaluationResult]:
        """Run evaluation with enhanced metrics"""
        try:
            # Remove the MultimodalRetriever context manager and use the cached retriever directly
            # Evaluate each question
            for question in self.test_questions:
                logger.info(f"Evaluating question: {question['question']}")
                
                # Get response using cached retriever
                retrieved_content = self.retriever.retrieve(question["question"])
                response = self.retriever.generate_response(question["question"], retrieved_content)
                
                # Evaluate response
                metrics = self.evaluate_response(question, response, retrieved_content)
                
                # Store result
                result = EvaluationResult(
                    question=question["question"],
                    response=response,
                    retrieved_content=retrieved_content,
                    metrics=metrics
                )
                self.results.append(result)
                
                # Log results
                logger.info(f"Evaluation metrics for question:")
                logger.info(f"Relevance: {metrics.relevance_score:.2f}")
                logger.info(f"Completeness: {metrics.completeness_score:.2f}")
                logger.info(f"Accuracy: {metrics.accuracy_score:.2f}")
                logger.info(f"Source Usage: {metrics.source_score:.2f}")
                logger.info(f"Image Handling: {metrics.image_handling_score:.2f}")
                logger.info(f"BLEU Score: {metrics.bleu_score:.2f}")
                logger.info(f"ROUGE-L Score: {metrics.rouge_l_score:.2f}")
                logger.info(f"Context Relevance: {metrics.context_relevance:.2f}")
                logger.info(f"Factual Accuracy: {metrics.factual_accuracy:.2f}")
                logger.info("-" * 80)
            
            return self.results
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise

def save_evaluation_results(results: List[EvaluationResult], avg_metrics: Dict, timestamp: str):
    """Save detailed evaluation results to JSON file"""
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Create detailed results structure
    detailed_results = {
        "metadata": {
            "timestamp": timestamp,
            "num_questions": len(results),
            "evaluation_date": datetime.now().isoformat()
        },
        "average_metrics": avg_metrics,
        "questions": [
            {
                "question": r.question,
                "response": r.response,
                "retrieved_content": {
                    "texts": [
                        {
                            "content": doc.page_content,
                            "metadata": {
                                "page_num": doc.metadata.get("page_num"),
                                "type": doc.metadata.get("type"),
                                "summary": doc.metadata.get("summary")
                            }
                        } for doc in r.retrieved_content["texts"]
                    ],
                    "tables": [
                        {
                            "content": doc.page_content,
                            "metadata": {
                                "page_num": doc.metadata.get("page_num"),
                                "type": doc.metadata.get("type"),
                                "title": doc.metadata.get("title")
                            }
                        } for doc in r.retrieved_content["tables"]
                    ],
                    "images": [
                        {
                            "metadata": img.get("metadata", {}),
                            "score": img.get("score", 0.0)
                        } for img in r.retrieved_content["images"]
                    ]
                },
                "metrics": {
                    "relevance": r.metrics.relevance_score,
                    "completeness": r.metrics.completeness_score,
                    "accuracy": r.metrics.accuracy_score,
                    "source_usage": r.metrics.source_score,
                    "image_handling": r.metrics.image_handling_score,
                    "bleu_score": r.metrics.bleu_score,
                    "rouge_l_score": r.metrics.rouge_l_score,
                    "context_relevance": r.metrics.context_relevance,
                    "factual_accuracy": r.metrics.factual_accuracy
                }
            }
            for r in results
        ]
    }
    
    # Save to JSON file
    output_file = output_dir / f"evaluation_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved detailed evaluation results to {output_file}")
    
    # Save summary to separate file
    summary_file = output_dir / f"evaluation_summary_{timestamp}.json"
    summary_results = {
        "metadata": detailed_results["metadata"],
        "average_metrics": avg_metrics,
        "questions_summary": [
            {
                "question": r.question,
                "metrics": {
                    "relevance": r.metrics.relevance_score,
                    "completeness": r.metrics.completeness_score,
                    "accuracy": r.metrics.accuracy_score,
                    "source_usage": r.metrics.source_score,
                    "image_handling": r.metrics.image_handling_score,
                    "bleu_score": r.metrics.bleu_score,
                    "rouge_l_score": r.metrics.rouge_l_score,
                    "context_relevance": r.metrics.context_relevance,
                    "factual_accuracy": r.metrics.factual_accuracy
                }
            }
            for r in results
        ]
    }
    
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved evaluation summary to {summary_file}")

def main():
    """Enhanced main function with improved results saving"""
    try:
        evaluator = RAGEvaluator()
        results = evaluator.run_evaluation()
        
        # Print summary
        print("\nEvaluation Summary:")
        print("=" * 80)
        
        avg_metrics = {
            "relevance": sum(r.metrics.relevance_score for r in results) / len(results),
            "completeness": sum(r.metrics.completeness_score for r in results) / len(results),
            "accuracy": sum(r.metrics.accuracy_score for r in results) / len(results),
            "source_usage": sum(r.metrics.source_score for r in results) / len(results),
            "image_handling": sum(r.metrics.image_handling_score for r in results) / len(results),
            "bleu_score": sum(r.metrics.bleu_score for r in results) / len(results),
            "rouge_l": sum(r.metrics.rouge_l_score for r in results) / len(results),
            "context_relevance": sum(r.metrics.context_relevance for r in results) / len(results),
            "factual_accuracy": sum(r.metrics.factual_accuracy for r in results) / len(results)
        }
        
        print("\nAverage Metrics:")
        for metric, value in avg_metrics.items():
            print(f"{metric.title()}: {value:.2f}")
        
        # Save results using the new function
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_evaluation_results(results, avg_metrics, timestamp)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise
    finally:
        # Clean up any resources if needed
        pass

if __name__ == "__main__":
    main() 