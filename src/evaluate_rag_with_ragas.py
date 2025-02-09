import logging
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import json
import numpy as np

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    FactualCorrectness
)
from ragas import EvaluationDataset
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from retrieval_llama_parse import MultimodalRetriever

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ragas_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RagasEvaluator:
    def __init__(self):
        """Initialize the Ragas evaluator"""
        # Initialize evaluator LLM
        self.evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4"))
        
        # Initialize metrics
        self.metrics = [
            faithfulness.Faithfulness(),
            answer_relevancy.AnswerRelevancy(),
            context_recall.ContextRecall(),
            FactualCorrectness()
        ]
        
        self.test_questions = [
            {
                "question": "How is the scaled dot product attention calculated?",
                "ground_truth": "Scaled dot-product attention is calculated by taking the dot product of queries and keys, dividing by √dk, applying softmax, and then multiplying with values."
            },
            {
                "question": "What is the BLEU score of the model in English to German translation EN-DE?",
                "ground_truth": "The model achieves a BLEU score of 28.4 on the WMT 2014 English-to-German translation task."
            },
            {
                "question": "How long were the base and big models trained?",
                "ground_truth": "The base model was trained for 100,000 steps and the big model for 300,000 steps."
            },
            {
                "question": "Which optimizer was used when training the models?",
                "ground_truth": "The models were trained using the Adam optimizer with β1 = 0.9, β2 = 0.98 and ε = 10^-9."
            },
            {
                "question": "Show me a picture that shows the difference between Scaled Dot-Product Attention and Multi-Head Attention.",
                "ground_truth": "The figure shows Scaled Dot-Product Attention on the left with matrix multiplications and scaling operations, and Multi-Head Attention on the right with parallel attention layers."
            }
        ]

    def prepare_evaluation_data(self, retriever: MultimodalRetriever) -> EvaluationDataset:
        """Prepare data in Ragas format"""
        dataset = []
        
        for question_data in self.test_questions:
            try:
                # Get retrieval results
                retrieved_content = retriever.retrieve(question_data["question"])
                
                # Get response
                response = retriever.generate_response(question_data["question"], retrieved_content)
                
                # Prepare contexts
                contexts = []
                
                # Add text contexts
                for doc in retrieved_content["texts"]:
                    contexts.append(doc.page_content)
                
                # Add table contexts
                for doc in retrieved_content["tables"]:
                    contexts.append(doc.page_content)
                
                # Add image contexts (descriptions and summaries)
                for doc in retrieved_content["images"]:
                    contexts.append(f"Image Description: {doc.page_content}\n"
                                 f"Image Summary: {doc.metadata.get('summary', 'No summary available')}")
                
                # Create evaluation item following Ragas format
                evaluation_item = {
                    "user_input": question_data["question"],
                    "retrieved_contexts": contexts,
                    "response": response,
                    "reference": question_data["ground_truth"]
                }
                
                dataset.append(evaluation_item)
                
            except Exception as e:
                logger.error(f"Error preparing evaluation data for question: {question_data['question']}")
                logger.error(f"Error: {e}")
                continue
        
        return EvaluationDataset.from_list(dataset)

    def run_evaluation(self) -> Dict:
        """Run evaluation using Ragas metrics"""
        try:
            with MultimodalRetriever() as retriever:
                # Initialize the system
                json_path = "llama_parse_output/llama_parse_output_4.json"
                summaries_path = "llama_parse_summary/summaries_5.json"
                retriever.load_and_store_content(json_path, summaries_path)
                
                # Prepare evaluation dataset
                logger.info("Preparing evaluation dataset...")
                evaluation_dataset = self.prepare_evaluation_data(retriever)
                
                # Run evaluation with LLM
                logger.info("Running Ragas evaluation...")
                results = evaluate(
                    dataset=evaluation_dataset,
                    metrics=self.metrics,
                    llm=self.evaluator_llm
                )
                
                return results
                
        except Exception as e:
            logger.error(f"Error during Ragas evaluation: {e}")
            raise

def main():
    """Run the Ragas evaluation and save results"""
    evaluator = RagasEvaluator()
    
    try:
        # Run evaluation
        results = evaluator.run_evaluation()
        
        # Print results
        print("\nRagas Evaluation Results:")
        print("=" * 80)
        for metric, score in results.items():
            print(f"{metric}: {score:.3f}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("evaluation_results")
        output_dir.mkdir(exist_ok=True)
        
        results_dict = {
            "timestamp": timestamp,
            "metrics": {k: float(v) for k, v in results.items()}
        }
        
        with open(output_dir / f"ragas_evaluation_{timestamp}.json", "w") as f:
            json.dump(results_dict, f, indent=2)
            
        logger.info(f"Results saved to evaluation_results/ragas_evaluation_{timestamp}.json")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()

