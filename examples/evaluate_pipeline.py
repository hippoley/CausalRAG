# examples/evaluate_pipeline.py
"""
Example script to evaluate CausalRAG pipeline using Ragas and custom metrics
"""

from causalrag import CausalRAGPipeline
from causalrag.evaluation.evaluator import CausalEvaluator
from causalrag.generator.llm_interface import LLMInterface
import json
import logging
import argparse
from pathlib import Path

def load_evaluation_data(filepath):
    """Load evaluation dataset"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def main(args):
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize CausalRAG pipeline
    logging.info("Initializing pipeline...")
    pipeline = CausalRAGPipeline(
        model_name=args.model_name,
        embedding_model=args.embedding_model
    )
    
    # Load evaluation data
    logging.info(f"Loading evaluation data from {args.eval_data}")
    eval_data = load_evaluation_data(args.eval_data)
    
    # Initialize LLM interface for evaluation
    llm = LLMInterface(
        model=args.eval_model or args.model_name,
        api_key=args.api_key,
        provider=args.provider,
        system_message="You are an expert evaluator assessing the quality of answers to questions."
    )
    
    # Create results directory
    results_dir = Path(args.output_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Define metrics to evaluate
    metrics = [
        "faithfulness", 
        "answer_relevancy", 
        "context_relevancy", 
        "context_recall",
        "causal_consistency", 
        "causal_completeness",
        "answer_quality"
    ]
    
    # Run evaluation
    logging.info("Running evaluation...")
    results = CausalEvaluator.evaluate_pipeline(
        pipeline=pipeline,
        eval_data=eval_data,
        metrics=metrics,
        llm_interface=llm,
        results_dir=results_dir
    )
    
    # Print summary
    logging.info("Evaluation complete! Summary:")
    for metric, score in results.metrics.items():
        logging.info(f"  {metric}: {score:.4f}")
    
    logging.info(f"Detailed results saved to {results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CausalRAG pipeline")
    parser.add_argument("--eval-data", required=True, help="Path to evaluation dataset JSON")
    parser.add_argument("--output-dir", default="./eval_results", help="Directory to save results")
    parser.add_argument("--model-name", default="gpt-4", help="LLM model for pipeline")
    parser.add_argument("--eval-model", default=None, help="LLM model for evaluation (if different)")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--api-key", default=None, help="API key (defaults to env variable)")
    parser.add_argument("--provider", default="openai", help="LLM provider (openai/anthropic/local)")
    
    args = parser.parse_args()
    main(args)