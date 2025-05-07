"""
Script to evaluate CausalRAG pipeline using the evaluation dataset
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from causalrag import CausalRAGPipeline
from causalrag.evaluation.evaluator import CausalEvaluator
from causalrag.generator.llm_interface import LLMInterface

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Load evaluation data
    eval_data_path = project_root / "data" / "evaluation" / "evaluation_dataset.json"
    logger.info(f"Loading evaluation data from {eval_data_path}")
    
    with open(eval_data_path, 'r') as f:
        eval_data = json.load(f)
    
    # Initialize pipeline
    logger.info("Initializing CausalRAG pipeline...")
    pipeline = CausalRAGPipeline(
        model_name="gpt-4",  # Change to your preferred model
        embedding_model="all-MiniLM-L6-v2"
    )
    
    # Setup evaluation directory
    results_dir = project_root / "results" / "evaluation"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Run evaluation
    logger.info("Running evaluation...")
    results = CausalEvaluator.evaluate_pipeline(
        pipeline=pipeline,
        eval_data=eval_data,
        results_dir=results_dir
    )
    
    # Print summary
    logger.info("Evaluation complete! Summary:")
    for metric, score in results.metrics.items():
        logger.info(f"  {metric}: {score:.4f}")
    
    logger.info(f"Detailed results saved to {results_dir}")

if __name__ == "__main__":
    main()