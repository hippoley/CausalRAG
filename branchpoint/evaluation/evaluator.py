# evaluation/evaluator.py
# Comprehensive evaluation module for CausalRAG using Ragas framework with causal extensions

import os
import json
import re
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass

# Ragas imports
try:
    from ragas.metrics import (
        faithfulness, 
        answer_relevancy,
        context_relevancy,
        context_recall
    )
    from ragas.metrics.critique import harmfulness
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logging.warning("Ragas package not installed. Install with: pip install ragas")

# Import LLM interface for critique-based evaluation
try:
    from ..generator.llm_interface import LLMInterface
except ImportError:
    try:
        # Alternative import path
        from causalrag.generator.llm_interface import LLMInterface
    except ImportError:
        # Handle case where module is used standalone
        LLMInterface = None

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    metrics: Dict[str, float]
    detailed_scores: Dict[str, List[float]]
    error_analysis: Optional[Dict[str, Any]] = None
    raw_evaluations: Optional[Dict[str, Any]] = None

class CausalEvaluator:
    """Evaluates RAG pipeline with focus on causal reasoning capabilities"""
    
    def __init__(self, 
                llm_interface=None, 
                metrics: List[str] = None,
                use_ragas: bool = True,
                results_dir: Optional[str] = None):
        """
        Initialize evaluator with specified metrics and LLM interface
        
        Args:
            llm_interface: LLM interface for critique-based evaluation
            metrics: List of metrics to compute (defaults to all available)
            use_ragas: Whether to use Ragas framework (if available)
            results_dir: Directory to save evaluation results
        """
        self.llm_interface = llm_interface
        self.use_ragas = use_ragas and RAGAS_AVAILABLE
        
        # Setup metrics
        self.default_metrics = [
            "faithfulness", 
            "answer_relevancy", 
            "context_relevancy", 
            "context_recall",
            "causal_consistency", 
            "causal_completeness"
        ]
        
        self.metrics = metrics or self.default_metrics
        
        # Setup results directory
        if results_dir:
            self.results_dir = Path(results_dir)
            os.makedirs(self.results_dir, exist_ok=True)
        else:
            self.results_dir = None
    
    def evaluate(self, 
                questions: List[str],
                answers: List[str],
                contexts: List[List[str]],
                causal_paths: List[List[List[str]]] = None,
                ground_truths: Optional[List[str]] = None) -> EvaluationResult:
        """
        Evaluate the quality of answers based on multiple metrics
        
        Args:
            questions: List of questions
            answers: List of generated answers
            contexts: List of context passages used for each answer
            causal_paths: List of causal paths used for each answer
            ground_truths: Optional list of ground truth answers
            
        Returns:
            EvaluationResult containing all computed metrics
        """
        # Initialize results containers
        all_metrics = {}
        detailed_scores = {}
        error_analysis = {}
        raw_results = {}
        
        # Validate inputs
        if len(questions) != len(answers) or len(questions) != len(contexts):
            raise ValueError("Number of questions, answers, and contexts must match")
        
        if causal_paths is not None and len(questions) != len(causal_paths):
            raise ValueError("Number of questions and causal paths must match")
        
        # Run Ragas evaluation if available and requested
        if self.use_ragas:
            ragas_results = self._run_ragas_evaluation(
                questions, answers, contexts, ground_truths
            )
            if ragas_results:
                all_metrics.update(ragas_results["metrics"])
                detailed_scores.update(ragas_results["detailed"])
                raw_results["ragas"] = ragas_results["raw"]
        
        # Run causal-specific evaluations if causal paths are provided
        if causal_paths:
            causal_results = self._evaluate_causal_reasoning(
                questions, answers, contexts, causal_paths, ground_truths
            )
            if causal_results:
                all_metrics.update(causal_results["metrics"])
                detailed_scores.update(causal_results["detailed"])
                error_analysis.update(causal_results.get("errors", {}))
                raw_results["causal"] = causal_results["raw"]
        
        # Run LLM critique evaluations if LLM interface is available
        if self.llm_interface:
            llm_results = self._run_llm_evaluations(
                questions, answers, contexts, causal_paths, ground_truths
            )
            if llm_results:
                all_metrics.update(llm_results["metrics"])
                detailed_scores.update(llm_results["detailed"])
                error_analysis.update(llm_results.get("errors", {}))
                raw_results["llm_critique"] = llm_results["raw"]
        
        # Create final result
        result = EvaluationResult(
            metrics=all_metrics,
            detailed_scores=detailed_scores,
            error_analysis=error_analysis,
            raw_evaluations=raw_results
        )
        
        # Save results if directory is specified
        if self.results_dir:
            self._save_results(result)
        
        return result
    
    def _run_ragas_evaluation(self,
                             questions: List[str],
                             answers: List[str],
                             contexts: List[List[str]],
                             ground_truths: Optional[List[str]]) -> Dict[str, Any]:
        """Run evaluation using Ragas framework"""
        if not self.use_ragas:
            return {}
        
        try:
            # Prepare data in Ragas format
            flattened_contexts = [" ".join(ctx) for ctx in contexts]
            
            # Define metrics to use
            ragas_metrics = []
            
            if "faithfulness" in self.metrics:
                ragas_metrics.append(faithfulness)
            
            if "answer_relevancy" in self.metrics:
                ragas_metrics.append(answer_relevancy)
            
            if "context_relevancy" in self.metrics:
                ragas_metrics.append(context_relevancy)
                
            if "context_recall" in self.metrics and ground_truths:
                ragas_metrics.append(context_recall)
            
            if not ragas_metrics:
                return {}
            
            # Create dataset dictionary
            dataset_dict = {
                "question": questions,
                "answer": answers,
                "contexts": [[ctx] for ctx in flattened_contexts],
            }
            
            if ground_truths:
                dataset_dict["ground_truth"] = ground_truths
            
            # Run evaluation
            result = evaluate(
                dataset_dict,
                metrics=ragas_metrics
            )
            
            # Extract results
            metrics_dict = {}
            detailed_dict = {}
            
            # Process result DataFrame
            for column in result.columns:
                if column in ['question', 'answer', 'contexts', 'ground_truth']:
                    continue
                
                metrics_dict[column] = float(result[column].mean())
                detailed_dict[column] = result[column].tolist()
            
            return {
                "metrics": metrics_dict,
                "detailed": detailed_dict,
                "raw": result.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error in Ragas evaluation: {e}")
            return {}
    
    def _evaluate_causal_reasoning(self,
                                  questions: List[str],
                                  answers: List[str],
                                  contexts: List[List[str]],
                                  causal_paths: List[List[List[str]]],
                                  ground_truths: Optional[List[str]]) -> Dict[str, Any]:
        """Evaluate causal reasoning aspects using specialized metrics"""
        results = {
            "metrics": {},
            "detailed": {},
            "errors": {},
            "raw": {}
        }
        
        # Skip if no LLM interface for evaluating causal aspects
        if not self.llm_interface:
            return results
        
        try:
            # Measure causal consistency - do answers respect the causal relationships?
            if "causal_consistency" in self.metrics:
                consistency_scores = []
                
                for i, (question, answer, paths) in enumerate(zip(questions, answers, causal_paths)):
                    # Skip if no causal paths
                    if not paths:
                        consistency_scores.append(1.0)  # No paths to be inconsistent with
                        continue
                    
                    # Format causal paths for prompt
                    paths_text = "\n".join([" → ".join(path) for path in paths])
                    
                    # Create evaluation prompt
                    prompt = f"""Evaluate if the answer respects the causal relationships provided.
                    
Question: {question}

Causal relationships that should be respected:
{paths_text}

Answer to evaluate:
{answer}

On a scale of 0-10, how well does the answer respect these causal relationships?
- Score 0: Answer directly contradicts the causal relationships
- Score 5: Answer is neutral or doesn't address the causal relationships
- Score 10: Answer perfectly aligns with and explains the causal relationships

Provide your rating as a number from 0-10, followed by a brief explanation.
Rating:"""
                    
                    # Get rating from LLM
                    response = self.llm_interface.generate(prompt, temperature=0.1)
                    
                    # Extract score
                    try:
                        # Try to find a number in the first line
                        first_line = response.strip().split('\n')[0]
                        score_match = re.search(r'(\d+(\.\d+)?)', first_line)
                        if score_match:
                            score = float(score_match.group(1))
                            # Normalize to 0-1 scale
                            score = min(max(score / 10, 0.0), 1.0)
                        else:
                            logger.warning(f"Could not extract score from: {first_line}")
                            score = 0.5  # Default to neutral
                    except Exception as e:
                        logger.error(f"Error extracting consistency score: {e}")
                        score = 0.5
                    
                    consistency_scores.append(score)
                
                # Add to results
                results["metrics"]["causal_consistency"] = np.mean(consistency_scores)
                results["detailed"]["causal_consistency"] = consistency_scores
                results["raw"]["causal_consistency_responses"] = []  # Could store LLM responses
            
            # Measure causal completeness - does the answer cover important causal factors?
            if "causal_completeness" in self.metrics:
                completeness_scores = []
                
                for i, (question, answer, paths) in enumerate(zip(questions, answers, causal_paths)):
                    # Skip if no causal paths
                    if not paths:
                        completeness_scores.append(1.0)  # No paths to miss
                        continue
                    
                    # Extract all unique causal factors from paths
                    all_factors = set()
                    for path in paths:
                        all_factors.update(path)
                    
                    # Format factors for prompt
                    factors_text = "\n".join([f"- {factor}" for factor in all_factors])
                    
                    # Create evaluation prompt
                    prompt = f"""Evaluate if the answer addresses all important causal factors.
                    
Question: {question}

Important causal factors that should be addressed:
{factors_text}

Answer to evaluate:
{answer}

On a scale of 0-10, how completely does the answer address these important causal factors?
- Score 0: Answer misses all important causal factors
- Score 5: Answer addresses some factors but misses several important ones
- Score 10: Answer comprehensively addresses all important causal factors

Provide your rating as a number from 0-10, followed by a brief explanation of which factors were missed (if any).
Rating:"""
                    
                    # Get rating from LLM
                    response = self.llm_interface.generate(prompt, temperature=0.1)
                    
                    # Extract score using similar method as above
                    try:
                        first_line = response.strip().split('\n')[0]
                        score_match = re.search(r'(\d+(\.\d+)?)', first_line)
                        if score_match:
                            score = float(score_match.group(1))
                            # Normalize to 0-1 scale
                            score = min(max(score / 10, 0.0), 1.0)
                        else:
                            score = 0.5
                    except Exception as e:
                        logger.error(f"Error extracting completeness score: {e}")
                        score = 0.5
                    
                    completeness_scores.append(score)
                
                # Add to results
                results["metrics"]["causal_completeness"] = np.mean(completeness_scores)
                results["detailed"]["causal_completeness"] = completeness_scores
            
            return results
            
        except Exception as e:
            logger.error(f"Error in causal evaluation: {e}")
            results["errors"]["causal_evaluation"] = str(e)
            return results
    
    def _run_llm_evaluations(self,
                            questions: List[str],
                            answers: List[str],
                            contexts: List[List[str]],
                            causal_paths: Optional[List[List[List[str]]]],
                            ground_truths: Optional[List[str]]) -> Dict[str, Any]:
        """Run general LLM-based evaluations"""
        if not self.llm_interface:
            return {}
        
        results = {
            "metrics": {},
            "detailed": {},
            "errors": {},
            "raw": {}
        }
        
        try:
            # Overall answer quality evaluation
            if "answer_quality" in self.metrics:
                quality_scores = []
                
                for i, (question, answer, context) in enumerate(zip(questions, answers, contexts)):
                    # Format context for prompt
                    context_text = "\n".join([f"[{j+1}] {c}" for j, c in enumerate(context)])
                    
                    # Add ground truth if available
                    gt_text = ""
                    if ground_truths and i < len(ground_truths):
                        gt_text = f"\nGround truth answer:\n{ground_truths[i]}"
                    
                    # Create evaluation prompt
                    prompt = f"""Evaluate the quality of this answer based on the question and provided context.
                    
Question: {question}

Context:
{context_text}{gt_text}

Answer to evaluate:
{answer}

Rate the answer on a scale of 0-10 based on:
1. Accuracy - Does it correctly use information from the context?
2. Completeness - Does it address all aspects of the question?
3. Conciseness - Is it appropriately detailed without unnecessary information?
4. Coherence - Is it well-structured and logical?

Provide your overall rating as a number from 0-10.
Overall rating:"""
                    
                    # Get rating from LLM
                    response = self.llm_interface.generate(prompt, temperature=0.1)
                    
                    # Extract score
                    try:
                        first_line = response.strip().split('\n')[0]
                        score_match = re.search(r'(\d+(\.\d+)?)', first_line)
                        if score_match:
                            score = float(score_match.group(1))
                            # Normalize to 0-1 scale
                            score = min(max(score / 10, 0.0), 1.0)
                        else:
                            score = 0.7  # Default to reasonable score
                    except Exception as e:
                        logger.error(f"Error extracting quality score: {e}")
                        score = 0.7
                    
                    quality_scores.append(score)
                
                # Add to results
                results["metrics"]["answer_quality"] = np.mean(quality_scores)
                results["detailed"]["answer_quality"] = quality_scores
            
            return results
            
        except Exception as e:
            logger.error(f"Error in LLM evaluation: {e}")
            results["errors"]["llm_evaluation"] = str(e)
            return results
    
    def _save_results(self, result: EvaluationResult) -> None:
        """Save evaluation results to disk"""
        if not self.results_dir:
            return
        
        try:
            # Create timestamp for filename
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save metrics summary
            metrics_file = self.results_dir / f"metrics_summary_{timestamp}.json"
            with open(metrics_file, 'w') as f:
                json.dump(result.metrics, f, indent=2)
            
            # Save detailed scores
            detailed_file = self.results_dir / f"detailed_scores_{timestamp}.json"
            with open(detailed_file, 'w') as f:
                json.dump(result.detailed_scores, f, indent=2)
            
            # Save error analysis if available
            if result.error_analysis:
                errors_file = self.results_dir / f"error_analysis_{timestamp}.json"
                with open(errors_file, 'w') as f:
                    json.dump(result.error_analysis, f, indent=2)
            
            # Create summary report as markdown
            report_file = self.results_dir / f"evaluation_report_{timestamp}.md"
            with open(report_file, 'w') as f:
                f.write(f"# CausalRAG Evaluation Report\n\n")
                f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("## Metrics Summary\n\n")
                for metric, score in result.metrics.items():
                    f.write(f"- **{metric}:** {score:.4f}\n")
                
                f.write("\n## Metric Details\n\n")
                for metric, scores in result.detailed_scores.items():
                    avg = np.mean(scores)
                    med = np.median(scores)
                    min_val = np.min(scores)
                    max_val = np.max(scores)
                    
                    f.write(f"### {metric}\n")
                    f.write(f"- **Average:** {avg:.4f}\n")
                    f.write(f"- **Median:** {med:.4f}\n")
                    f.write(f"- **Min:** {min_val:.4f}\n")
                    f.write(f"- **Max:** {max_val:.4f}\n\n")
            
            logger.info(f"Evaluation results saved to {self.results_dir}")
            
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")
    
    @classmethod
    def evaluate_pipeline(cls, 
                         pipeline, 
                         eval_data: List[Dict[str, str]],
                         metrics: List[str] = None,
                         llm_interface = None,
                         results_dir: Optional[str] = None) -> EvaluationResult:
        """
        Convenience method to evaluate a complete CausalRAG pipeline
        
        Args:
            pipeline: CausalRAG pipeline to evaluate
            eval_data: List of evaluation examples (each with 'question' and optionally 'ground_truth')
            metrics: List of metrics to compute
            llm_interface: LLM interface for critique-based evaluation
            results_dir: Directory to save evaluation results
            
        Returns:
            EvaluationResult containing all computed metrics
        """
        # Initialize evaluator
        evaluator = cls(
            llm_interface=llm_interface or pipeline.llm,
            metrics=metrics,
            results_dir=results_dir
        )
        
        # Run pipeline on all questions
        questions = [item["question"] for item in eval_data]
        ground_truths = [item.get("ground_truth") for item in eval_data if "ground_truth" in item]
        
        # If ground_truths is an empty list, set it to None
        if not ground_truths:
            ground_truths = None
        
        # Process all questions
        answers = []
        contexts = []
        causal_paths = []
        
        for question in questions:
            result = pipeline.run(question)
            answers.append(result["answer"])
            contexts.append(result.get("context", []))
            causal_paths.append(result.get("causal_paths", []))
        
        # Run evaluation
        return evaluator.evaluate(
            questions=questions,
            answers=answers,
            contexts=contexts,
            causal_paths=causal_paths,
            ground_truths=ground_truths
        )