#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demonstration of enhanced causal triple extraction and graph building capabilities.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from causalrag.causal_graph.builder import CausalGraphBuilder, CausalTripleExtractor
from causalrag.generator.llm_interface import LLMInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("causal_extraction_demo")

def main():
    """Run the causal extraction demonstration."""
    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not found in environment. "
                     "Set this environment variable before running.")
        logger.info("For demo purposes, we'll continue with rule-based extraction only.")
        llm_interface = None
    else:
        # Initialize LLM interface for extraction
        logger.info("Initializing LLM interface with OpenAI API")
        llm_interface = LLMInterface(model_name="gpt-4")

    # Create extractor with appropriate method
    extractor_method = "hybrid" if llm_interface else "rule"
    logger.info(f"Using {extractor_method} extraction method")
    
    # Initialize graph builder with the specified extraction method
    graph_builder = CausalGraphBuilder(
        normalize_nodes=True,
        confidence_threshold=0.5,
        extractor_method=extractor_method,
        llm_interface=llm_interface
    )
    
    # Sample documents with causal relationships
    documents = [
        """Climate change causes rising sea levels through the melting of polar ice caps and the 
        thermal expansion of water. These rising sea levels lead to coastal flooding, which can 
        damage infrastructure and result in population displacement. Additionally, climate change 
        increases the frequency and intensity of extreme weather events such as hurricanes, 
        which further contribute to flooding and infrastructure damage.""",
        
        """Deforestation reduces carbon capture capacity as fewer trees are available to absorb CO2. 
        This leads to increased atmospheric CO2 levels, which enhances the greenhouse effect. 
        The enhanced greenhouse effect causes global warming, which in turn exacerbates climate change. 
        Deforestation also contributes to soil erosion because tree roots no longer hold soil in place.""",
        
        """Regular exercise improves cardiovascular health by strengthening the heart muscle and 
        improving circulation. Better circulation leads to lower blood pressure and reduced risk of 
        heart disease. Exercise also helps maintain healthy weight, which further reduces strain on the 
        cardiovascular system. Additionally, regular physical activity contributes to reduced stress levels, 
        which indirectly benefits heart health by lowering inflammation.""",
        
        """Higher education levels contribute to economic growth through multiple pathways. 
        Educated workers have higher productivity, which increases overall economic output. 
        Better education also fosters innovation and technological advancement, key drivers of 
        long-term economic growth. Furthermore, higher educational attainment is associated with 
        lower unemployment rates and higher incomes, resulting in increased tax revenue and consumer spending."""
    ]
    
    # Process documents and build graph
    logger.info(f"Processing {len(documents)} documents")
    num_edges = graph_builder.index_documents(documents, show_progress=True)
    logger.info(f"Added {num_edges} edges to the causal graph")
    
    # Get graph statistics
    stats = graph_builder.get_extraction_statistics()
    
    # Print key statistics
    logger.info(f"Graph has {stats['graph_statistics']['nodes']} nodes and "
              f"{stats['graph_statistics']['edges']} edges")
    
    # Show top relationships
    logger.info("Top causal relationships:")
    for i, relation in enumerate(stats['top_relationships'][:5], 1):
        logger.info(f"{i}. {relation['cause']} → {relation['effect']} "
                  f"(confidence: {relation['confidence']:.2f})")
    
    # Generate visualization
    try:
        # Create output directory if it doesn't exist
        output_dir = project_root / "results" / "visualizations"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Generate HTML visualization
        html_path = output_dir / "causal_graph.html"
        logger.info(f"Generating interactive visualization at {html_path}")
        graph_builder.visualize_graph(
            output_path=str(html_path),
            format="html",
            title="Causal Knowledge Graph Demo"
        )
        
        # Also generate JSON representation
        json_path = output_dir / "causal_graph.json"
        logger.info(f"Generating JSON representation at {json_path}")
        graph_builder.visualize_graph(
            output_path=str(json_path),
            format="json"
        )
        
        logger.info("Visualizations generated successfully")
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
    
    # Save the graph for later use
    try:
        graph_path = project_root / "results" / "causal_graph.json"
        logger.info(f"Saving causal graph to {graph_path}")
        graph_builder.save(str(graph_path))
    except Exception as e:
        logger.error(f"Error saving graph: {e}")


if __name__ == "__main__":
    main() 