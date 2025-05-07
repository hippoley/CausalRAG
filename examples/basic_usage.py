"""
Basic usage example for CausalRAG.

This example shows how to:
1. Initialize the CausalRAG pipeline
2. Index a collection of documents
3. Query the system
4. Save and load the index
"""

import os
from causalrag import CausalRAGPipeline

# Sample documents with causal relationships
documents = [
    "Climate change causes rising sea levels, which leads to coastal flooding.",
    "Deforestation reduces carbon capture, increasing atmospheric CO2.",
    "Higher CO2 levels accelerate global warming, exacerbating climate change.",
    "Coastal flooding damages infrastructure and causes population displacement.",
    "Climate policies aim to reduce emissions, thereby mitigating climate change effects."
]

def main():
    # Initialize pipeline
    pipeline = CausalRAGPipeline()
    print("Pipeline initialized")

    # Index documents
    print("Indexing documents...")
    result = pipeline.index(documents)
    print(f"Indexed {len(documents)} documents with causal relationships")

    # Save the index (optional)
    save_dir = "causalrag_index"
    os.makedirs(save_dir, exist_ok=True)
    pipeline.save(save_dir)
    print(f"Saved index to {save_dir}")

    # Query examples
    queries = [
        "What are the effects of climate change?",
        "How does deforestation affect the environment?",
        "What can mitigate coastal flooding?"
    ]

    for query in queries:
        print("\n" + "="*80)
        print(f"Query: {query}")
        print("="*80)
        
        result = pipeline.run(query)
        
        print(f"\nAnswer: {result['answer']}")
        
        print("\nSupporting context:")
        for i, ctx in enumerate(result["context"]):
            print(f"[{i+1}] {ctx[:200]}...")
        
        if "causal_paths" in result and result["causal_paths"]:
            print("\nRelevant causal pathways:")
            for path in result["causal_paths"]:
                print(f"- {' → '.join(path)}")

    print("\nDemo completed!")

if __name__ == "__main__":
    main() 