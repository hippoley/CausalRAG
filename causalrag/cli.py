#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Command-line interface for CausalRAG.
"""

import argparse
import logging
import sys
import os
from pathlib import Path

from causalrag import CausalRAGPipeline, __version__
from causalrag.utils.logging import setup_logging

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CausalRAG: Causal Graph Enhanced Retrieval-Augmented Generation"
    )
    
    parser.add_argument("--version", action="store_true", help="Show version and exit")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index documents")
    index_parser.add_argument("--input", "-i", required=True, help="Input directory or file")
    index_parser.add_argument("--output", "-o", required=True, help="Output directory for index")
    index_parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model name")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the system")
    query_parser.add_argument("--index", "-i", required=True, help="Index directory")
    query_parser.add_argument("--query", "-q", required=True, help="Query text")
    query_parser.add_argument("--model", default="gpt-4", help="LLM model name")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of documents to retrieve")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--index", "-i", help="Index directory")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    
    return parser.parse_args()

def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    # Setup logging
    setup_logging(level=logging.INFO)
    
    if args.version:
        print(f"CausalRAG version {__version__}")
        return 0
    
    if args.command == "index":
        from causalrag.utils.io import list_files, read_text_file
        
        # Load documents
        input_path = Path(args.input)
        if input_path.is_dir():
            file_paths = list_files(input_path, pattern="*.txt", recursive=True)
            documents = [read_text_file(fp) for fp in file_paths if read_text_file(fp)]
            logger.info(f"Loaded {len(documents)} documents from {input_path}")
        else:
            document = read_text_file(str(input_path))
            documents = [document] if document else []
            logger.info(f"Loaded document from {input_path}")
        
        if not documents:
            logger.error("No documents found to index")
            return 1
        
        # Create and run pipeline
        pipeline = CausalRAGPipeline(embedding_model=args.model)
        result = pipeline.index(documents, save_path=args.output)
        
        logger.info(f"Indexing complete: {len(documents)} documents indexed")
        return 0
    
    elif args.command == "query":
        from causalrag import create_pipeline
        
        # Create pipeline with pre-built index
        pipeline = create_pipeline(
            model_name=args.model,
            graph_path=os.path.join(args.index, "causal_graph.json"),
            index_path=args.index
        )
        
        # Execute query
        result = pipeline.run(args.query, top_k=args.top_k)
        
        # Print result
        print("\n" + "="*80)
        print(f"Query: {args.query}")
        print("="*80)
        print(f"\nAnswer: {result['answer']}")
        print("\nSupporting Context:")
        for i, ctx in enumerate(result["context"]):
            print(f"\n[{i+1}] {ctx[:200]}...")
        
        if "causal_paths" in result and result["causal_paths"]:
            print("\nRelevant Causal Pathways:")
            for i, path in enumerate(result["causal_paths"]):
                print(f"[{i+1}] {' → '.join(path)}")
        
        return 0
    
    elif args.command == "serve":
        import uvicorn
        
        if args.index:
            # Set environment variables for API
            os.environ["CAUSALRAG_GRAPH_PATH"] = os.path.join(args.index, "causal_graph.json")
            os.environ["CAUSALRAG_INDEX_PATH"] = args.index
        
        logger.info(f"Starting API server on {args.host}:{args.port}")
        uvicorn.run(
            "causalrag.interface.api:app",
            host=args.host,
            port=args.port,
            log_level="info"
        )
        return 0
    
    else:
        # No command specified, show help
        print("No command specified. Use --help for usage information.")
        return 1

if __name__ == "__main__":
    sys.exit(main())