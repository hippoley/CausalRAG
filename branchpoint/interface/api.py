# interface/api.py
# FastAPI interface for remote access to the CausalRAG pipeline

from fastapi import FastAPI, Body, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import logging
import time
import os
import json
from contextlib import asynccontextmanager

# Import pipeline
from ..pipeline import CausalRAGPipeline
from ..utils.logging import Timer, logger

# Define API models
class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str = Field(..., description="User question or query", min_length=1)
    top_k: int = Field(5, description="Number of documents to retrieve", ge=1, le=20)
    include_sources: bool = Field(False, description="Whether to include source documents in response")
    include_graph: bool = Field(False, description="Whether to include causal graph info in response")

class RerankRequest(BaseModel):
    """Request model for rerank endpoint"""
    query: str = Field(..., description="User question or query")
    documents: List[str] = Field(..., description="List of documents to rerank")
    include_scores: bool = Field(True, description="Whether to include scores in response")

class CausalPathRequest(BaseModel):
    """Request model for causal path endpoint"""
    query: str = Field(..., description="User question or query")
    max_paths: int = Field(3, description="Maximum number of causal paths to return")

class IndexRequest(BaseModel):
    """Request model for document indexing"""
    documents: List[str] = Field(..., description="List of documents to index")
    document_ids: Optional[List[str]] = Field(None, description="Optional document IDs")

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    version: str
    uptime: float

# API lifespan for setup and cleanup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup: Load models and resources when API starts
    logger.info("API starting up - initializing resources")
    # Initialize global pipeline with config from environment
    app.state.pipeline = CausalRAGPipeline(
        model_name=os.getenv("CAUSALRAG_MODEL", "gpt-3.5-turbo"),
        embedding_model=os.getenv("CAUSALRAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        graph_path=os.getenv("CAUSALRAG_GRAPH_PATH", None),
        index_path=os.getenv("CAUSALRAG_INDEX_PATH", None)
    )
    app.state.start_time = time.time()
    yield
    # Cleanup: Release resources when API shuts down
    logger.info("API shutting down - cleaning up resources")

# Create FastAPI app
app = FastAPI(
    title="CausalRAG API",
    description="API for causal graph enhanced retrieval-augmented generation",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CAUSALRAG_CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Exception handler
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Error processing request: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# Get pipeline from app state
def get_pipeline():
    return app.state.pipeline

# Endpoints
@app.get("/health")
def health_check():
    """Check if the API is running and return basic information"""
    return HealthResponse(
        status="healthy",
        version=app.version,
        uptime=time.time() - app.state.start_time
    )

@app.post("/query")
def query_endpoint(payload: QueryRequest, pipeline: CausalRAGPipeline = Depends(get_pipeline)):
    """
    Main endpoint for querying the CausalRAG system
    
    Returns an answer generated from causal-enhanced retrieval
    """
    with Timer("query_processing"):
        try:
            response = pipeline.run(
                payload.query, 
                top_k=payload.top_k
            )
            
            result = {"answer": response["answer"]}
            
            # Include optional fields if requested
            if payload.include_sources and "context" in response:
                result["sources"] = response["context"]
                
            if payload.include_graph and "causal_paths" in response:
                result["causal_paths"] = response["causal_paths"]
                
            return result
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/rerank")
def rerank_endpoint(payload: RerankRequest, pipeline: CausalRAGPipeline = Depends(get_pipeline)):
    """
    Endpoint for just reranking documents using causal path information
    
    Useful for integration with external RAG systems
    """
    try:
        reranked = pipeline.reranker.rerank(
            payload.query, 
            payload.documents, 
            include_scores=payload.include_scores
        )
        
        if payload.include_scores:
            return {
                "reranked_documents": [
                    {"document": doc, "score": score} 
                    for doc, score in reranked
                ]
            }
        else:
            return {"reranked_documents": reranked}
    except Exception as e:
        logger.error(f"Error reranking documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/causal-paths")
def causal_paths_endpoint(
    payload: CausalPathRequest, 
    pipeline: CausalRAGPipeline = Depends(get_pipeline)
):
    """
    Endpoint to retrieve relevant causal paths for a query
    
    Returns causal paths from the graph that are relevant to the query
    """
    try:
        paths = pipeline.graph_retriever.retrieve_paths(
            payload.query, 
            max_paths=payload.max_paths
        )
        
        explanation = pipeline.graph_retriever.get_causal_explanation(payload.query)
        
        return {
            "causal_paths": paths,
            "explanation": explanation
        }
    except Exception as e:
        logger.error(f"Error retrieving causal paths: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index")
def index_documents_endpoint(payload: IndexRequest, pipeline: CausalRAGPipeline = Depends(get_pipeline)):
    """
    Endpoint to index new documents into the system
    
    Adds documents to both vector store and causal graph
    """
    try:
        # Index documents
        doc_count = len(payload.documents)
        with Timer(f"indexing_{doc_count}_documents"):
            # Index in vector store
            vector_count = pipeline.vector_retriever.index_corpus(
                payload.documents, 
                ids=payload.document_ids
            )
            
            # Extract causal information and update graph
            graph_triples = pipeline.graph_builder.index_documents(payload.documents)
            
        return {
            "status": "success",
            "documents_indexed": doc_count,
            "vectors_added": vector_count,
            "causal_triples_extracted": graph_triples
        }
    except Exception as e:
        logger.error(f"Error indexing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
def get_stats(pipeline: CausalRAGPipeline = Depends(get_pipeline)):
    """
    Get statistics about the current state of the system
    
    Returns information about the causal graph and indexed documents
    """
    try:
        # Get graph statistics
        graph = pipeline.graph_builder.get_graph()
        graph_stats = {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "connected_components": sum(1 for _ in nx.weakly_connected_components(graph))
        }
        
        # Get vector store statistics
        vector_stats = {
            "documents": len(pipeline.vector_retriever.passages) if hasattr(pipeline.vector_retriever, "passages") else 0,
        }
        
        return {
            "graph_stats": graph_stats,
            "vector_stats": vector_stats,
            "api_uptime": time.time() - app.state.start_time
        }
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain")
def explain_endpoint(query: str = Body(...), document: str = Body(...), pipeline: CausalRAGPipeline = Depends(get_pipeline)):
    """
    Explain why a document is relevant to a query from a causal perspective
    
    Returns detailed explanation of causal relationships
    """
    try:
        explanation = pipeline.reranker.get_explanation(query, document)
        return {"explanation": explanation}
    except Exception as e:
        logger.error(f"Error generating explanation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn causalrag.interface.api:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)