from fastapi import APIRouter, Request, HTTPException, Query
import pandas as pd
import numpy as np
import os

from fastapi_backend.models.schemas import AttackReport, BuildGraphsResponse, GenerateReportRequest, GenerateReportResponse, GraphPredictRequest, GraphPredictResponse, LogsRequest, ReportHealthResponse


router = APIRouter(prefix="/graphs", tags=["Graphs"])

# --- 2. ADD THE RESPONSE MODEL ---
@router.post("/build", response_model=BuildGraphsResponse)
async def build_graphs_endpoint(
    request: Request, 
    graph_request: LogsRequest 
):
    """
    Receives a list of logs, generates embeddings, builds inference graphs,
    and returns the paths to the saved .npz files.
    """
    
    # Get the singleton instance from app.state
    builder = request.app.state.graph_builder
    
    # Get the embedding model
    emb_model = request.app.state.classifier_service.embedding_model

    if not builder or not emb_model:
        raise HTTPException(status_code=503, detail="A required service is not available.")

    try:
        # Create DataFrame from Pydantic models
        # model_dump(by_alias=True) is crucial for '@timestamp'
        log_dicts = [log.model_dump(by_alias=True) for log in graph_request.logs]
        
        if not log_dicts:
            return BuildGraphsResponse(message="No logs provided, 0 graphs built.", graph_files=[])
            
        df = pd.DataFrame(log_dicts)

        # Create embeddings
        descriptions = df['rule_description'].tolist()
        vectors = emb_model.encode(descriptions, convert_to_numpy=True)
        
        # Build graphs
        graph_paths = builder.build_graphs_from_dataframe(df, vectors)
        
        return BuildGraphsResponse(
            message=f"Successfully built {len(graph_paths)} graphs.",
            graph_files=graph_paths
        )

    except Exception as e:
        print(f"Error building graphs: {e}") # Simple print for debugging
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data")
async def get_graph_data(path: str = Query(..., description="Path to the .npz graph file")):
    """
    Reads a .npz graph file and returns its data as JSON.
    """
    try:
        # Security check: ensure the path is within your graphs directory
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Graph file not found")
        
        # Load the .npz file
        data = np.load(path, allow_pickle=True)
        
        # Extract filename and metadata
        filename = os.path.basename(path)
        match = filename.replace('.npz', '').split('_')
        
        # Parse agent_ip and window_num from filename
        # Format: inference_{agent_ip}_win{num}
        agent_ip = match[1] if len(match) > 1 else "unknown"
        window_num = match[2].replace('win', '') if len(match) > 2 else "0"
        
        # Convert numpy arrays to lists for JSON serialization
        response = {
            "filename": filename,
            "agent_ip": agent_ip,
            "window_num": window_num,
            "num_nodes": int(data['node_feats'].shape[0]),
            "num_edges": int(data['edge_index'].shape[1]),
            "node_feats": data['node_feats'].tolist(),
            "edge_index": data['edge_index'].tolist(),
            "log_ids": data['log_ids'].tolist()
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading graph file: {str(e)}")
    


# ==================== Endpoint ====================

@router.post("/predict", response_model=GraphPredictResponse)
async def predict_graph(
    request_data: GraphPredictRequest,
    request: Request
):
    """
    Predict attack edges in a graph using provided logs.
    
    This endpoint:
    1. Loads a graph from the specified path (relative to graph_cache)
    2. Uses the provided logs as metadata
    3. Runs edge prediction using the EdgePredictorService
    4. Returns attack graph with nodes and edges
    
    Args:
        request_data: GraphPredictRequest containing graph path, logs, and parameters
        request: FastAPI Request object to access app.state services
        
    Returns:
        GraphPredictResponse with prediction results
        
    Example:
        POST /graphs/predict
        {
            "graph_path": "graph_chunk_0.npz",
            "logs": [
                {
                    "id": "log_001",
                    "timestamp": "2024-01-15T10:30:00",
                    "rule_description": "SSH authentication failed",
                    "rule_level": 5,
                    ...
                }
            ],
            "threshold": 0.9,
            "summarize": true
        }
    """
    try:
        # Get the EdgePredictorService from app.state
        edge_predictor_service = request.app.state.edge_predictor_service
        
        graph_full_path = os.path.join(request_data.graph_path)
        
        # Load graph data
        if not os.path.exists(graph_full_path):
            raise HTTPException(
                status_code=404,
                detail=f"Graph file not found: {request_data.graph_path}"
            )
        
        loaded_data = np.load(graph_full_path, allow_pickle=True)
        
        # Validate graph structure
        if 'node_feats' not in loaded_data or 'edge_index' not in loaded_data or 'log_ids' not in loaded_data:
            raise HTTPException(
                status_code=400,
                detail="Graph file must contain 'node_feats', 'edge_index', and 'log_ids'"
            )
        
        node_features = loaded_data['node_feats']
        edge_index = loaded_data['edge_index']
        log_ids = loaded_data['log_ids']
        
        # Convert logs to DataFrame
        if not request_data.logs:
            raise HTTPException(
                status_code=400,
                detail="Logs list cannot be empty"
            )
        
        logs_df = pd.DataFrame([log.dict() for log in request_data.logs])
        
        # Ensure id column exists and is string type
        if 'id' not in logs_df.columns:
            raise HTTPException(
                status_code=400,
                detail="All logs must have an 'id' field"
            )
        
        logs_df['id'] = logs_df['id'].astype(str)
        logs_df = logs_df.set_index('id')
        
        # Convert timestamp if present
        if '@timestamp' in logs_df.columns:
            logs_df['@timestamp'] = pd.to_datetime(logs_df['@timestamp'], errors='coerce')
        elif 'timestamp' in logs_df.columns:
            logs_df['@timestamp'] = pd.to_datetime(logs_df['timestamp'], errors='coerce')
        
        # Run prediction
        result = edge_predictor_service.predict(
            node_features=node_features,
            edge_index=edge_index,
            log_ids=log_ids,
            logs_df=logs_df,
            threshold=request_data.threshold,
            summarize=request_data.summarize
        )
        
        # Add metadata
        result['metadata'] = {
            'graph_path': request_data.graph_path,
            'num_nodes': len(node_features),
            'num_edges': edge_index.shape[1],
            'num_features': node_features.shape[1],
            'num_logs_provided': len(request_data.logs)
        }
        
        return GraphPredictResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )
    

@router.post("/generate-report", response_model=GenerateReportResponse)
async def generate_report(
    request_data: GenerateReportRequest,
    request: Request
):
    """
    Generate an attack analysis report from an attack graph summary.
    
    This endpoint uses a local LLM (Ollama) to analyze the attack graph
    and generate a structured report with:
    - Attack identification and classification
    - NIST CSF mapping
    - Attack timeline
    - Recommended actions
    - Indicators of compromise
    
    Args:
        request_data: GenerateReportRequest with graph summary and optional model
        request: FastAPI Request object to access app.state services
        
    Returns:
        GenerateReportResponse with the generated report or error information
        
    Example:
        POST /graphs/generate-report
        {
            "graph_summary": {
                "total_nodes": 15,
                "total_edges": 42,
                "sample_nodes": [...],
                "sample_edges": [...]
            },
            "model_name": "llama3.2"
        }
    """
    try:
        # Get the ReportGenerationService from app.state
        report_service = request.app.state.report_service
        
        if report_service is None:
            raise HTTPException(
                status_code=503,
                detail="Report generation service is not available. Ensure Ollama is running."
            )
        
        # Validate graph summary
        if not request_data.graph_summary:
            raise HTTPException(
                status_code=400,
                detail="Graph summary cannot be empty"
            )
        
        # Check if Ollama is available
        if request_data.llm_engine == "ollama" and not report_service.check_ollama_health():
            raise HTTPException(
                status_code=503,
                detail="Ollama service is not available. Please start Ollama and try again."
            )
        
        elif request_data.llm_engine == "groq" and not report_service.check_groq_health():
            raise HTTPException(
                status_code=503,
                detail="Ollama service is not available. Please start Ollama and try again."
            )
                
        # Generate the report
        result = report_service.generate_report(
            graph_summary=request_data.graph_summary,
            model_name=request_data.model_name,
            llm_engine=request_data.llm_engine
        )
        
        if result.get("success"):
            return GenerateReportResponse(
                success=True,
                message="Report generated successfully",
                report=AttackReport(**result["report"])
            )
        else:
            # Return error but don't raise HTTPException
            return GenerateReportResponse(
                success=False,
                error=result.get("error", "Unknown error"),
                raw_output=result.get("raw_output")
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Report generation error: {str(e)}"
        )


@router.get("/report/health", response_model=ReportHealthResponse)
async def report_health(request: Request, llm_engine: str):
    """
    Check if the report generation service is healthy.
    
    Args:
        request: FastAPI Request object to access app.state services
        
    Returns:
        Health status information
    """
    try:
        report_service = request.app.state.report_service
        
        if report_service is None:
            return {
                "status": "unavailable",
                "available": False,
                "message": "Report service not initialized"
            }
        
        engine_name = "Ollama"
        llm_available = False
        model_name = report_service.default_ollama_model

        if llm_engine.lower() == 'groq':
            engine_name = "Groq"
            llm_available = await report_service.check_groq_health()
            model_name = report_service.default_groq_model # Use Groq's model name
        else:
            engine_name = "Ollama"
            llm_available = await report_service.check_ollama_health()
            model_name = report_service.default_ollama_model # Use Ollama's model name
        
        return ReportHealthResponse(
            status = "healthy" if llm_available else "degraded",
            available = llm_available,
            default_model = model_name,
            message = f"{engine_name} is running" if llm_available else f"{engine_name} is not accessible"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Health check error: {str(e)}"
        )