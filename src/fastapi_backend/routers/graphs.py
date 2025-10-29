from fastapi import APIRouter, Request, HTTPException, Query
import pandas as pd
import numpy as np
import os

from fastapi_backend.models.schemas import BuildGraphsResponse, LogsRequest


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