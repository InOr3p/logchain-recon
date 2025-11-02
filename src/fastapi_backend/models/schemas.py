from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from typing import Optional, List, Dict, Any

class LogItem(BaseModel):
    id: str
    rule_id: str
    rule_groups: List[str]
    rule_nist_800_53: Optional[List[str]] = None
    rule_gdpr: Optional[List[str]] = None
    rule_level: int
    rule_firedtimes: int
    agent_ip: str
    data_srcip: Optional[str] = None
    timestamp: str
    unix_timestamp: datetime
    rule_description: str

class Prediction(BaseModel):
    id: str
    prediction_label: str
    prediction_score: float
    is_attack: bool

class ClassifyResponse(BaseModel):
    predictions: List[Prediction]

class LogsRequest(BaseModel):
    logs: List[LogItem]

# =========================================================
# SCHEMAS FOR GRAPH BUILDER AND GNN SERVICE
# =========================================================

class BuildGraphsResponse(BaseModel):
    """
    Response from the /build-graphs endpoint.
    Provides the paths to the generated .npz graph files.
    """
    message: str
    graph_files: List[str] # List of paths to the saved .npz files


class GraphPredictRequest(BaseModel):
    """Request model for graph prediction."""
    graph_path: str = Field(..., description="Path to the .npz graph file in graph_cache")
    logs: List[LogItem] = Field(..., description="List of log entries")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Probability threshold")
    summarize: bool = Field(default=True, description="Return summarized or full graph")


class GraphPredictResponse(BaseModel):
    """Response model for graph prediction."""
    success: bool
    message: str
    graph: Optional[Dict[str, Any]] = None
    threshold: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
