from pydantic import BaseModel, Field
from typing import Literal, Optional, List
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


# =========================================================
# SCHEMAS FOR REPORT GENERATION
# =========================================================

class AttackTimelineStep(BaseModel):
    """Single step in the attack timeline."""
    step: int
    action: str
    timestamp: Optional[str] = None


class NistCsfMapping(BaseModel):
    """NIST Cybersecurity Framework mapping."""
    Identify: str
    Protect: str
    Detect: str
    Respond: str
    Recover: str


class AttackReport(BaseModel):
    """Generated attack analysis report."""
    attack_name: str
    attack_summary: str
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    confidence: str
    nist_csf_mapping: NistCsfMapping
    attack_timeline: List[AttackTimelineStep]
    recommended_actions: List[str]
    indicators_of_compromise: List[str]


class GenerateReportRequest(BaseModel):
    """Request model for report generation."""
    graph_summary: Dict[str, Any] = Field(
        ...,
        description="Summarized attack graph data from EdgePredictor"
    )
    model_name: Optional[str] = Field(
        default=None,
        description="LLM model to use (defaults to configured model)"
    )
    llm_engine: str = Field(
        default="ollama"
    )


class GenerateReportResponse(BaseModel):
    """Response model for report generation."""
    success: bool
    message: Optional[str] = None
    report: Optional[AttackReport] = None
    error: Optional[str] = None
    raw_output: Optional[str] = None


class ReportHealthResponse(BaseModel):
    """Response model for report service health check."""
    status: str
    available: bool
    default_model: str
    message: Optional[str] = None
