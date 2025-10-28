from fastapi import APIRouter, HTTPException, Query
from fastapi_backend.services.logs_service import get_agent_logs, get_wazuh_agents

router = APIRouter(prefix="/agents", tags=["Wazuh Logs"])

@router.get("/")
def get_agents(limit: int = Query(2000, le=5000)):
    """
    Retrieve all registered agents from Wazuh.
    """
    try:
        agents = get_wazuh_agents(limit)
        if not agents:
            raise HTTPException(status_code=404, detail="No agents found.")
        return {"count": len(agents), "agents": agents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.get("/{agent_id}/logs/latest")
def get_latest_agent_logs(agent_id: str, limit: int = Query(50, le=200)):
    """
    Retrieve the latest logs for a specific Wazuh agent directly from the Wazuh API.
    """
    try:
        logs = get_agent_logs(agent_id, limit)
        if not logs:
            raise HTTPException(status_code=404, detail=f"No logs found for agent {agent_id}")
        return {
            "agent_id": agent_id,
            "count": len(logs),
            "logs": logs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    