import requests
from requests.auth import HTTPBasicAuth
from fastapi import HTTPException
import pandas as pd
from fastapi_backend.models.schemas import LogItem
from fastapi_backend.routers.users import INDEXER_PASSWORD, INDEXER_URL, INDEXER_USER, WAZUH_API_URL, is_auth_on_indexer, get_headers, indexer_auth

def get_wazuh_agents(limit=2000):
    """Retrieve the list of registered agents."""
    try:
        params = {"limit": limit, "sort": "id"}
        headers = get_headers()
        response = requests.get(
            f"{WAZUH_API_URL}/agents",
            headers=headers,
            params=params,
            verify=False
        )
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_agent_logs(agent_id, limit=50):
    """Retrieve latest logs for a specific agent, flatten them, 
    and fill missing agent_ip with the known one from Wazuh API."""
    if not is_auth_on_indexer:
        indexer_auth()

    agents_list = get_wazuh_agents()
    agent_data = next((agent for agent in agents_list["data"]["affected_items"] if agent["id"] == agent_id), None)
    if not agent_data:
        agent_ip = "unknown"
    else:
        agent_ip = agent_data.get("ip") or agent_data.get("registerIP")

    # Query the Indexer for logs
    query = {
        "size": limit,
        "sort": [{"@timestamp": {"order": "desc"}}],
        "query": {"match": {"agent.id": agent_id}}
    }

    response = requests.get(
        f"{INDEXER_URL}/wazuh-alerts-*/_search",
        auth=HTTPBasicAuth(INDEXER_USER, INDEXER_PASSWORD),
        json=query,
        verify=False
    )

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    hits = response.json().get("hits", {}).get("hits", [])
    if not hits:
        return []

    # Flatten the data
    raw_logs = [hit["_source"] for hit in hits]
    df = pd.json_normalize(raw_logs, sep="_").where(pd.notnull(pd.json_normalize(raw_logs, sep="_")), None)

    # Generate timestamps
    if "@timestamp" in df.columns:
        df["timestamp"] = df["@timestamp"]
        df["unix_timestamp"] = pd.to_datetime(df["@timestamp"], errors="coerce")

    # Fill missing agent_ip with known one
    if "agent_ip" not in df.columns:
        df["agent_ip"] = agent_ip
    else:
        df["agent_ip"] = df["agent_ip"].fillna(agent_ip)

    # Filter columns
    keep_cols = [
        "id",
        "rule_id",
        "rule_groups",
        "rule_nist_800_53",
        "rule_gdpr",
        "rule_level",
        "rule_firedtimes",
        "agent_ip",
        "data_srcip",
        "timestamp",
        "unix_timestamp",
        "rule_description",
        "full_log"
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

    # Sort logs chronologically
    df = df.sort_values(by="unix_timestamp", ascending=True, na_position="last")

    # Map to Pydantic model
    logs = [LogItem(**row) for row in df.to_dict(orient="records")]
    return logs


