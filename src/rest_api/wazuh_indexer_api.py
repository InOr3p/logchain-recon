from fastapi import FastAPI, HTTPException
import requests
from requests.auth import HTTPBasicAuth

app = FastAPI(title="AI Module for Wazuh Indexer API")

# Wazuh Indexer credentials
WAZUH_INDEXER_URL = "https://127.0.0.1:9200"
WAZUH_INDEXER_USER = "admin"
WAZUH_INDEXER_PASS = "admin"

# -----------------------------
# Endpoints
# -----------------------------

@app.get("/auth")
def auth():
    """Check if Indexer credentials are valid."""
    try:
        resp = requests.get(
            WAZUH_INDEXER_URL,
            auth=HTTPBasicAuth(WAZUH_INDEXER_USER, WAZUH_INDEXER_PASS),
            verify=False
        )
        resp.raise_for_status()
        return {"message": "Authentication successful", "cluster_info": resp.json()}
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Auth failed: {e}")


def send_get_alerts(agent_id: str = None, size: int = 10):
    url = f"{WAZUH_INDEXER_URL}/wazuh-alerts-*/_search"

    if agent_id is None:
        query = {
            "size": size,
            "sort": [{"@timestamp": {"order": "desc"}}]
        }
    else:
        query = {
            "query": {"match": {"agent.id": agent_id}},
            "size": size,
            "sort": [{"@timestamp": {"order": "desc"}}]
        }        

    try:
        resp = requests.get(
            url,
            auth=HTTPBasicAuth(WAZUH_INDEXER_USER, WAZUH_INDEXER_PASS),
            headers={"Content-Type": "application/json"},
            json=query,
            verify=False
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching alerts: {e}")
    
@app.get("/alerts")
def get_alerts(size: int = 10):
    return send_get_alerts(size=size)


@app.get("/alerts/{agent_id}")
def get_agent_alerts(agent_id: str, size: int = 10):
    return send_get_alerts(agent_id=agent_id, size=size)


@app.get("/alerts/suricata/{agent_id}")
def get_suricata_alerts(agent_id: str, size: int = 10):
    url = f"{WAZUH_INDEXER_URL}/wazuh-alerts-*/_search"

    query = {
        "query": {
            "bool": {
                "must": [
                { "match": { "agent.id": agent_id } },
                { "match": { "rule.groups": "suricata" } }
                ]
            }
        },
        "size": size,
        "sort": [{"@timestamp": {"order": "desc"}}]
    }        

    try:
        resp = requests.get(
            url,
            auth=HTTPBasicAuth(WAZUH_INDEXER_USER, WAZUH_INDEXER_PASS),
            headers={"Content-Type": "application/json"},
            json=query,
            verify=False
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching alerts: {e}")