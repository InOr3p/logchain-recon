from fastapi import FastAPI, HTTPException
import requests

app = FastAPI(title="AI Module for Wazuh Server API")

# -----------------------------
# Wazuh API configuration
# -----------------------------
WAZUH_URL = "https://127.0.0.1:55000"
WAZUH_USER = "fastapi-user"
WAZUH_PASS = "Password1234!"

# Cache token for reuse
token_cache = {"token": None}


def get_token():
    """Authenticate with Wazuh API and get JWT token."""
    try:
        response = requests.post(
            f"{WAZUH_URL}/security/user/authenticate",
            auth=(WAZUH_USER, WAZUH_PASS),
            verify=False
        )
        response.raise_for_status()
        return response.json()["data"]["token"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auth failed: {str(e)}")


def get_headers():
    """Return headers with valid JWT token."""
    if not token_cache["token"]:
        token_cache["token"] = get_token()
    return {"Authorization": f"Bearer {token_cache['token']}"}


# -----------------------------
# Endpoints
# -----------------------------

@app.get("/auth")
def auth():
    """Manually refresh JWT token."""
    token_cache["token"] = get_token()
    return {"token": token_cache["token"]}


@app.get("/agents")
def list_agents():
    """Retrieve the list of registered agents."""
    try:
        headers = get_headers()
        response = requests.get(
            f"{WAZUH_URL}/agents",
            headers=headers,
            verify=False
        )
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent/{agent_id}/fim")
def get_agent_fim(agent_id: str, limit: int = 10):
    """Return FIM (File Integrity Monitoring) findings in the specified agent."""
    try:
        headers = get_headers()
        response = requests.get(
            f"{WAZUH_URL}/syscheck/{agent_id}?limit={limit}",
            headers=headers,
            verify=False
        )
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents/{agent_id}/report")
def get_agent_report(agent_id: str, limit: int = 10):
    """
    Retrieve CIS-CAT report for a specific agent.
    CIS-CAT scans the system according to a specific benchmark 
    and generate a report with passed controls, failed controls and not applicable controls
    """
    try:
        headers = get_headers()
        response = requests.get(
            f"{WAZUH_URL}/ciscat/{agent_id}/results?limit={limit}",
            headers=headers,
            verify=False
        )
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents/{agent_id}/hwinfo")
def get_agent_hw_info(agent_id: str):
    """Return the agent's hardware information."""
    try:
        headers = get_headers()
        response = requests.get(
            f"{WAZUH_URL}/syscollector/{agent_id}/hardware",
            headers=headers,
            verify=False
        )
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/agents/{agent_id}/sca")
def get_agent_sca(agent_id: str):
    """Return the agent's sca database."""
    try:
        headers = get_headers()
        response = requests.get(
            f"{WAZUH_URL}/sca/{agent_id}",
            headers=headers,
            verify=False
        )
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents/{agent_id}/sca/{policy_id}")
def get_agent_sca_policy(agent_id: str, policy_id: str):
    """Return all the checks performed for the specified agent according to a specific policy."""
    try:
        headers = get_headers()
        response = requests.get(
            f"{WAZUH_URL}/sca/{agent_id}/checks/{policy_id}",
            headers=headers,
            verify=False
        )
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))