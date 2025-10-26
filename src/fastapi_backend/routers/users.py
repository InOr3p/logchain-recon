from fastapi import APIRouter, HTTPException
from http.client import HTTPException
from requests.auth import HTTPBasicAuth
import os
from dotenv import load_dotenv
import requests

router = APIRouter(prefix="/auth", tags=["Wazuh users"])

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path)

WAZUH_API_URL = os.getenv("WAZUH_API_URL", "https://localhost:55000")
WAZUH_USER = os.getenv("WAZUH_USER")
WAZUH_PASSWORD = os.getenv("WAZUH_PASSWORD")

INDEXER_URL = os.getenv("INDEXER_URL", "https://localhost:9200")
INDEXER_USER = os.getenv("INDEXER_USER")
INDEXER_PASSWORD = os.getenv("INDEXER_PASSWORD")

# Cache token for reuse
token_cache = {"token": None}
is_auth_on_indexer = False


def get_wazuh_token():
    """Authenticate with Wazuh API and get JWT token."""
    try:
        response = requests.post(
            f"{WAZUH_API_URL}/security/user/authenticate",
            auth=(WAZUH_USER, WAZUH_PASSWORD),
            verify=False
        )
        response.raise_for_status()
        return response.json()["data"]["token"]
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Auth failed: {str(e)}")

def indexer_auth():
    """Check if Indexer credentials are valid."""
    try:
        resp = requests.get(
            INDEXER_URL,
            auth=HTTPBasicAuth(INDEXER_USER, INDEXER_PASSWORD),
            verify=False
        )
        resp.raise_for_status()
        is_auth_on_indexer = True
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Auth failed: {e}")


def get_headers():
    """Return headers with valid JWT token."""
    if not token_cache["token"]:
        token_cache["token"] = get_wazuh_token()
    return {"Authorization": f"Bearer {token_cache['token']}"}


@router.post("/refresh-token")
def refresh_wazuh_token():
    """
    Refresh the Wazuh auths
    """
    try:
        get_wazuh_token()
        indexer_auth()
        return {"detail": "Token refreshed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))