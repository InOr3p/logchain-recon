from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_backend.routers import agents, logs, users
import os
from sentence_transformers import SentenceTransformer
import joblib
from typing import List

from fastapi_backend.services.classifier_service import ClassifierService

app = FastAPI(title="Logchain Recon API")

# Allow frontend (SvelteKit) to call it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(logs.router)
app.include_router(users.router)
app.include_router(agents.router)
# app.include_router(graphs.router)
# app.include_router(reports.router)

@app.get("/")
def root():
    return {"message": "Logchain Recon backend is running"}


# Load a pre-trained Sentence Transformer model.
# This model is excellent for semantic similarity tasks.
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("Model loaded.")

# Load LightGBM model once at startup
AI_MODELS_DIR = os.path.join("fastapi_backend", "AI_models")
LIGHTGBM_PATH = os.path.join(AI_MODELS_DIR, "LightGBM_fd_all_cat.joblib")
FEATURE_LIST_PATH = os.path.join(AI_MODELS_DIR, "feature_columns.joblib")

lightgbm_model = None
embedding_model = None
expected_features: List[str] = []
global_feature_lists = {
    "all_groups": [],
    "all_nist": [],
    "all_gdpr": [],
    "basic_cols": ['rule_id', 'rule_level', 'rule_firedtimes'],
    "emb_cols": []
}

@app.on_event("startup")
def load_all_models():
    """
    Load all models and feature lists into memory and
    create the classifier service instance on app.state.
    """
    print("--- Application Startup: Loading models ---")
    
    # 1. Load LightGBM model
    if not os.path.exists(LIGHTGBM_PATH):
        raise FileNotFoundError(f"Model file not found: {LIGHTGBM_PATH}")
    model = joblib.load(LIGHTGBM_PATH)
    print(f"Successfully loaded model from {LIGHTGBM_PATH}")
    
    # 2. Load expected feature columns list
    if not os.path.exists(FEATURE_LIST_PATH):
        raise FileNotFoundError(f"Feature list file not found: {FEATURE_LIST_PATH}")
    expected_features = joblib.load(FEATURE_LIST_PATH)
    print(f"Successfully loaded {len(expected_features)} feature columns.")
    
    # 3. Load Embedding model
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"Successfully loaded embedding model: {EMBEDDING_MODEL_NAME}")
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model. Error: {e}")

    # 4. Create and store the service instance
    # This instance will be accessible via `request.app.state.classifier_service`
    service_instance = ClassifierService(
        model=model,
        embedding_model=embedding_model,
        expected_features=expected_features
    )
    app.state.classifier_service = service_instance
    
    print("--- Model loading complete. Service is ready. ---")
