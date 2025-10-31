import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
from sentence_transformers import SentenceTransformer
from fastapi_backend.routers import agents, graphs, logs, users
from fastapi_backend.services.classifier_service import ClassifierService
from fastapi_backend.services.graph_builder_service import GraphBuilder

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
app.include_router(graphs.router)
# app.include_router(reports.router)

@app.get("/")
def root():
    return {"message": "Logchain Recon backend is running"}


# --- Model & Path Constants ---
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
AI_MODELS_DIR = os.path.join("fastapi_backend", "AI_models")

# Paths for ClassifierService
LIGHTGBM_PATH = os.path.join(AI_MODELS_DIR, "LightGBM_fd_all_cat.joblib")
FEATURE_LIST_PATH = os.path.join(AI_MODELS_DIR, "feature_columns.joblib")

# Directory where the builder will save temporary .npz graph files
GRAPH_CACHE_DIR = os.path.join("fastapi_backend", "graph_cache")

# --- 3. (Removed unused/redundant global model variables) ---


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
    # Use a clear local variable name
    lgbm_model = joblib.load(LIGHTGBM_PATH)
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
        model=lgbm_model,
        embedding_model=embedding_model,
        expected_features=expected_features
    )
    app.state.classifier_service = service_instance
    print("--- ClassifierService is ready. ---")
    
    # --- LOAD THE GRAPH BUILDER ---
    print("--- Loading GraphBuilder ---")
    try:
        # Ensure the state directory exists
        if not os.path.isdir(AI_MODELS_DIR):
            raise FileNotFoundError(
                f"GraphBuilder state directory not found: {AI_MODELS_DIR}\n"
                f"(Expected to find scaler.joblib and cat_mappers.json here)"
            )

        # Ensure the cache directory exists
        os.makedirs(GRAPH_CACHE_DIR, exist_ok=True)
        
        # Instantiate the builder
        # !! IMPORTANT: Update these params to match your training config !!
        graph_builder = GraphBuilder(
            output_dir=GRAPH_CACHE_DIR, 
            numeric_cols=['rule_level', 'rule_firedtimes', 'rule_id'],
            cat_cols=['rule_groups', 'rule_nist_800_53', 'rule_gdpr'],
            ip_cols=['agent_ip', 'data_srcip'],
            desc_col='description_vector', # This must match what you use
            min_nodes_per_graph=5,
            max_nodes_per_graph=20,
            candidate_edge_topk=10,
            positive_neighbor_window=5,
            hash_dim_ip=16,
            window_size=50,
            stride=25,
        )
        
        # Load the pre-fitted scaler and mappers
        graph_builder.load_state(AI_MODELS_DIR)
        
        # Store the ready-to-use instance on app.state
        # This will be accessible via `request.app.state.graph_builder`
        app.state.graph_builder = graph_builder
        print("--- GraphBuilder is ready. ---")
        
    except Exception as e:
        print(f"CRITICAL: Failed to load GraphBuilder. Error: {e}")
        # Depending on your needs, you might want to exit the app
        # raise e 
    
    print("--- All models loaded. Service is ready. ---")