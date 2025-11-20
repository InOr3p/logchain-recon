import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
from sentence_transformers import SentenceTransformer
from fastapi_backend.routers import agents, graphs, logs, users
from fastapi_backend.services.classifier_service import ClassifierService
from fastapi_backend.services.graph_builder_service import GraphBuilder
from fastapi_backend.services.edge_predictor_service import EdgePredictorService
from fastapi_backend.services.report_generation_service import ReportGenerationService

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
CLASSIFIER_PATH = os.path.join(AI_MODELS_DIR, "XGBoost_fd_all_cat.joblib")
FEATURE_LIST_PATH = os.path.join(AI_MODELS_DIR, "feature_columns.joblib")

# Paths for EdgePredictorService
EDGE_PREDICTOR_MODEL_PATH = os.path.join(AI_MODELS_DIR, "gnn_edge_predictor_10epochs.pth")
EDGE_PREDICTOR_HIDDEN_CHANNELS = 128

# Directory where the builder will save temporary .npz graph files
GRAPH_CACHE_DIR = os.path.join("fastapi_backend", "graph_cache")

# Logs dataset path
LOGS_DATASET_PATH = os.path.join("extracted_dataset", "sorted_ds_with_labels.parquet")


@app.on_event("startup")
async def load_all_models():
    """
    Load all models and feature lists into memory and
    create service instances on app.state.
    """
    print("--- Application Startup: Loading models ---")
    
    # 1. Load classifier model
    if not os.path.exists(CLASSIFIER_PATH):
        raise FileNotFoundError(f"Model file not found: {CLASSIFIER_PATH}")
    lgbm_model = joblib.load(CLASSIFIER_PATH)
    print(f"Successfully loaded classifier model from {CLASSIFIER_PATH}")
    
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

    # 4. Create and store the ClassifierService instance
    classifier_service = ClassifierService(
        model=lgbm_model,
        embedding_model=embedding_model,
        expected_features=expected_features
    )
    app.state.classifier_service = classifier_service
    print("ClassifierService is ready.")
    
    # 5. Load the GraphBuilder
    print("\n--- Loading GraphBuilder ---")
    try:
        if not os.path.isdir(AI_MODELS_DIR):
            raise FileNotFoundError(
                f"GraphBuilder state directory not found: {AI_MODELS_DIR}\n"
                f"(Expected to find scaler.joblib and cat_mappers.json here)"
            )

        os.makedirs(GRAPH_CACHE_DIR, exist_ok=True)
        
        graph_builder = GraphBuilder(
            output_dir=GRAPH_CACHE_DIR, 
            numeric_cols=['rule_level', 'rule_firedtimes', 'rule_id'],
            cat_cols=['rule_groups', 'rule_nist_800_53', 'rule_gdpr'],
            ip_cols=['agent_ip', 'data_srcip'],
            desc_col='description_vector',
            min_nodes_per_graph=5,
            max_nodes_per_graph=20,
            candidate_edge_topk=10,
            positive_neighbor_window=5,
            hash_dim_ip=16,
            window_size=50,
            stride=25,
        )
        
        graph_builder.load_state(AI_MODELS_DIR)
        app.state.graph_builder = graph_builder
        print("GraphBuilder is ready.")
        
    except Exception as e:
        print(f"Failed to load GraphBuilder. Error: {e}")
        raise e
    
    # 6. Load the EdgePredictorService
    print("\n--- Loading EdgePredictorService ---")
    try:
        if not os.path.exists(EDGE_PREDICTOR_MODEL_PATH):
            print(f"Warning: Edge predictor model not found at {EDGE_PREDICTOR_MODEL_PATH}")
            print("EdgePredictorService will be available but model will load on first use.")
        
        edge_predictor_service = EdgePredictorService(
            model_path=EDGE_PREDICTOR_MODEL_PATH,
            hidden_channels=EDGE_PREDICTOR_HIDDEN_CHANNELS,
        )
        
        app.state.edge_predictor_service = edge_predictor_service
        print("EdgePredictorService is ready.")
        
    except Exception as e:
        print(f"Failed to initialize EdgePredictorService. Error: {e}")
        raise e
    
    # 7. Load the ReportGenerationService
    print("\n--- Loading ReportGenerationService ---")
    try:
        report_service = ReportGenerationService(
            timeout=180
        )
        
        # Check if Groq is available
        if report_service.check_groq_health():
            print("Groq service is running and accessible")
        else:
            print("Warning: Groq service is not accessible")
            print("Reports will fail until Groq is started")
        
        app.state.report_service = report_service
        print("ReportGenerationService is ready.")
        
    except Exception as e:
        print(f"Warning: Failed to initialize ReportGenerationService. Error: {e}")
        print("Report generation will not be available")
        app.state.report_service = None
    
    print("\n--- All models loaded. Service is ready. ---")