from fastapi import APIRouter, HTTPException, Request
from typing import List
from fastapi_backend.models.schemas import ClassifyResponse, LogItem, Prediction
from fastapi_backend.services.classifier_service import ClassifierService


router = APIRouter(prefix="/logs", tags=["Wazuh Logs"])

@router.post("/classify", response_model=ClassifyResponse)
async def classify_logs(logs: List[LogItem], request: Request):
    """
    Receives a list of logs, preprocesses them, and returns predictions.
    """
    
    # 1. Get the service instance from the app state
    try:
        service: ClassifierService = request.app.state.classifier_service
    except AttributeError:
        raise HTTPException(
            status_code=503, # Service Unavailable
            detail="Classifier service is not loaded. Please check server logs."
        )

    if not logs:
        return ClassifyResponse(predictions=[])

    try:
        # 2. Preprocess the data
        df_features = service.preprocess(logs)
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback
        raise HTTPException(
            status_code=400, # Bad Request
            detail=f"Data preprocessing error: {e}. Check log format."
        )

    # 3. Run Inference
    pred_scores = service.predict(df_features)

    # 4. Format the response
    threshold = 0.5  # Your decision threshold
    label_map = {0: "Benign", 1: "Attack"}
    
    response_predictions = []
    for i, log in enumerate(logs):
        score = float(pred_scores[i])
        is_attack = score >= threshold
        label_int = 1 if is_attack else 0
        
        response_predictions.append(
            Prediction(
                id=log.id,
                prediction_label=label_map[label_int],
                prediction_score=score,
                is_attack=is_attack
            )
        )
        
    return ClassifyResponse(predictions=response_predictions)