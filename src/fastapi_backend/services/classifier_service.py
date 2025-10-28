# /classifier_service.py
import pandas as pd
import numpy as np
import ast
from typing import List, Any
from sentence_transformers import SentenceTransformer

from fastapi_backend.models.schemas import LogItem

# --- Module-level Helper Functions (copied from your logic) ---

def _safe_parse_list(x):
    """Robustly parse list-based features."""
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, list): return x
    if x is None or (isinstance(x, float) and np.isnan(x)): return []
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []

def _one_hot_encode_cat_cols(df: pd.DataFrame, col_str: str, col_name: str, col_values: List[str]):
    """Creates one-hot columns on the DataFrame."""
    for el in col_values:
        new_col_name = f'{col_str}{el}'
        df[new_col_name] = df[col_name].apply(
            lambda x: 1 if isinstance(x, list) and el in x else 0
        )

def _optimize_dtypes(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Applies the same integer type optimization as training."""
    for col in feature_cols:
        if col in df.columns and df[col].dtype in ['int64', 'int32']:
            max_val = df[col].max()
            min_val = df[col].min()
            if min_val >= 0:
                if max_val < 256: df[col] = df[col].astype('uint8')
                elif max_val < 65536: df[col] = df[col].astype('uint16')
                elif max_val < 4294967296: df[col] = df[col].astype('uint32')
            else:
                if min_val >= -128 and max_val <= 127: df[col] = df[col].astype('int8')
                elif min_val >= -32768 and max_val <= 32767: df[col] = df[col].astype('int16')
                elif min_val >= -2147483648 and max_val <= 2147483647: df[col] = df[col].astype('int32')
    
    emb_cols = [col for col in df.columns if col.startswith('emb_')]
    if emb_cols:
        df[emb_cols] = df[emb_cols].astype('float16')
    return df

# --- The Main Service Class ---

class ClassifierService:
    def __init__(self, model: Any, embedding_model: SentenceTransformer, expected_features: List[str]):
        """
        Initializes the service with the loaded models and feature list.
        """
        self.model = model
        self.embedding_model = embedding_model
        self.expected_features = expected_features
        
        # Pre-calculate feature name lists from the loaded list
        self.global_feature_lists = {
            "all_groups": [col.replace('group_', '') for col in expected_features if col.startswith('group_')],
            "all_nist": [col.replace('nist_', '') for col in expected_features if col.startswith('nist_')],
            "all_gdpr": [col.replace('gdpr_', '') for col in expected_features if col.startswith('gdpr_')],
            "basic_cols": ['rule_id', 'rule_level', 'rule_firedtimes'],
            "emb_cols": [col for col in expected_features if col.startswith('emb_')]
        }
        print("ClassifierService initialized successfully.")
        print(f"Found {len(self.global_feature_lists['all_groups'])} 'group' features.")
        print(f"Found {len(self.global_feature_lists['emb_cols'])} 'embedding' features.")

    def preprocess(self, logs: List[LogItem]) -> pd.DataFrame:
        """
        The main preprocessing pipeline for inference.
        Replicates the 'engineer_features' function.
        """
        # 1. Convert Pydantic models to DataFrame
        df = pd.DataFrame([log.model_dump() for log in logs])
        
        # 2. Generate Embeddings
        if not self.global_feature_lists["emb_cols"]:
            embedding_df = pd.DataFrame(index=df.index)
        else:
            descriptions = df['rule_description'].fillna('').tolist()
            embeddings = self.embedding_model.encode(descriptions, show_progress_bar=False)
            
            expected_dim = len(self.global_feature_lists['emb_cols'])
            if embeddings.shape[1] != expected_dim:
                raise ValueError(
                    f"Embedding dimension mismatch! Model produced {embeddings.shape[1]} features, "
                    f"but expected_features.joblib expects {expected_dim}."
                )
                
            embedding_df = pd.DataFrame(
                embeddings, 
                columns=self.global_feature_lists["emb_cols"]
            ).astype('float16')

        # 3. Robustly parse list-based features
        df['rule_groups'] = df['rule_groups'].apply(_safe_parse_list)
        df['rule_nist_800_53'] = df['rule_nist_800_53'].apply(_safe_parse_list)
        df['rule_gdpr'] = df['rule_gdpr'].apply(_safe_parse_list)

        # 4. Create binary columns (one-hot encoding)
        _one_hot_encode_cat_cols(df, 'group_', 'rule_groups', self.global_feature_lists['all_groups'])
        _one_hot_encode_cat_cols(df, 'nist_', 'rule_nist_800_53', self.global_feature_lists['all_nist'])
        _one_hot_encode_cat_cols(df, 'gdpr_', 'rule_gdpr', self.global_feature_lists['all_gdpr'])

        # 5. Combine all features
        feature_cols = (
            self.global_feature_lists['basic_cols'] +
            [f'group_{e}' for e in self.global_feature_lists['all_groups']] +
            [f'nist_{e}' for e in self.global_feature_lists['all_nist']] +
            [f'gdpr_{e}' for e in self.global_feature_lists['all_gdpr']]
        )
        
        df['rule_id'] = df['rule_id'].astype(int)
        X_basic = df[feature_cols].copy()
        X_basic = _optimize_dtypes(X_basic, feature_cols)
        
        X_combined = pd.concat(
            [X_basic.reset_index(drop=True), embedding_df.reset_index(drop=True)],
            axis=1
        )
        
        # 6. Align with saved feature space (CRITICAL)
        X_aligned = pd.DataFrame(columns=self.expected_features, dtype='float16')
        X_aligned = X_aligned.reindex(columns=X_combined.columns, fill_value=0).combine_first(X_combined)
        X_aligned = X_aligned[self.expected_features]
        X_aligned = X_aligned.fillna(0)
        X_aligned = _optimize_dtypes(X_aligned, self.expected_features)

        return X_aligned

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Runs inference using the loaded model.
        Returns the raw probabilities for the positive class.
        """
        pred_probabilities = self.model.predict_proba(features)
        pred_scores = pred_probabilities[:, 1]
        return pred_scores