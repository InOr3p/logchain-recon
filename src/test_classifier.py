import pandas as pd
import numpy as np
import joblib
import json
import os
import ast
from sentence_transformers import SentenceTransformer

class LogEvaluator:
    """
    Evaluates exported Wazuh logs using a pre-trained LightGBM classifier.
    """
    
    def __init__(self, model_path, feature_columns_path, vectorizer_model='all-MiniLM-L6-v2'):
        """
        Initialize the evaluator.
        
        Parameters
        ----------
        model_path : str
            Path to the saved LightGBM model (.joblib file)
        feature_columns_path : str
            Path to the saved feature columns list (.joblib file)
        vectorizer_model : str
            Name of the sentence transformer model for text vectorization
        """
        print("Loading model and feature space...")
        self.model = joblib.load(model_path)
        self.feature_columns = joblib.load(feature_columns_path)
        self.vectorizer = SentenceTransformer(vectorizer_model)
        print(f"Model loaded. Expected features: {len(self.feature_columns)}")
        
    def load_logs(self, log_file_path):
        """
        Load logs from JSON or CSV file and normalize column names.
        
        Parameters
        ----------
        log_file_path : str
            Path to the exported logs file (.json or .csv)
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing the logs
        """
        print(f"\nLoading logs from {log_file_path}...")
        
        if log_file_path.endswith('.json'):
            with open(log_file_path, 'r') as f:
                logs_data = json.load(f)
            df = pd.DataFrame(logs_data)
        elif log_file_path.endswith('.csv'):
            df = pd.read_csv(log_file_path)
        else:
            raise ValueError("Unsupported file format. Use .json or .csv")
        
        print(f"Loaded {len(df)} logs")
        print(f"Original columns: {df.columns.tolist()}")
        
        # Map exported column names to expected column names
        column_mapping = {
            'Timestamp': '@timestamp',
            'Description': 'rule_description',
            'Rule ID': 'rule_id',
            'Level': 'rule_level',
            'Firedtimes': 'rule_firedtimes',
            'Agent IP': 'agent_ip',
            'Source IP': 'data_srcip',
            'Groups': 'rule_groups',
            'NIST 800-53': 'rule_nist_800_53',
            'GDPR': 'rule_gdpr'
        }
        
        # Rename columns
        df.rename(columns=column_mapping, inplace=True)
        
        print(f"Normalized columns: {df.columns.tolist()}")
        
        return df
    
    def _safe_parse_list(self, x):
        """Parse list-based features robustly."""
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, list):
            return x
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return []
        if isinstance(x, str):
            try:
                parsed = ast.literal_eval(x)
                return parsed if isinstance(parsed, list) else []
            except Exception:
                return []
        return []
    
    def engineer_features(self, df):
        """
        Engineer the same features used during training.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw logs dataframe
        
        Returns
        -------
        pd.DataFrame
            Feature matrix ready for prediction
        """
        print("\nEngineering features...")
        
        # Check if rule_description exists
        if 'rule_description' not in df.columns:
            raise ValueError("Column 'rule_description' not found in logs")
        
        # Vectorize descriptions
        print("Vectorizing log descriptions...")
        descriptions = df['rule_description'].fillna('').tolist()
        description_vectors = self.vectorizer.encode(descriptions, show_progress_bar=True)
        
        embedding_dim = description_vectors.shape[1]
        embedding_df = pd.DataFrame(
            description_vectors,
            columns=[f'emb_{i}' for i in range(embedding_dim)]
        ).astype('float16')
        
        # Parse list-based features
        df['rule_groups'] = df['rule_groups'].apply(self._safe_parse_list)
        df['rule_nist_800_53'] = df['rule_nist_800_53'].apply(self._safe_parse_list)
        df['rule_gdpr'] = df['rule_gdpr'].apply(self._safe_parse_list)
        
        # Extract categorical values from feature columns
        group_cols = [col for col in self.feature_columns if col.startswith('group_')]
        nist_cols = [col for col in self.feature_columns if col.startswith('nist_')]
        gdpr_cols = [col for col in self.feature_columns if col.startswith('gdpr_')]
        
        all_groups = [g.replace('group_', '') for g in group_cols]
        all_nist_values = [n.replace('nist_', '') for n in nist_cols]
        all_gdpr_values = [g.replace('gdpr_', '') for g in gdpr_cols]
        
        # Create binary columns for categorical features
        for el in all_groups:
            col_name = f'group_{el}'
            df[col_name] = df['rule_groups'].apply(
                lambda x: 1 if isinstance(x, list) and el in x else 0
            )
        
        for el in all_nist_values:
            col_name = f'nist_{el}'
            df[col_name] = df['rule_nist_800_53'].apply(
                lambda x: 1 if isinstance(x, list) and el in x else 0
            )
        
        for el in all_gdpr_values:
            col_name = f'gdpr_{el}'
            df[col_name] = df['rule_gdpr'].apply(
                lambda x: 1 if isinstance(x, list) and el in x else 0
            )
        
        # Convert rule_id to integer
        df['rule_id'] = df['rule_id'].astype(int)
        
        # Build basic features
        basic_feature_cols = ['rule_id', 'rule_level', 'rule_firedtimes']
        
        # Check if rule_firedtimes exists, if not create it with default value
        if 'rule_firedtimes' not in df.columns:
            df['rule_firedtimes'] = 1
        
        feature_cols = (
            basic_feature_cols +
            [f'group_{e}' for e in all_groups] +
            [f'nist_{e}' for e in all_nist_values] +
            [f'gdpr_{e}' for e in all_gdpr_values]
        )
        
        # Combine basic features with embeddings
        X_basic = df[feature_cols].copy()
        X_combined = pd.concat(
            [X_basic.reset_index(drop=True), embedding_df.reset_index(drop=True)],
            axis=1
        )
        
        # Align with expected feature space
        for col in self.feature_columns:
            if col not in X_combined.columns:
                X_combined[col] = 0
        
        X_combined = X_combined[self.feature_columns]
        
        print(f"Feature engineering complete. Shape: {X_combined.shape}")
        return X_combined
    
    def evaluate(self, log_file_path, output_file=None):
        """
        Evaluate logs and return predictions.
        
        Parameters
        ----------
        log_file_path : str
            Path to the exported logs file
        output_file : str, optional
            Path to save results (CSV format)
        
        Returns
        -------
        pd.DataFrame
            Original logs with predictions and confidence scores
        """
        # Load logs
        df = self.load_logs(log_file_path)
        
        # Engineer features
        X = self.engineer_features(df)
        
        # Make predictions
        print("\nMaking predictions...")
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        # Add predictions to original dataframe
        df['prediction'] = ['Attack' if p == 1 else 'Benign' for p in y_pred]
        df['confidence'] = y_pred_proba
        df['confidence_pct'] = (y_pred_proba * 100).round(2)
        
        # Display results summary
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"\nTotal logs evaluated: {len(df)}")
        print(f"Predicted as Attack: {sum(y_pred == 1)} ({sum(y_pred == 1)/len(df)*100:.1f}%)")
        print(f"Predicted as Benign: {sum(y_pred == 0)} ({sum(y_pred == 0)/len(df)*100:.1f}%)")
        
        # Show high-confidence attacks
        high_conf_attacks = df[(df['prediction'] == 'Attack') & (df['confidence'] >= 0.8)]
        if len(high_conf_attacks) > 0:
            print(f"\nHigh-confidence attacks detected: {len(high_conf_attacks)}")
            print("\nTop 5 high-confidence attack predictions:")
            print("-" * 60)
            for idx, row in high_conf_attacks.nlargest(5, 'confidence').iterrows():
                print(f"\nTimestamp: {row.get('@timestamp', 'N/A')}")
                print(f"Rule ID: {row.get('rule_id', 'N/A')}")
                print(f"Description: {row.get('rule_description', 'N/A')[:80]}...")
                print(f"Confidence: {row['confidence_pct']:.2f}%")
                print(f"Level: {row.get('rule_level', 'N/A')}")
        
        # Show statistics by rule level
        if 'rule_level' in df.columns:
            print("\n" + "="*60)
            print("PREDICTIONS BY RULE LEVEL")
            print("="*60)
            level_stats = df.groupby('rule_level').agg({
                'prediction': lambda x: (x == 'Attack').sum(),
                'confidence': 'mean'
            }).rename(columns={'prediction': 'attack_count', 'confidence': 'avg_confidence'})
            level_stats['total'] = df.groupby('rule_level').size()
            level_stats['attack_pct'] = (level_stats['attack_count'] / level_stats['total'] * 100).round(1)
            print(level_stats.to_string())
        
        # Save results if requested
        if output_file:
            # Select relevant columns for output
            output_cols = [
                'timestamp', 'rule_id', 'rule_description', 'rule_level',
                'agent_ip', 'data_srcip', 'prediction', 'confidence_pct'
            ]
            output_cols = [col for col in output_cols if col in df.columns]
            
            df[output_cols].to_csv(output_file, index=False)
            print(f"\nResults saved to {output_file}")
        
        return df


# --- Main execution ---
if __name__ == "__main__":
    # Configuration
    EXTRACT_DIR = "extracted_dataset"
    MODEL_PATH = os.path.join("models", "LightGBM_fd_all_cat.joblib")
    FEATURE_COLUMNS_PATH = os.path.join(EXTRACT_DIR, "feature_columns.joblib")
    LOG_FILE = os.path.join(EXTRACT_DIR, "logs_export_2025-10-29.csv")
    OUTPUT_FILE = os.path.join(EXTRACT_DIR, "lightgbm_evaluation_results_11-18.csv")
        
    # Check if files exist
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        exit(1)
    
    if not os.path.exists(FEATURE_COLUMNS_PATH):
        print(f"Error: Feature columns file not found at {FEATURE_COLUMNS_PATH}")
        exit(1)
    
    if not os.path.exists(LOG_FILE):
        print(f"Error: Log file not found at {LOG_FILE}")
        exit(1)
    
    # Initialize evaluator
    evaluator = LogEvaluator(
        model_path=MODEL_PATH,
        feature_columns_path=FEATURE_COLUMNS_PATH
    )
    
    # Evaluate logs
    results_df = evaluator.evaluate(LOG_FILE, output_file=OUTPUT_FILE)
    
    # Optional: Display sample predictions
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS (First 10 logs)")
    print("="*60)
    sample_cols = ['@timestamp', 'rule_id', 'rule_description', 'rule_level', 
                   'prediction', 'confidence_pct']
    sample_cols = [col for col in sample_cols if col in results_df.columns]
    print(results_df[sample_cols].head(10).to_string(index=False))