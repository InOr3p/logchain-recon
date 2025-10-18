import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_recall_fscore_support, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from sklearn.model_selection import KFold
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os
import joblib


class WazuhLogClassifier:
    """
    A classifier for Wazuh logs to detect attacks vs benign traffic.
    Compares Neural Network, LightGBM, CatBoost, XGBoost models.
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None
        self.X_train_scaled, self.X_test_scaled = None, None # For NN
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.categorical_features = []
        
    def load_and_preprocess_random_data(self):
        """Load and preprocess the Wazuh log data."""
        print("Loading data...")
        self.df = pd.read_parquet(self.data_path)

        description_vectors = np.load(os.path.join(extract_dir, 'vectorized_descr.npy'))
        self.df['description_vector'] = list(description_vectors)

        threshold = 700_000
        if len(self.df) > threshold:
            print(f"Original dataset size: {len(self.df)}. Sampling down to {threshold} rows.")
            self.df = self.df.sample(n=threshold, random_state=42).reset_index(drop=True)

        print(f"Dataset shape: {self.df.shape}")
        print(f"\nClass distribution:\n{self.df['attack_label'].value_counts()}")
        
        return self
    

    def load_and_preprocess_fixed_data(self):
        """
        Load and preprocess Wazuh log data, creating a balanced dataset of a fixed size
        by compensating for under-represented attack types.
        """
        print("Loading data...")
        self.df = pd.read_parquet(self.data_path)
        
        description_vectors = np.load(os.path.join(extract_dir, 'vectorized_descr.npy'))
        self.df['description_vector'] = list(description_vectors)

        # --- Configuration for the new dataset ---
        TOTAL_SIZE = 250_000
        RANDOM_STATE = 42  # For reproducibility

        benign_size = TOTAL_SIZE // 2
        attack_size = TOTAL_SIZE - benign_size

        attack_types_to_include = [
            'dirb', 'wpscan', 'dnsteal', 'cracking', 'service_scans',
            'network_scans', 'privilege_escalation', 'webshell', 'reverse_shell', 'service_stop'
        ]
        
        num_per_attack_type = attack_size // len(attack_types_to_include)
        print(f"Targeting {benign_size} benign logs and {attack_size} attack logs.")
        print(f"Ideal number per attack type: {num_per_attack_type}")
        print("-" * 30)

        # --- Sample Benign Logs ---
        benign_df = self.df[self.df['attack_label'] == 'benign']
        if len(benign_df) < benign_size:
            raise ValueError(f"Not enough benign logs to meet the target. Have {len(benign_df)}, need {benign_size}.")
        sampled_benign = benign_df.sample(n=benign_size, random_state=RANDOM_STATE)

        # --- Sample Attack Logs (Two-Pass Strategy) ---

        # Take a "fair share" from each attack type, up to the ideal number.
        print("Attack Sampling - Taking a fair share from each type...")
        sampled_attack_dfs = []
        already_sampled_indices = set()

        for attack_type in attack_types_to_include:
            attack_subset_df = self.df[self.df['type_attack_label'] == attack_type]
            
            # Take at most num_per_attack_type, or all if fewer are available
            n_to_sample = min(len(attack_subset_df), num_per_attack_type)

            if n_to_sample > 0:
                sampled_subset = attack_subset_df.sample(n=n_to_sample, random_state=RANDOM_STATE)
                sampled_attack_dfs.append(sampled_subset)
                already_sampled_indices.update(sampled_subset.index)
            
            if n_to_sample < num_per_attack_type:
                print(f"  - Warning: Only found {n_to_sample} logs for '{attack_type}' (target was {num_per_attack_type}).")

        # Calculate the shortfall after the first pass
        current_attack_count = sum(len(df) for df in sampled_attack_dfs)
        shortfall = attack_size - current_attack_count

        # Compensate for the shortfall from the remaining pool.
        if shortfall > 0:
            print(f"\nAttack Sampling - Shortfall of {shortfall} logs detected. Compensating...")
            
            # Create a pool of all eligible attack logs that haven't been sampled yet
            compensation_pool = self.df[
                (self.df['type_attack_label'].isin(attack_types_to_include)) &
                (~self.df.index.isin(already_sampled_indices))
            ]

            if len(compensation_pool) < shortfall:
                # This is a critical failure - not enough total attack logs exist
                raise ValueError(f"Cannot compensate for shortfall. Only {len(compensation_pool)} remaining attack logs available, but need {shortfall}.")

            # Sample from the compensation pool to fill the gap
            compensation_samples = compensation_pool.sample(n=shortfall, random_state=RANDOM_STATE)
            sampled_attack_dfs.append(compensation_samples)
            print(f"  - Successfully sampled {len(compensation_samples)} additional logs to meet the target.")

        # --- Combine and Finalize the Dataset ---
        final_df = pd.concat([sampled_benign] + sampled_attack_dfs, ignore_index=True)

        # Shuffle the final combined dataset
        self.df = final_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

        print("\n--- Final Dataset Summary ---")
        print(f"Final dataset shape: {self.df.shape}")
        print(f"\nFinal 'attack_label' distribution:\n{self.df['attack_label'].value_counts()}")
        
        attack_logs_in_final_df = self.df[self.df['attack_label'] == 'attack']
        print(f"\nFinal 'type_attack_label' distribution for attacks:\n{attack_logs_in_final_df['type_attack_label'].value_counts()}")
        
        return self

    def engineer_features(self):
        """
        Engineer deterministic features from the raw data, ensuring a consistent
        feature space across different datasets (same number of columns).
        """
        print("\nEngineering features...")

        feature_list_path = os.path.join(extract_dir, "feature_columns.joblib")

        # --- Parse description_vector ---
        if isinstance(self.df['description_vector'].iloc[0], str):
            self.df['description_vector'] = self.df['description_vector'].apply(
                lambda x: np.array(ast.literal_eval(x)) if pd.notna(x) else np.zeros(384)
            )

        embedding_dim = len(self.df['description_vector'].iloc[0])
        embedding_features = np.vstack(self.df['description_vector'].values)
        embedding_df = pd.DataFrame(
            embedding_features,
            columns=[f'emb_{i}' for i in range(embedding_dim)]
        ).astype('float16')

        # --- Parse and flatten rule_groups ---
        if isinstance(self.df['rule_groups'].iloc[0], str):
            self.df['rule_groups'] = self.df['rule_groups'].apply(
                lambda x: ast.literal_eval(x) if pd.notna(x) else []
            )

        # Determine if we already have a saved feature list (for consistent space)
        if os.path.exists(feature_list_path):
            print("Loading existing feature space...")
            saved_features = joblib.load(feature_list_path)
            # Extract only the 'group_' features from saved list
            saved_groups = [col for col in saved_features if col.startswith('group_')]
            all_groups = [g.replace('group_', '') for g in saved_groups]
        else:
            print("Creating new feature space from current dataset...")
            all_groups = sorted(self.df['rule_groups'].explode().dropna().unique())

        print(f"Total unique rule_groups: {len(all_groups)}")

        # Create binary columns for all possible groups
        for group in all_groups:
            col_name = f'group_{group}'
            self.df[col_name] = self.df['rule_groups'].apply(lambda x: 1 if group in x else 0)

        # --- Process NIST and GDPR rules ---
        for col in ['rule_nist_800_53', 'rule_gdpr']:
            if isinstance(self.df[col].iloc[0], str):
                self.df[col] = self.df[col].apply(
                    lambda x: ast.literal_eval(x) if pd.notna(x) else []
                )
            else:
                self.df[col] = self.df[col].apply(lambda x: x if isinstance(x, list) else [])

        self.df['nist_count'] = self.df['rule_nist_800_53'].apply(len)
        self.df['has_nist'] = (self.df['nist_count'] > 0).astype(int)
        self.df['gdpr_count'] = self.df['rule_gdpr'].apply(len)
        self.df['has_gdpr'] = (self.df['gdpr_count'] > 0).astype(int)

        # --- Combine all features ---
        feature_cols = (
            ['rule_id', 'rule_level', 'rule_firedtimes',
            'nist_count', 'has_nist', 'gdpr_count', 'has_gdpr'] +
            [f'group_{g}' for g in all_groups]
        )

        # Convert rule_id to integer
        print("Converting 'rule_id' to integer...")
        self.df['rule_id'] = self.df['rule_id'].astype(int)

        # Optimize integer types
        print("\nOptimizing data types to save memory...")
        for col in feature_cols:
            if self.df[col].dtype == 'int64':
                max_val = self.df[col].max()
                min_val = self.df[col].min()
                if min_val >= 0:
                    if max_val < 256:
                        self.df[col] = self.df[col].astype('uint8')
                    elif max_val < 65536:
                        self.df[col] = self.df[col].astype('uint16')
                    elif max_val < 4294967296:
                        self.df[col] = self.df[col].astype('uint32')
                else:
                    if min_val >= -128 and max_val <= 127:
                        self.df[col] = self.df[col].astype('int8')
                    elif min_val >= -32768 and max_val <= 32767:
                        self.df[col] = self.df[col].astype('int16')
                    elif min_val >= -2147483648 and max_val <= 2147483647:
                        self.df[col] = self.df[col].astype('int32')

        # Combine basic and embedding features
        X_basic = self.df[feature_cols].copy()
        X_combined = pd.concat(
            [X_basic.reset_index(drop=True), embedding_df.reset_index(drop=True)],
            axis=1
        )

        y = (self.df['attack_label'] == 'attack').astype(int)

        # --- Align with saved feature space (if exists) ---
        if os.path.exists(feature_list_path):
            print("Aligning current dataset with saved feature space...")
            expected_features = joblib.load(feature_list_path)

            # Add any missing columns with zeros
            for col in expected_features:
                if col not in X_combined.columns:
                    X_combined[col] = 0

            # Remove any unexpected extra columns
            X_combined = X_combined[expected_features]
        else:
            # First run → save feature list
            print(f"Saving feature list with {len(X_combined.columns)} columns.")
            joblib.dump(X_combined.columns.tolist(), feature_list_path)

        # Define categorical features
        self.categorical_features = [
            i for i, col in enumerate(X_combined.columns)
            if col.startswith('group_') or col in ['has_nist', 'has_gdpr']
        ]

        print(f"Total features: {X_combined.shape[1]}")
        print(f"  - Basic + embedding: {len(X_combined.columns)}")
        print(f"  - Categorical: {len(self.categorical_features)}")

        return X_combined, y
        
    
    def split_and_scale_data(self, X, y, test_size=0.2):
        """Split data into train/test and scale features."""
        print("\nSplitting and scaling data...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features (needed for Neural Network)
        # X_train_np = self.X_train.to_numpy(dtype='float32')
        # X_test_np = self.X_test.to_numpy(dtype='float32')
    
        # Scale the NumPy arrays
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
            
        print(f"Train set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
        return self
    

    def train_neural_network(self, epochs=20, batch_size=32, n_splits=5):
        """
        Train a Neural Network classifier using K-Fold Cross Validation and then
        retrain on the full training dataset for final evaluation.

        Stores both K-Fold metrics and final model metrics for comparison.

        Parameters
        ----------
        epochs : int
            Number of epochs for each fold / final training.
        batch_size : int
            Batch size used during training.
        n_splits : int
            Number of folds for K-Fold cross-validation.
        """
        
        print("\n" + "="*60)
        print("Training Neural Network Classifier with K-Fold Cross Validation...")
        print("="*60)

        # ---------------------------
        # Define function to build the NN model
        # ---------------------------
        def build_model(input_dim):
            model = keras.Sequential([
                layers.Input(shape=(input_dim,)),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(1, activation='sigmoid')
            ])
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
            )
            return model

        # ---------------------------
        # Initialize K-Fold cross-validator
        # ---------------------------
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_metrics = []
        histories = []

        self.X_train_scaled = self.X_train_scaled.astype(np.float32)
        self.y_train = self.y_train.to_numpy()  # Convert Series to NumPy array

        # ---------------------------
        # Perform K-Fold CV
        # ---------------------------
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_train_scaled)):
            print(f"\n--- Fold {fold+1}/{n_splits} ---")
            X_train_fold, X_val_fold = self.X_train_scaled[train_idx], self.X_train_scaled[val_idx]
            y_train_fold, y_val_fold = self.y_train[train_idx], self.y_train[val_idx]

            # Compute class weights
            class_weight = {
                0: len(y_train_fold) / (2 * np.sum(y_train_fold == 0)),
                1: len(y_train_fold) / (2 * np.sum(y_train_fold == 1))
            }

            # Build fresh model for this fold
            model = build_model(self.X_train_scaled.shape[1])

            # Early stopping callback
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )

            # Train on fold
            history = model.fit(
                X_train_fold, y_train_fold,
                validation_data=(X_val_fold, y_val_fold),
                epochs=epochs,
                batch_size=batch_size,
                class_weight=class_weight,
                callbacks=[early_stopping],
                verbose=1
            )
            histories.append(history.history)

            # Evaluate fold
            val_loss, val_acc, val_prec, val_rec = model.evaluate(X_val_fold, y_val_fold, verbose=0)
            f1 = 2 * (val_prec * val_rec) / (val_prec + val_rec + 1e-6)
            fold_metrics.append({
                "loss": val_loss,
                "accuracy": val_acc,
                "precision": val_prec,
                "recall": val_rec,
                "f1": f1
            })

        # ---------------------------
        # Compute average metrics across folds
        # ---------------------------
        avg_metrics = {
            metric: np.mean([fold[metric] for fold in fold_metrics])
            for metric in fold_metrics[0]
        }

        print("\nAverage CV Results (across all folds):")
        for k, v in avg_metrics.items():
            print(f"{k.capitalize()}: {v:.4f}")

        # ---------------------------
        # Retrain final model on full training set
        # ---------------------------
        print("\nRetraining final model on the full training dataset...")
        final_model = build_model(self.X_train_scaled.shape[1])

        # Compute class weights for full training data
        class_weight_full = {
            0: len(self.y_train) / (2 * np.sum(self.y_train == 0)),
            1: len(self.y_train) / (2 * np.sum(self.y_train == 1))
        }

        final_history = final_model.fit(
            self.X_train_scaled, self.y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_full,
            verbose=1
        )

        # Evaluate final model on test set
        y_pred_proba = final_model.predict(self.X_test_scaled).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Store models and results
        self.models['SimpleNN'] = final_model
        self.results['SimpleNN'] = {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'cv_metrics': fold_metrics,        # per-fold CV metrics
            'cv_avg_metrics': avg_metrics,     # mean CV metrics
            'cv_histories': histories,         # per-fold training curves
            'final_history': final_history.history  # full training history
        }

        # Print final test metrics
        self._print_metrics('SimpleNN', y_pred, y_pred_proba)

        return self

    
    def train_lightgbm(self):
        """Train LightGBM classifier with hyperparameter tuning."""
        print("\n" + "="*60)
        print("Training LightGBM Classifier...")
        print("="*60)
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = len(self.y_train[self.y_train == 0]) / len(self.y_train[self.y_train == 1])
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 400],
            'learning_rate': [0.05, 0.1],
            'max_depth': [5, 10, 15],
            'num_leaves': [31, 50, 80],
            'min_child_samples': [20, 50],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        lgbm = lgb.LGBMClassifier(
            objective='binary',
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        # Randomized search with cross-validation
        random_search = RandomizedSearchCV(
            lgbm,
            param_distributions=param_grid,
            n_iter=20,
            cv=3,    
            scoring='f1',
            n_jobs=1,
            verbose=1,
            random_state=42
        )

        # LightGBM print the evaluation metrics every 50 boosting rounds.
        log_callback = lgb.log_evaluation(period=50)

        random_search.fit(self.X_train, self.y_train, callbacks=[log_callback])
        
        best_lgbm = random_search.best_estimator_
        print(f"\nBest parameters: {random_search.best_params_}")

        best_lgbm.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],
            eval_names=['train', 'valid'],
            eval_metric=['binary_logloss', 'accuracy'],
        )
                
        # Predictions
        y_pred = best_lgbm.predict(self.X_test)
        y_pred_proba = best_lgbm.predict_proba(self.X_test)[:, 1]
        
        # Store model and results
        self.models['LightGBM'] = best_lgbm
        self.results['LightGBM'] = {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'best_params': random_search.best_params_
        }

        # Store evals_result_
        self.results['LightGBM']['evals_result'] = best_lgbm.evals_result_
        
        self._print_metrics('LightGBM', y_pred, y_pred_proba)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': best_lgbm.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        return self
    
    def train_catboost(self):
        """Train CatBoost classifier with hyperparameter tuning."""
        print("\n" + "="*60)
        print("Training CatBoost Classifier...")
        print("="*60)
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = np.sum(self.y_train == 0) / np.sum(self.y_train == 1)

        # Hyperparameter grid
        param_grid = {
            'iterations': [100, 200, 300],
            'learning_rate': [0.05, 0.1],
            'depth': [4, 6, 8],
            'l2_leaf_reg': [1, 3, 5],
            'border_count': [64, 128]
        }
        
        catboost = CatBoostClassifier(
            loss_function='Logloss',
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=50,
            thread_count=-1
        )
        
        # Randomized search with cross-validation
        random_search = RandomizedSearchCV(
            catboost,
            param_distributions=param_grid,
            n_iter=20,  # Try 50 random combinations
            cv=3,
            scoring='f1',
            n_jobs=1,
            verbose=1,
            random_state=42
        )
        random_search.fit(self.X_train, self.y_train, cat_features=self.categorical_features)
        
        best_catboost = random_search.best_estimator_
        print(f"\nBest parameters: {random_search.best_params_}")

        train_pool = Pool(self.X_train, self.y_train, cat_features=self.categorical_features)
        valid_pool = Pool(self.X_test, self.y_test, cat_features=self.categorical_features)

        best_catboost.fit(train_pool, eval_set=valid_pool)

        # Predictions
        y_pred = best_catboost.predict(self.X_test)
        y_pred_proba = best_catboost.predict_proba(self.X_test)[:, 1]
        
        # Store model and results
        self.models['CatBoost'] = best_catboost
        self.results['CatBoost'] = {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'best_params': random_search.best_params_
        }

        # Store evals_result_
        self.results['CatBoost']['evals_result'] = best_catboost.get_evals_result()
        
        self._print_metrics('CatBoost', y_pred, y_pred_proba)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': best_catboost.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        return self
    
    def train_xgboost(self):
        """Train XGBoost classifier with hyperparameter tuning."""
        print("\n" + "="*60)
        print("Training XGBoost Classifier...")
        print("="*60)

        le = LabelEncoder()
        y_train_enc = le.fit_transform(self.y_train)
        y_test_enc = le.transform(self.y_test)
        self.label_encoder = le

        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = np.sum(self.y_train == 0) / np.sum(self.y_train == 1)
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 400],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'gamma': [0, 0.1, 0.2]
        }

        
        xgb_clf = xgb.XGBClassifier(
            objective='binary:logistic',
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )

        # Randomized search with cross-validation
        random_search = RandomizedSearchCV(
            xgb_clf,
            param_distributions=param_grid,
            n_iter=20,
            cv=3,
            scoring='f1',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )

        random_search.fit(self.X_train, self.y_train)
        
        best_xgb = random_search.best_estimator_
        print(f"\nBest parameters: {random_search.best_params_}")

        best_xgb.fit(
            self.X_train, y_train_enc,
            eval_set=[(self.X_train, y_train_enc), (self.X_test, y_test_enc)]
        )
        
        # Predictions
        y_pred = best_xgb.predict(self.X_test)
        y_pred_proba = best_xgb.predict_proba(self.X_test)[:, 1]
        
        # Store model and results
        self.models['XGBoost'] = best_xgb
        self.results['XGBoost'] = {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'best_params': random_search.best_params_
        }

        # Store evals_result_
        self.results['XGBoost']['evals_result'] = best_xgb.evals_result()
        
        self._print_metrics('XGBoost', y_pred, y_pred_proba)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': best_xgb.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        return self
    
    def _print_metrics(self, model_name, y_pred, y_pred_proba):
        """Print evaluation metrics for a model."""
        print(f"\n{model_name} Results:")
        print("-" * 40)
        
        acc = accuracy_score(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"Accuracy: {acc:.4f}")
        print(f"ROC-AUC: {auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, 
                                   target_names=['Benign', 'Attack']))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))
    
    def compare_models(self):
        """Compare all trained models and select the best one."""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        comparison = []
        for model_name, results in self.results.items():
            y_pred = results['predictions']
            y_pred_proba = results['probabilities']
            
            acc = accuracy_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Calculate F1 scores
            precision, recall, f1, _ = precision_recall_fscore_support(
                self.y_test, y_pred, average='binary'
            )
            
            comparison.append({
                'Model': model_name,
                'Accuracy': acc,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC-AUC': auc
            })
        
        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        
        print("\n" + comparison_df.to_string(index=False))
        
        best_model_name = comparison_df.iloc[0]['Model']
        print(f"\nBEST MODEL: {best_model_name}")
        print(f"   F1-Score: {comparison_df.iloc[0]['F1-Score']:.4f}")
        print(f"   ROC-AUC: {comparison_df.iloc[0]['ROC-AUC']:.4f}")
        
        return best_model_name, comparison_df
    
    def plot_results(self):
        """Plot confusion matrices and performance comparison for all models."""

        n_models = len(self.results)
        n_cols = 3
        n_rows = (n_models + 1 + n_cols - 1) // n_cols  # +1 for comparison chart

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))

        # Flatten axes robustly
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]

        # --- Plot confusion matrices ---
        for idx, (model_name, results) in enumerate(self.results.items()):
            if idx < len(axes):
                cm = confusion_matrix(self.y_test, results['predictions'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
                axes[idx].set_title(f'{model_name} - Confusion Matrix')
                axes[idx].set_ylabel('True Label')
                axes[idx].set_xlabel('Predicted Label')

        # --- Model comparison bar chart ---
        metrics = []
        for model_name, results in self.results.items():
            y_pred = results['predictions']
            y_proba = results.get('probabilities', None)
            acc = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_proba) if y_proba is not None else np.nan
            metrics.append({'Model': model_name, 'Accuracy': acc, 'F1-Score': f1, 'ROC-AUC': auc})

        metrics_df = pd.DataFrame(metrics)
        x = np.arange(len(metrics_df))
        width = 0.25

        comparison_idx = len(self.results)
        if comparison_idx < len(axes):
            ax = axes[comparison_idx]
            ax.bar(x - width, metrics_df['Accuracy'], width, label='Accuracy')
            ax.bar(x, metrics_df['F1-Score'], width, label='F1-Score')
            ax.bar(x + width, metrics_df['ROC-AUC'], width, label='ROC-AUC')
            ax.set_xlabel('Model')
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics_df['Model'], rotation=45, ha='right')
            ax.legend()
            ax.set_ylim([0, 1])

        # --- Hide unused axes ---
        for idx in range(len(self.results) + 1, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("\nResults saved to 'model_comparison.png'")
        plt.show()

    
    def plot_nn_training(self):
        """
        Plot both the average K-Fold and final training curves for the Neural Network.
        Shows loss and accuracy evolution over epochs.

        - Averages the per-fold histories (train and validation metrics)
        - Plots the final model's training curve (if available)
        """
        if 'SimpleNN' not in self.results:
            print("No neural network results found.")
            return

        results = self.results['SimpleNN']
        histories = results.get('cv_histories', [])
        final_history = results.get('final_history', None)

        if not histories and final_history is None:
            print("No training histories available to plot.")
            return

        plt.figure(figsize=(12, 5))

        # --------------------------------------------
        # Compute averaged K-Fold curves (if available)
        # --------------------------------------------
        if histories:
            # Align folds to same length by trimming to min epoch length
            min_len = min(len(h['loss']) for h in histories)
            avg_loss = np.mean([h['loss'][:min_len] for h in histories], axis=0)
            avg_acc = np.mean([h['accuracy'][:min_len] for h in histories], axis=0)

            val_loss, val_acc = None, None
            if all('val_loss' in h for h in histories):
                val_loss = np.mean([h['val_loss'][:min_len] for h in histories], axis=0)
            if all('val_accuracy' in h for h in histories):
                val_acc = np.mean([h['val_accuracy'][:min_len] for h in histories], axis=0)

            plt.subplot(1, 2, 1)
            plt.plot(avg_loss, label='K-Fold Train Loss', color='tab:blue')
            if val_loss is not None:
                plt.plot(val_loss, label='K-Fold Val Loss', color='tab:orange')
            plt.title('Average K-Fold Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(avg_acc, label='K-Fold Train Acc', color='tab:blue')
            if val_acc is not None:
                plt.plot(val_acc, label='K-Fold Val Acc', color='tab:orange')
            plt.title('Average K-Fold Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

        # --------------------------------------------
        # Overlay Final Model Curves (if available)
        # --------------------------------------------
        if final_history is not None:
            if 'loss' in final_history:
                plt.subplot(1, 2, 1)
                plt.plot(final_history['loss'], '--', label='Final Train Loss', color='tab:green')
                if 'val_loss' in final_history:
                    plt.plot(final_history['val_loss'], '--', label='Final Val Loss', color='tab:red')

            if 'accuracy' in final_history:
                plt.subplot(1, 2, 2)
                plt.plot(final_history['accuracy'], '--', label='Final Train Acc', color='tab:green')
                if 'val_accuracy' in final_history:
                    plt.plot(final_history['val_accuracy'], '--', label='Final Val Acc', color='tab:red')

        plt.tight_layout()
        plt.savefig('nn_training_curves.png', dpi=300, bbox_inches='tight')
        print("\nNeural Network training curves saved to 'nn_training_curves.png'")
        plt.show()


    def plot_boosting_training(self, model_name):
        print(f"\n{'='*60}\nPlotting Training Curves for {model_name}\n{'='*60}")

        if model_name not in self.results:
            print(f"Model '{model_name}' not found. Train it first.")
            return

        evals_result = self.results[model_name].get('evals_result', None)
        if evals_result is None:
            print(f"No evals_result found for {model_name}.")
            return

        if model_name == 'LightGBM':
            train_key = 'train'
            valid_key = 'valid'
        elif model_name == 'XGBoost':
            train_key = 'validation_0'
            valid_key = 'validation_1'
        else:
            train_key = 'learn'
            valid_key = 'validation'

        metrics = list(evals_result[train_key].keys())

        for metric in metrics:
            plt.figure(figsize=(8, 5))
            plt.plot(evals_result[train_key][metric], label='Train')
            plt.plot(evals_result[valid_key][metric], label='Validation')
            plt.title(f"{model_name} {metric} over epochs")
            plt.xlabel("Iteration")
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show()

    # --- Evaluates performance vs training set size ---
    def plot_learning_curve(self, model_name, X, y):
        """Plot learning curve by constructing the model inside the function."""
        
        # --- Construct the model according to the name ---
        if model_name == "LightGBM":
            model = lgb.LGBMClassifier(
                n_estimators=1000, learning_rate=0.05, 
                num_leaves=31, random_state=42
            )
        elif model_name == "XGBoost":
            model = xgb.XGBClassifier(
                n_estimators=1000, learning_rate=0.05,
                max_depth=6, use_label_encoder=False,
                eval_metric='logloss', random_state=42
            )
        elif model_name == "CatBoost":
            model = CatBoostClassifier(
                iterations=1000, learning_rate=0.05, 
                depth=6, verbose=0, random_seed=42
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # --- Compute learning curve ---
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=3, scoring='f1', n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5), random_state=42
        )
        
        train_mean = train_scores.mean(axis=1)
        test_mean = test_scores.mean(axis=1)

        # --- Plot ---
        plt.figure(figsize=(8, 5))
        plt.plot(train_sizes, train_mean, 'o-', label='Training F1')
        plt.plot(train_sizes, test_mean, 'o-', label='Validation F1')
        plt.title(f"Learning Curve - {model_name}")
        plt.xlabel("Training examples")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    
    def load_models(self, model_path):
        """Load all saved models and scaler, then generate predictions for comparison."""
        print("\nLoading saved models and scaler...")
        
        # Load scaler
        scaler_path = os.path.join(model_path, 'standard_scaler.joblib')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            print("Scaler loaded.")
        else:
            print("Warning: Scaler not found, neural network results may be invalid.")

        # Load models
        nn_path = os.path.join(model_path, 'SimpleNN.h5')
        lgb_path = os.path.join(model_path, 'LightGBM.joblib')
        xgb_path = os.path.join(model_path, 'XGBoost.joblib')
        cat_path = os.path.join(model_path, 'CatBoost.txt')

        if os.path.exists(nn_path):
            nn = load_model(nn_path)
            self.models['SimpleNN'] = nn
            print("Neural Network loaded.")

            # Predict using scaled test set
            y_pred_proba = nn.predict(self.X_test_scaled).ravel()
            y_pred = (y_pred_proba > 0.5).astype(int)
            self.results['SimpleNN'] = {
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'history': None
            }

        if os.path.exists(lgb_path):
            lgbm = joblib.load(lgb_path)
            self.models['LightGBM'] = lgbm
            print("LightGBM loaded.")
            y_pred = lgbm.predict(self.X_test)
            y_pred_proba = lgbm.predict_proba(self.X_test)[:, 1]
            self.results['LightGBM'] = {'predictions': y_pred, 'probabilities': y_pred_proba}

        if os.path.exists(xgb_path):
            xgb_model = joblib.load(xgb_path)
            self.models['XGBoost'] = xgb_model
            print("XGBoost loaded.")
            y_pred = xgb_model.predict(self.X_test)
            y_pred_proba = xgb_model.predict_proba(self.X_test)[:, 1]
            self.results['XGBoost'] = {'predictions': y_pred, 'probabilities': y_pred_proba}

        if os.path.exists(cat_path):
            cat_model = CatBoostClassifier()
            cat_model.load_model(cat_path)
            self.models['CatBoost'] = cat_model
            print("CatBoost loaded.")
            y_pred = cat_model.predict(self.X_test)
            y_pred_proba = cat_model.predict_proba(self.X_test)[:, 1]
            self.results['CatBoost'] = {'predictions': y_pred, 'probabilities': y_pred_proba}

        print("\nAll available models loaded and evaluated on test set.")


    def save_model(self, model_name):
        """Save the best model to disk."""
        model = self.models[model_name]
        
        if model_name == 'SimpleNN':
            model.save(f'{model_name}.h5')
            print(f"\nModel saved to '{model_name}.h5'")
            joblib.dump(self.scaler, 'standard_scaler.joblib')
            print("Scaler saved to 'standard_scaler.joblib'")
        elif model_name == 'CatBoost':
            model.save_model(f'{model_name}.txt')
            print(f"\nModel saved to '{model_name}.txt'")
        # Works for LightGBM and XGBoost
        else:
            joblib.dump(model, f'{model_name}.joblib')
            print(f"\nModel saved to '{model_name}.joblib'")


# --- Configuration ---
# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

extract_dir = "extracted_dataset"
DATA_PATH = os.path.join(extract_dir, "sorted_ds_with_labels.parquet")
MODEL_PATH = "models"
MODELS_TO_USE = ["LightGBM", "XGBoost", "CatBoost"]
USE_RANDOM_SAMPLE = False  # True -> random sampling, False -> fixed/balanced
TRAIN_MODELS = False       # True -> train models, False -> load pre-trained models

# --- Initialize classifier ---
classifier = WazuhLogClassifier(DATA_PATH)

# --- Load and preprocess ---
classifier.load_and_preprocess_random_data() if USE_RANDOM_SAMPLE else classifier.load_and_preprocess_fixed_data()

# --- Feature engineering ---
X, y = classifier.engineer_features()

# --- Split and scale ---
classifier.split_and_scale_data(X, y)

# --- Train or load models ---
if TRAIN_MODELS:
    if "LightGBM" in MODELS_TO_USE:
        classifier.train_lightgbm()
        classifier.save_model("LightGBM")
    if "XGBoost" in MODELS_TO_USE:
        classifier.train_xgboost()
        classifier.save_model("XGBoost")
    if "CatBoost" in MODELS_TO_USE:
        classifier.train_catboost()
        classifier.save_model("CatBoost")
    if "SimpleNN" in MODELS_TO_USE:
        classifier.train_neural_network(epochs=10, batch_size=32)
        classifier.save_model("SimpleNN")     
else:
    classifier.load_models(MODEL_PATH)

# --- Plot boosting training curves ---
# for model_name in MODELS_TO_USE:
#    classifier.plot_boosting_training(model_name)

# --- Plot learning curves ---
for model_name in MODELS_TO_USE:
    print(f"\n{'='*60}\nLearning Curve for {model_name}\n{'='*60}")
    classifier.plot_learning_curve(model_name, X, y)

# --- Compare models and plot results ---
best_model_name, comparison_df = classifier.compare_models()
classifier.plot_results()
print(f"\nBest model: {best_model_name}")