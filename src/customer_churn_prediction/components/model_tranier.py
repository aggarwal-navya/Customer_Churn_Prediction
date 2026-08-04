import os
import sys
import json
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlparse
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.customer_churn_prediction.utils import save_object, load_object

class ModelTrainer:
    def __init__(self):
        self.model_file_path = "models/churn_model.pkl"
        self.metrics_file_path = "models/metrics.json"
        self.columns_file_path = "models/model_columns.pkl"
        self.feature_importance_file_path = "models/feature_importance.json"

    def _uri_to_local_path(self, uri):
        parsed = urlparse(uri)
        if parsed.scheme != "file":
            return None

        path = unquote(parsed.path)
        if os.name == "nt" and path.startswith("/"):
            path = path[1:]
        return path

    def _is_writable_dir(self, path):
        try:
            os.makedirs(path, exist_ok=True)
            probe = os.path.join(path, ".mlflow_write_probe")
            with open(probe, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(probe)
            return True
        except Exception:
            return False

    def _configure_mlflow(self):
        tracking_db = Path("mlflow.db").resolve()
        local_artifacts_dir = Path("mlruns").resolve()
        local_artifact_uri = local_artifacts_dir.as_uri()

        mlflow.set_tracking_uri(f"sqlite:///{tracking_db.as_posix()}")

        base_name = "Telco_Customer_Churn_Prediction"
        experiment = mlflow.get_experiment_by_name(base_name)

        if experiment is None:
            exp_id = mlflow.create_experiment(base_name, artifact_location=local_artifact_uri)
            mlflow.set_experiment(experiment_id=exp_id)
            return

        artifact_path = self._uri_to_local_path(experiment.artifact_location)
        if artifact_path and self._is_writable_dir(artifact_path):
            mlflow.set_experiment(experiment_name=base_name)
            return

        # Existing experiment points to a non-writable location (common after copying repos between machines).
        fallback_name = f"{base_name}_local_{datetime.now().strftime('%Y%m%d')}"
        fallback = mlflow.get_experiment_by_name(fallback_name)
        if fallback is None:
            exp_id = mlflow.create_experiment(fallback_name, artifact_location=local_artifact_uri)
            mlflow.set_experiment(experiment_id=exp_id)
        else:
            mlflow.set_experiment(experiment_name=fallback_name)

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            os.makedirs(os.path.dirname(self.model_file_path), exist_ok=True)
            
            # Gradient Boosting parameters matching the notebook tuning
            params = {
                'n_estimators': 150,
                'max_depth': 4,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'random_state': 42
            }

            # MLflow configuration
            self._configure_mlflow()

            with mlflow.start_run() as run:
                model = GradientBoostingClassifier(**params)
                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]

                # Calculate metrics
                metrics = {
                    'accuracy': float(accuracy_score(y_test, y_pred)),
                    'precision': float(precision_score(y_test, y_pred)),
                    'recall': float(recall_score(y_test, y_pred)),
                    'f1': float(f1_score(y_test, y_pred)),
                    'roc_auc': float(roc_auc_score(y_test, y_prob))
                }

                # Log parameters and metrics to MLflow
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, name="model")

                # Save model to disk
                save_object(self.model_file_path, model)

                # Save metrics to json file for Flask access
                with open(self.metrics_file_path, 'w') as f:
                    json.dump(metrics, f, indent=2)

                # Extract and save feature importances
                try:
                    columns = load_object(self.columns_file_path)
                    importances = model.feature_importances_.tolist()
                    feat_imp = sorted(zip(columns, importances), key=lambda x: x[1], reverse=True)[:5]
                    with open(self.feature_importance_file_path, 'w') as f:
                        json.dump(feat_imp, f, indent=2)
                except Exception as ex:
                    print(f"Warning: Failed to save feature importances: {ex}")

                return metrics
        except Exception as e:
            raise Exception(f"Model training failed: {e}")
