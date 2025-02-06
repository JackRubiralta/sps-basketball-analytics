# model.py

import os
import xgboost as xgb
import numpy as np

class Model:
    def __init__(self, model_file_path: str):
        """
        If model_file_path exists, load the XGBoost model from that path.
        Otherwise, keep self.model = None.
        """
        self.model_file_path = model_file_path
        self.model = None

        if os.path.exists(self.model_file_path):
            self.load_model()
        else:
            print(f"Model file not found at {self.model_file_path}. "
                  f"Will need to generate a new model or load manually.")

    def generate_model(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray = None, y_val: np.ndarray = None,
                       **train_params):
        """
        Create and train a new XGBoost model using the given training data.
        Optionally pass separate validation data and additional training parameters.
        """
        # You can tune hyperparameters here
        # Example: default or some optimized parameters
        params = {
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 200,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "use_label_encoder": False,
            "early_stopping_rounds": 10

        }
        # Overwrite defaults with anything in train_params
        params.update(train_params)

        self.model = xgb.XGBClassifier(**params)

        if X_val is not None and y_val is not None:
            self.model.fit(X_train, y_train,
                           eval_set=[(X_val, y_val)],
                           verbose=True)
        else:
            self.model.fit(X_train, y_train)

    def save_model(self):
        """
        Saves the model to self.model_file_path.
        """
        if self.model is None:
            print("No model found to save.")
            return

        self.model.save_model(self.model_file_path)
        print(f"Model saved to {self.model_file_path}")

    def load_model(self):
        """
        Loads a model from self.model_file_path.
        """
        self.model = xgb.XGBClassifier()
        self.model.load_model(self.model_file_path)
        print(f"Model loaded from {self.model_file_path}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return predicted probabilities of the positive class (e.g., home team winning).
        """
        if self.model is None:
            raise ValueError("No model found. Train or load a model before predicting.")
        # For a binary:logistic objective, predict_proba gives [prob_neg, prob_pos]
        prob_pos = self.model.predict_proba(X)[:, 1]
        return prob_pos
