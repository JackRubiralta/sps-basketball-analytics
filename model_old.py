# model.py

import os
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

from games_data import GamesData  # We'll reference your new class here

class Model:
    def __init__(self, model_file_path: str, games_data: GamesData):
        """
        We store a reference to the GamesData instance for use in training/predicting.
        If model_file_path exists, load the XGBoost model from that path.
        Otherwise, self.model = None until we call 'generate_model' or 'load_model'.
        """
        self.model_file_path = model_file_path
        self.model = None
        self.games_data = games_data  # reference to your custom data class

        if os.path.exists(self.model_file_path):
            self.load_model()
        else:
            print(f"Model file not found at {self.model_file_path}. "
                  f"Will need to generate a new model or load manually.")

    def hyperparam_search(self, X, y, cv=3, n_iter=40, random_state=42):
        """
        Uses RandomizedSearchCV to find good hyperparameters for XGBoost.
        Returns the best parameter dictionary found.
        """
        param_dist = {
            "learning_rate":   [0.01, 0.02, 0.05, 0.1],
            "max_depth":       [4, 6, 8, 10],
            "n_estimators":    [100, 200, 300, 500, 800],
            "subsample":       [0.5, 0.7, 0.8, 1.0],
            "colsample_bytree":[0.5, 0.7, 0.8, 1.0],
            "gamma":           [0, 0.1, 0.2, 0.5],
            "reg_lambda":      [1, 2, 5, 10],
            "reg_alpha":       [0, 0.1, 1, 2]
        }

        xgb_model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=random_state
        )

        search = RandomizedSearchCV(
            xgb_model,
            param_distributions=param_dist,
            scoring='neg_log_loss',
            n_iter=n_iter,
            cv=cv,
            verbose=1,
            random_state=random_state
        )

        print("Starting hyperparameter search...")
        search.fit(X, y)
        print("Best params found:", search.best_params_)
        return search.best_params_

    def generate_model(self, do_hyperparam_search=True, **train_params):
        """
        Creates and trains a new XGBoost model.
        We do NOT take X_train or y_train as parameters because we can get them
        directly from our GamesData instance (self.games_data).
        """
        # 1) Prepare the training data
        merged_df = self.games_data.prepare_training_data()
        X, y = self.games_data.get_feature_and_label_arrays(merged_df)

        # We'll do a train/val split here or you can do it differently
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 2) Possibly do hyperparameter search
        best_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_jobs": -1
        }
        if do_hyperparam_search:
            if len(X_val) > 0:
                # Combine train+val for search or do cross-val
                X_search = np.concatenate([X_train, X_val])
                y_search = np.concatenate([y_train, y_val])
                found_params = self.hyperparam_search(X_search, y_search)
            else:
                found_params = self.hyperparam_search(X_train, y_train)
            best_params.update(found_params)
        else:
            best_params.update(train_params)

        # 3) Initialize final model
        self.model = xgb.XGBClassifier(**best_params)

        # 4) Fit
        if len(X_val) > 0:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=True
            )
        else:
            self.model.fit(X_train, y_train, verbose=True)

    def predict_single_game(self,
                            team_home: str,
                            team_away: str,
                            home_rest_days: float,
                            home_travel_dist: float,
                            away_rest_days: float,
                            away_travel_dist: float) -> float:
        """
        For a single matchup, build the feature vector using self.games_data,
        then return the predicted probability that the home team wins.
        """
        if self.model is None:
            raise ValueError("No model loaded or trained. Please call generate_model() or load_model().")

        # Build a 1-row feature vector
        X_single = self.games_data.build_prediction_features(
            team_home, team_away,
            home_rest_days, home_travel_dist,
            away_rest_days, away_travel_dist
        )
        prob_home_win = self.model.predict_proba(X_single)[:, 1][0]
        return prob_home_win

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        For batch predictions, you already have feature vectors in shape (N, num_features).
        Return array of probabilities.
        """
        if self.model is None:
            raise ValueError("No model found. Train or load a model before predicting.")
        return self.model.predict_proba(X)[:, 1]

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
