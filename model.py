import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score

import xgboost as xgb
from xgboost import XGBClassifier

from games_data import GamesData

class Model:
    def __init__(self, model_file_path: str, games_data: GamesData):
        """
        :param model_file_path: path to save or load model via joblib
        :param games_data: the GamesData instance providing features
        """
        self.model_file_path = model_file_path
        self.model = None
        self.games_data = games_data

        if os.path.exists(self.model_file_path):
            self.load_model()
        else:
            print(f"No existing model found at {self.model_file_path}. "
                  "Need to train a new one.")

    def generate_model(self, do_hyperparam_search=True):
        """
        Train stacking model with possible hyperparameter search
        for XGB and RandomForest.
        """
        # 1) Prepare data
        merged_df = self.games_data.prepare_training_data()
        X, y = self.games_data.get_feature_and_label_arrays(merged_df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # 2) Define base estimators
        xgb_clf = XGBClassifier(
            eval_metric='logloss',
            random_state=42,
        )
        rf_clf = RandomForestClassifier(random_state=42)
        meta_learner = LogisticRegression(max_iter=2000)

        # Optional hyperparam search
        if do_hyperparam_search:
            # Hyperparameter search for XGB
            xgb_params = {
                "n_estimators":    [100, 200, 300, 500],
                "max_depth":       [3, 4, 6, 8],
                "learning_rate":   [0.01, 0.05, 0.1],
                "subsample":       [0.6, 0.8, 1.0],
                "colsample_bytree":[0.6, 0.8, 1.0],
                "gamma":           [0, 0.1, 0.2],
                "reg_lambda":      [1, 2, 5],
                "reg_alpha":       [0, 0.1, 1]
            }
            xgb_search = RandomizedSearchCV(
                estimator=xgb_clf,
                param_distributions=xgb_params,
                n_iter=10,  # can increase for more thorough search
                scoring='roc_auc',
                cv=3,
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
            xgb_search.fit(X_train, y_train)
            xgb_clf = xgb_search.best_estimator_
            print("Best XGB params:", xgb_search.best_params_)

            # Hyperparameter search for RF
            rf_params = {
                "n_estimators": [100, 200, 500],
                "max_depth":    [None, 5, 10, 20],
                "max_features": ["sqrt", "log2", None]
            }
            rf_search = RandomizedSearchCV(
                estimator=rf_clf,
                param_distributions=rf_params,
                n_iter=5,
                scoring='roc_auc',
                cv=3,
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
            rf_search.fit(X_train, y_train)
            rf_clf = rf_search.best_estimator_
            print("Best RF params:", rf_search.best_params_)

        # 3) Build stacking classifier
        estimators = [
            ('xgb', xgb_clf),
            ('rf', rf_clf)
        ]
        stack_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=5,
            n_jobs=-1,
            passthrough=False
        )

        # 4) Train stacking
        print("\nTraining Stacking Classifier...")
        stack_clf.fit(X_train, y_train)
        self.model = stack_clf

        # 5) Evaluate on test set
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        ll = log_loss(y_test, self.model.predict_proba(X_test))
        try:
            brier = brier_score_loss(y_test, y_proba)
        except:
            brier = None

        print(f"\n=== MODEL PERFORMANCE ON HOLD-OUT TEST ===")
        print(f"Accuracy:  {acc:.4f}")
        print(f"ROC AUC:   {auc:.4f}")
        print(f"Log Loss:  {ll:.4f}")
        if brier is not None:
            print(f"Brier:     {brier:.4f}")

        return self.model

    def predict_single_game(self,
                            team_home: str,
                            team_away: str,
                            home_rest_days: float,
                            home_travel_dist: float,
                            away_rest_days: float,
                            away_travel_dist: float) -> float:
        """
        For a single matchup, build the feature vector (including difference/ratio),
        then return the predicted probability that the home team wins.
        """
        if self.model is None:
            raise ValueError("No model loaded or trained. Please train or load a model first.")

        X_single = self.games_data.build_prediction_features(
            team_home, team_away,
            home_rest_days, home_travel_dist,
            away_rest_days, away_travel_dist
        )
        prob = self.model.predict_proba(X_single)[:, 1][0]
        return prob

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        For batch predictions where X is already constructed,
        returns predicted probabilities for the home team.
        """
        if self.model is None:
            raise ValueError("No model found. Please train or load a model.")
        return self.model.predict_proba(X)[:, 1]

    def save_model(self):
        """
        Saves the trained model via joblib.
        """
        if self.model is None:
            print("No trained model to save.")
            return
        joblib.dump(self.model, self.model_file_path)
        print(f"Model saved to '{self.model_file_path}'")

    def load_model(self):
        """
        Loads the model from joblib if it exists.
        """
        self.model = joblib.load(self.model_file_path)
        print(f"Model loaded from '{self.model_file_path}'")
