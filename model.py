import os
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier
import joblib  # for saving and loading scikit-learn models

from games_data import GamesData  # We'll reference your new class here

class Model:
    def __init__(self, model_file_path: str, games_data: GamesData):
        """
        We store a reference to the GamesData instance for use in training/predicting.
        If model_file_path exists, load the saved model from that path.
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

    def generate_model(self, do_hyperparam_search=True):
        """
        Build and train an ensemble model using stacking and hyperparameter tuning.
        Parameters:
            do_hyperparam_search (bool): If True, perform hyperparameter search for base models.
        Returns:
            Trained stacking ensemble model.
        """
        # Prepare training data using GamesData
        merged_df = self.games_data.prepare_training_data()
        X, y = self.games_data.get_feature_and_label_arrays(merged_df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Define base estimators for stacking
        xgb_clf = XGBClassifier(eval_metric='logloss', random_state=42)
        rf_clf  = RandomForestClassifier(random_state=42)
        # Meta-learner (level-1 model)
        meta_learner = LogisticRegression(max_iter=1000)

        if do_hyperparam_search:
            # Hyperparameter grid for XGBoost
            xgb_param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }
            xgb_search = RandomizedSearchCV(
                estimator=XGBClassifier(eval_metric='logloss', random_state=42),
                param_distributions=xgb_param_grid,
                n_iter=20,
                scoring='roc_auc',
                cv=3,
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
            xgb_search.fit(X_train, y_train)
            xgb_clf = xgb_search.best_estimator_

            # Hyperparameter grid for Random Forest
            rf_param_grid = {
                'n_estimators': [100, 200, 500],
                'max_depth': [None, 5, 10, 20],
                'max_features': ['sqrt', 'log2', None]
            }
            rf_search = RandomizedSearchCV(
                estimator=RandomForestClassifier(random_state=42),
                param_distributions=rf_param_grid,
                n_iter=10,
                scoring='roc_auc',
                cv=3,
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
            rf_search.fit(X_train, y_train)
            rf_clf = rf_search.best_estimator_

            meta_learner = LogisticRegression(max_iter=1000)

        # Create the stacking ensemble classifier
        estimators = [
            ('xgb', xgb_clf),
            ('rf', rf_clf)
        ]
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=5,
            n_jobs=-1,
            passthrough=False
        )

        # Train the stacking ensemble on the training data
        stacking_clf.fit(X_train, y_train)
        self.model = stacking_clf

        # Evaluate the model on test set with various metrics
        y_pred = self.model.predict(X_test)
        if len(np.unique(y_test)) == 2:
            y_proba = self.model.predict_proba(X_test)[:, 1]
        else:
            y_proba = self.model.predict_proba(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) == 2 else None
        logloss = log_loss(y_test, self.model.predict_proba(X_test))
        brier = None
        if len(np.unique(y_test)) == 2:
            try:
                brier = brier_score_loss(y_test, y_proba)
            except Exception:
                brier = None

        print(f"Accuracy: {accuracy:.4f}")
        if auc is not None:
            print(f"ROC AUC: {auc:.4f}")
        else:
            print("ROC AUC: N/A (multi-class case)")
        print(f"Log Loss: {logloss:.4f}")
        if brier is not None:
            print(f"Brier Score: {brier:.4f}")
        else:
            print("Brier Score: N/A (only for binary classification)")

        return self.model

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
        Saves the model to self.model_file_path using joblib.
        """
        if self.model is None:
            print("No model found to save.")
            return
        joblib.dump(self.model, self.model_file_path)
        print(f"Model saved to {self.model_file_path}")

    def load_model(self):
        """
        Loads a model from self.model_file_path using joblib.
        """
        self.model = joblib.load(self.model_file_path)
        print(f"Model loaded from {self.model_file_path}")
