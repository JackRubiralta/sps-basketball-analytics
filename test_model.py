import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss, roc_auc_score,
    confusion_matrix, classification_report
)

from model import Model
from games_data import GamesData

def main():
    # 1) Instantiate your GamesData (the same one used for training)
    training_csv_path = "games_data.csv"
    games_data = GamesData(training_csv_path)

    # 2) Create the Model
    model_file_path = "model_stuff.json"
    my_model = Model(model_file_path=model_file_path, games_data=games_data)
    if my_model.model is None:
        raise ValueError("No model loaded. Please train or check the path.")

    # -----------------------------------------
    # Example 1: Single-Game "manual" prediction
    # -----------------------------------------
    prob = my_model.predict_single_game(
        team_home="lsu_tigers",
        team_away="georgia_lady_bulldogs",
        home_rest_days=3.0,
        home_travel_dist=100.0,
        away_rest_days=2.0,
        away_travel_dist=200.0
    )
    print(f"\nSingle-Game Prediction: Probability home wins = {prob:.4f}")

    # -----------------------------------------
    # Example 2: Evaluate entire test CSV
    # -----------------------------------------
    test_csv_path = "games_data_testing.csv"
    if not os.path.exists(test_csv_path):
        print(f"\nNo {test_csv_path} found, skipping batch evaluation.")
        return

    test_df = pd.read_csv(test_csv_path)
    print(f"\nLoaded {len(test_df)} rows from {test_csv_path}...")

    # We'll build predictions row by row
    y_probs = []
    y_true = []

    # We'll check if final scores exist for metrics
    has_scores = ("team_score_Home" in test_df.columns
                  and "team_score_Away" in test_df.columns)

    for _, row in test_df.iterrows():
        # Pull out the needed columns
        # e.g. 'team_home', 'team_away', 'home_rest_days', 'away_rest_days', etc.
        team_home = row.get("team_home", "")
        team_away = row.get("team_away", "")

        # You might name them "home_rest_days" in your CSV or "rest_days_Home".
        # We'll try both, defaulting to 0.0 if missing.
        home_rest_days = row.get("home_rest_days", row.get("rest_days_Home", 0.0))
        away_rest_days = row.get("away_rest_days", row.get("rest_days_Away", 0.0))
        home_travel_dist = row.get("home_travel_dist", row.get("travel_dist_Home", 0.0))
        away_travel_dist = row.get("away_travel_dist", row.get("travel_dist_Away", 0.0))

        # Call single-game prediction
        prob_home_win = my_model.predict_single_game(
            team_home=team_home,
            team_away=team_away,
            home_rest_days=home_rest_days,
            home_travel_dist=home_travel_dist,
            away_rest_days=away_rest_days,
            away_travel_dist=away_travel_dist
        )
        y_probs.append(prob_home_win)

        # If we have actual scores, figure out if the home team won
        if has_scores:
            home_score = row["team_score_Home"]
            away_score = row["team_score_Away"]
            actual_home_win = 1 if home_score > away_score else 0
            y_true.append(actual_home_win)

    y_probs = np.array(y_probs)

    # -------------------------
    # Evaluate if we have actual scores
    # -------------------------
    if has_scores and len(y_true) == len(y_probs):
        y_true = np.array(y_true)
        # Convert probabilities to 0/1 predictions at threshold=0.5
        y_pred = (y_probs >= 0.5).astype(int)

        # Classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        brier    = brier_score_loss(y_true, y_probs)
        ll       = log_loss(y_true, np.column_stack([1 - y_probs, y_probs]))
        auc      = roc_auc_score(y_true, y_probs)
        cm       = confusion_matrix(y_true, y_pred)
        report   = classification_report(y_true, y_pred, digits=4)

        print("\n=== MODEL PERFORMANCE (TEST) ===")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Brier:     {brier:.4f}")
        print(f"Log Loss:  {ll:.4f}")
        print(f"ROC AUC:   {auc:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(report)

        # -------------
        # Plot graphs
        # -------------
        # 1) Probability vs. actual point diff (if "home_team_score" in test_df)
        actual_diff = test_df["team_score_Home"] - test_df["team_score_Away"]

        plt.figure(figsize=(12, 5))

        # Plot 1: Probability vs. Actual Point Difference
        plt.subplot(1, 2, 1)
        plt.scatter(y_probs, actual_diff, alpha=0.5)
        plt.xlabel("Predicted Probability (Home Win)")
        plt.ylabel("Actual Home Team Point Diff")
        plt.title("Prediction vs. Actual Point Diff")
        plt.grid(True)

        # Plot 2: Calibration Plot
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_probs, bins) - 1

        actual_win_rates = []
        mean_predicted = []

        for i in range(n_bins):
            idxs = np.where(bin_indices == i)[0]
            if len(idxs) > 0:
                bin_mean_prob = np.mean(y_probs[idxs])
                bin_actual_win = np.mean(y_true[idxs])
            else:
                bin_mean_prob = np.nan
                bin_actual_win = np.nan

            mean_predicted.append(bin_mean_prob)
            actual_win_rates.append(bin_actual_win)

        plt.subplot(1, 2, 2)
        plt.plot(mean_predicted, actual_win_rates, 'o-', label="Actual vs. Predicted")
        plt.plot([0, 1], [0, 1], 'k--', label="Perfect Calibration")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Actual Home Win Rate")
        plt.title("Calibration Plot")
        plt.legend(loc="best")
        plt.grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
