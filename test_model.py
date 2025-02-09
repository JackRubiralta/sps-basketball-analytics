import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)

from model import Model
from games_data import GamesData

def main():
    # 1) Create the GamesData (for consistent feature engineering)
    training_csv_path = "games_data.csv"
    games_data = GamesData(training_csv_path)

    # 2) Load the pre-trained Stacking Model
    model_file_path = "model_stuff.json"
    my_model = Model(model_file_path=model_file_path, games_data=games_data)
    if my_model.model is None:
        raise ValueError(f"No model loaded from {model_file_path}.")

    # 3) Example single-game prediction
    prob = my_model.predict_single_game(
        team_home="lsu_tigers",
        team_away="georgia_lady_bulldogs",
        home_rest_days=3.0,
        home_travel_dist=100.0,
        away_rest_days=2.0,
        away_travel_dist=200.0
    )
    print(f"\nSingle-Game Prediction: Probability home wins = {prob:.4f}")

    # 4) Evaluate entire test CSV
    test_csv_path = "games_data_testing.csv"
    if not os.path.exists(test_csv_path):
        print(f"No {test_csv_path} found, skipping batch evaluation.")
        return

    test_df = pd.read_csv(test_csv_path)
    print(f"\nLoaded {len(test_df)} rows from {test_csv_path}...")

    y_probs = []
    y_true = []

    has_scores = ("team_score_Home" in test_df.columns
                  and "team_score_Away" in test_df.columns)

    for _, row in test_df.iterrows():
        team_home = row.get("team_home", "")
        team_away = row.get("team_away", "")

        # We look for "home_rest_days"/"rest_days_Home"
        home_rest_days = row.get("home_rest_days", row.get("rest_days_Home", 0.0))
        away_rest_days = row.get("away_rest_days", row.get("rest_days_Away", 0.0))
        home_travel_dist = row.get("home_travel_dist", row.get("travel_dist_Home", 0.0))
        away_travel_dist = row.get("away_travel_dist", row.get("travel_dist_Away", 0.0))

        # Predict
        prob_home_win = my_model.predict_single_game(
            team_home=team_home,
            team_away=team_away,
            home_rest_days=home_rest_days,
            home_travel_dist=home_travel_dist,
            away_rest_days=away_rest_days,
            away_travel_dist=away_travel_dist
        )
        y_probs.append(prob_home_win)

        if has_scores:
            home_score = row["team_score_Home"]
            away_score = row["team_score_Away"]
            y_true.append(1 if home_score > away_score else 0)

    y_probs = np.array(y_probs)

    # If we have actual scores, evaluate
    if has_scores and len(y_true) == len(y_probs):
        y_true = np.array(y_true)
        # Convert probabilities to 0/1
        y_pred = (y_probs >= 0.5).astype(int)

        accuracy = accuracy_score(y_true, y_pred)
        brier = brier_score_loss(y_true, y_probs)
        ll = log_loss(y_true, np.column_stack([1 - y_probs, y_probs]))
        auc = roc_auc_score(y_true, y_probs)
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, digits=4)

        print("\n=== MODEL PERFORMANCE (TEST) ===")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Brier:     {brier:.4f}")
        print(f"Log Loss:  {ll:.4f}")
        print(f"ROC AUC:   {auc:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(report)

        # ----------------------
        # Plot Additional Graphs
        # ----------------------
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        axs = axs.ravel()

        # 1) Probability vs. Actual Point Difference
        if "team_score_Home" in test_df.columns and "team_score_Away" in test_df.columns:
            actual_diff = test_df["team_score_Home"] - test_df["team_score_Away"]
            axs[0].scatter(y_probs, actual_diff, alpha=0.5)
            axs[0].set_xlabel("Predicted Probability (Home Win)")
            axs[0].set_ylabel("Actual Home Team Point Diff")
            axs[0].set_title("Prediction vs. Actual Point Diff")
            axs[0].grid(True)

        # 2) Calibration Plot
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

        axs[1].plot(mean_predicted, actual_win_rates, 'o-', label="Actual vs. Predicted")
        axs[1].plot([0, 1], [0, 1], 'k--', label="Perfect Calibration")
        axs[1].set_xlabel("Mean Predicted Probability")
        axs[1].set_ylabel("Actual Home Win Rate")
        axs[1].set_title("Calibration Plot")
        axs[1].legend(loc="best")
        axs[1].grid(True)

        # 3) ROC Curve
        fpr, tpr, thresholds_roc = roc_curve(y_true, y_probs)
        axs[2].plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        axs[2].plot([0, 1], [0, 1], 'k--')
        axs[2].set_xlabel("False Positive Rate")
        axs[2].set_ylabel("True Positive Rate")
        axs[2].set_title("ROC Curve")
        axs[2].legend(loc="lower right")
        axs[2].grid(True)

        # 4) Precision-Recall Curve
        precision, recall, thresholds_pr = precision_recall_curve(y_true, y_probs)
        axs[3].plot(recall, precision, label="Precision-Recall")
        axs[3].set_xlabel("Recall")
        axs[3].set_ylabel("Precision")
        axs[3].set_title("Precision-Recall Curve")
        axs[3].legend(loc="best")
        axs[3].grid(True)

        # 5) Histogram of predicted probabilities
        axs[4].hist(y_probs, bins=20, range=(0,1), alpha=0.7, color='g', edgecolor='k')
        axs[4].set_xlabel("Predicted Probability (Home Win)")
        axs[4].set_ylabel("Count")
        axs[4].set_title("Distribution of Predictions")
        axs[4].grid(True)

        # 6) Confusion Matrix (visual)
        axs[5].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axs[5].set_title("Confusion Matrix (Visual)")
        axs[5].set_xticks([0, 1])
        axs[5].set_yticks([0, 1])
        axs[5].set_xticklabels(["Pred 0", "Pred 1"])
        axs[5].set_yticklabels(["True 0", "True 1"])

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axs[5].text(j, i, str(cm[i, j]),
                            ha="center", va="center",
                            color="black", fontsize=12)

        axs[5].set_ylabel("True label")
        axs[5].set_xlabel("Predicted label")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
