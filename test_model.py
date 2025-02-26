import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.model_selection import train_test_split
import os

from model import Model
from games_data import GamesData

def main():
    # 1) Create the GamesData (for consistent feature engineering)
    training_csv_path = "games_data.csv"
    games_data = GamesData(training_csv_path)

    # 2) Load the pre-trained Stacking Model (or train a new one if desired)
    model_file_path = "model_stuff.json"
    my_model = Model(model_file_path=model_file_path, games_data=games_data)
    if my_model.model is None:
        raise ValueError(f"No model loaded from {model_file_path}.")

    # 3) Single-game prediction example
    prob = my_model.predict_single_game(
        team_home="lsu_tigers",
        team_away="georgia_lady_bulldogs",
        home_rest_days=3.0,
        home_travel_dist=100.0,
        away_rest_days=2.0,
        away_travel_dist=200.0
    )
    print(f"\nSingle-Game Prediction: Probability home wins = {prob:.4f}")

    # 4) Evaluate on a held-out test split
    merged_df = games_data.prepare_training_data()

    # -- Compute the actual point difference for each row --
    merged_df["point_diff"] = merged_df["home_team_score"] - merged_df["away_team_score"]

    # -- Build X (features) and y (labels) --
    X, y = games_data.get_feature_and_label_arrays(merged_df)

    # We'll store the point_diff in a separate array so we can split consistently
    diff_array = merged_df["point_diff"].values

    X_train, X_test, y_train, y_test, diff_train, diff_test = train_test_split(
        X, y, diff_array,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # Predict probabilities for the test set
    y_probs = my_model.predict_batch(X_test)
    y_pred = (y_probs >= 0.5).astype(int)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_probs)
    ll = log_loss(y_test, np.column_stack([1 - y_probs, y_probs]))
    auc = roc_auc_score(y_test, y_probs)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    print("\n=== MODEL PERFORMANCE (TEST SPLIT) ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Brier:     {brier:.4f}")
    print(f"Log Loss:  {ll:.4f}")
    print(f"ROC AUC:   {auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    # Optional plots for deeper analysis
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    axs = axs.ravel()

    # 1) Probability vs. Actual Point Difference
    axs[0].scatter(y_probs, diff_test, alpha=0.5)
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
            bin_actual_win = np.mean(y_test[idxs])
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
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_probs)
    axs[2].plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    axs[2].plot([0, 1], [0, 1], 'k--')
    axs[2].set_xlabel("False Positive Rate")
    axs[2].set_ylabel("True Positive Rate")
    axs[2].set_title("ROC Curve")
    axs[2].legend(loc="lower right")
    axs[2].grid(True)

    # 4) Precision-Recall Curve
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_probs)
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

    # plt.tight_layout()
    # plt.show()
    
    # # 1) Prediction vs. Actual Point Difference
    # plt.figure(figsize=(6,4))
    # plt.scatter(y_probs, diff_test, alpha=0.5)
    # plt.xlabel("Predicted Probability (Home Win)")
    # plt.ylabel("Actual Home Team Point Diff")
    # plt.title("Prediction vs. Actual Point Diff")
    # plt.grid(True)
    # plt.savefig("graph_prediction_vs_actual.png", dpi=300, bbox_inches='tight')
    # plt.close()

    # # 2) Calibration Plot
    # plt.figure(figsize=(6,4))
    # plt.plot(mean_predicted, actual_win_rates, 'o-', label="Actual vs. Predicted")
    # plt.plot([0, 1], [0, 1], 'k--', label="Perfect Calibration")
    # plt.xlabel("Mean Predicted Probability")
    # plt.ylabel("Actual Home Win Rate")
    # plt.title("Calibration Plot")
    # plt.legend(loc="best")
    # plt.grid(True)
    # plt.savefig("graph_calibration.png", dpi=300, bbox_inches='tight')
    # plt.close()

    # # 3) ROC Curve
    # plt.figure(figsize=(6,4))
    # fpr, tpr, thresholds_roc = roc_curve(y_test, y_probs)
    # plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("ROC Curve")
    # plt.legend(loc="lower right")
    # plt.grid(True)
    # plt.savefig("graph_roc_curve.png", dpi=300, bbox_inches='tight')
    # plt.close()

    # # 4) Precision-Recall Curve
    # plt.figure(figsize=(6,4))
    # precision, recall, thresholds_pr = precision_recall_curve(y_test, y_probs)
    # plt.plot(recall, precision, label="Precision-Recall")
    # plt.xlabel("Recall")
    # plt.ylabel("Precision")
    # plt.title("Precision-Recall Curve")
    # plt.legend(loc="best")
    # plt.grid(True)
    # plt.savefig("graph_precision_recall.png", dpi=300, bbox_inches='tight')
    # plt.close()

    # # 5) Histogram of Predicted Probabilities
    # plt.figure(figsize=(6,4))
    # plt.hist(y_probs, bins=20, range=(0,1), alpha=0.7, color='g', edgecolor='k')
    # plt.xlabel("Predicted Probability (Home Win)")
    # plt.ylabel("Count")
    # plt.title("Distribution of Predictions")
    # plt.grid(True)
    # plt.savefig("graph_histogram.png", dpi=300, bbox_inches='tight')
    # plt.close()

    # # 6) Confusion Matrix (Visual)
    # plt.figure(figsize=(6,4))
    # plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.title("Confusion Matrix (Visual)")
    # plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    # plt.yticks([0, 1], ["True 0", "True 1"])
    # for i in range(cm.shape[0]):
    #     for j in range(cm.shape[1]):
    #         plt.text(j, i, str(cm[i, j]),
    #                 ha="center", va="center",
    #                 color="black", fontsize=12)
    # plt.xlabel("Predicted label")
    # plt.ylabel("True label")
    # plt.savefig("graph_confusion_matrix.png", dpi=300, bbox_inches='tight')
    # plt.close()


if __name__ == "__main__":
    main()
