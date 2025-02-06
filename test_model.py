# test_model.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_utility import load_games_data, prepare_matchup_data, get_feature_and_label_arrays
from model import Model, FEATURE_COLS

def main():
    # Paths
    csv_path = "games_data_half1.csv"
    model_file_path = "model_stuff.json"

    print("Loading test data...")
    df = load_games_data(csv_path)
    merged_df = prepare_matchup_data(df)

    # Example feature columns
  
    label_col = "home_team_won"

    X, y = get_feature_and_label_arrays(merged_df, FEATURE_COLS, label_col)

    # Load the existing model
    my_model = Model(model_file_path=model_file_path)
    if my_model.model is None:
        raise ValueError("No valid model loaded. Please generate_model first.")

    print("Predicting on test data...")
    probs = my_model.predict(X)  # Probability that the home team wins

    # === Plot 1: Predicted probability vs. actual point difference ===
    actual_diff = merged_df["home_team_score"] - merged_df["away_team_score"]
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)  # Put the first plot on the left
    plt.scatter(probs, actual_diff, alpha=0.5)
    plt.xlabel("Predicted Probability (Home Team Win)")
    plt.ylabel("Actual Home Team Point Difference")
    plt.title("Model Prediction vs. Actual Point Difference")
    plt.grid(True)

    # === Plot 2: Calibration-like plot (Actual win rate vs. predicted probability) ===
    # We group predictions into bins and compute actual win percentage in each bin
    n_bins = 10
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(probs, bins) - 1  # which bin each prob goes into

    actual_win_rates = []
    mean_predicted = []
    for i in range(n_bins):
        idxs = np.where(bin_indices == i)[0]
        if len(idxs) > 0:
            # Actual fraction of home-team wins in this bin
            bin_actual_win = np.mean(y[idxs])
            # Mean predicted probability in this bin
            bin_mean_prob = np.mean(probs[idxs])
        else:
            bin_actual_win = np.nan
            bin_mean_prob = np.nan

        actual_win_rates.append(bin_actual_win)
        mean_predicted.append(bin_mean_prob)

    plt.subplot(1, 2, 2)  # Put the second plot on the right
    plt.plot(mean_predicted, actual_win_rates, "o-", label="Actual vs. Predicted")
    # Add a reference line for perfectly calibrated predictions
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Actual Win Rate (Home Team)")
    plt.title("Calibration Plot")
    plt.legend(loc="best")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # === Calculate performance metrics ===
    from sklearn.metrics import (
        accuracy_score, brier_score_loss, log_loss, roc_auc_score,
        confusion_matrix, classification_report
    )

    # Convert probabilistic predictions to binary predictions using a 0.5 threshold
    predictions_binary = (probs >= 0.5).astype(int)

    accuracy = accuracy_score(y, predictions_binary)
    brier = brier_score_loss(y, probs)
    # For log_loss, we need probabilities of both classes
    logloss = log_loss(y, np.column_stack([1 - probs, probs]))
    roc_auc = roc_auc_score(y, probs)
    cm = confusion_matrix(y, predictions_binary)
    report = classification_report(y, predictions_binary, digits=4)

    print("\n=== MODEL PERFORMANCE METRICS ===")
    print(f"Accuracy:         {accuracy:.4f}")
    print(f"Brier Score:      {brier:.4f} (lower is better)")
    print(f"Log Loss:         {logloss:.4f} (lower is better)")
    print(f"ROC AUC:          {roc_auc:.4f} (higher is better)")
    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    print(cm)
    print("\nClassification Report:")
    print(report)

if __name__ == "__main__":
    main()
