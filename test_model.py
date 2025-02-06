# test_model.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_utility import load_games_data, prepare_matchup_data, get_feature_and_label_arrays
from model import Model

def main():
    # Paths
    csv_path = "games_data_half_test.csv"
    model_file_path = "model_stuff.json"

    print("Loading test data...")
    df = load_games_data(csv_path)
    merged_df = prepare_matchup_data(df)

    # Example feature columns
    feature_cols = [
        "diff_FGA_2", "diff_FGM_2",
        "diff_FGA_3", "diff_FGM_3",
        "diff_FTA", "diff_FTM",
        "diff_AST", "diff_BLK", "diff_STL", "diff_TOV",
        "diff_DREB", "diff_OREB", "diff_F_personal",
        "diff_rest_days", "diff_travel_dist"
    ]
    label_col = "home_team_won"

    X, y = get_feature_and_label_arrays(merged_df, feature_cols, label_col)

    # Load the existing model
    my_model = Model(model_file_path=model_file_path)
    if my_model.model is None:
        raise ValueError("No valid model loaded. Please generate_model first.")

    print("Predicting on test data...")
    probs = my_model.predict(X)  # Probability that the home team wins

    # Actual point difference (home - away)
    actual_diff = merged_df["home_team_score"] - merged_df["away_team_score"]

    # Plot: X-axis is predicted probability, Y-axis is actual point difference
    plt.figure(figsize=(8,6))
    plt.scatter(probs, actual_diff, alpha=0.5)
    plt.xlabel("Predicted Probability (Home Team Win)")
    plt.ylabel("Actual Home Team Point Difference")
    plt.title("Model Prediction vs. Actual Point Difference")
    plt.grid(True)
    plt.show()

    # Calculate performance metrics
    from sklearn.metrics import (
        accuracy_score, brier_score_loss, log_loss, roc_auc_score, 
        confusion_matrix, classification_report
    )

    # Convert probabilistic predictions to binary predictions using 0.5 threshold
    predictions_binary = (probs >= 0.5).astype(int)

    accuracy = accuracy_score(y, predictions_binary)
    brier = brier_score_loss(y, probs)
    # For log_loss, we need probabilities of both classes:
    # For a binary classifier, we can do: np.column_stack([1 - probs, probs])
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
