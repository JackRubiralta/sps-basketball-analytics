# test_model.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_utility import load_games_data, prepare_matchup_data, get_feature_and_label_arrays
from model import Model

def main():
    # Paths
    csv_path = "games_data.csv"
    model_file_path = "model_stuff.json"

    print("Loading test data...")
    df = load_games_data(csv_path)
    merged_df = prepare_matchup_data(df)

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

    # For a "true" points difference, we might do:
    # home_points_diff = merged_df["home_team_score"] - merged_df["away_team_score"]

    # Load the existing model
    my_model = Model(model_file_path=model_file_path)
    if my_model.model is None:
        raise ValueError("No valid model loaded. Please generate_model first.")

    print("Predicting on test data...")
    probs = my_model.predict(X)  # Probability that the home team wins

    # Example: We'll compare predicted probability vs. actual outcome
    # Convert y (0/1) into "actual home team won or not"
    # Also show how we might plot vs. the actual point difference
    actual_diff = merged_df["home_team_score"] - merged_df["away_team_score"]

    # Plot: X-axis is actual point difference, Y-axis is predicted probability
    plt.figure(figsize=(8,6))
    plt.scatter(probs, actual_diff, alpha=0.5)
    plt.xlabel("Actual Home Team Point Difference")
    plt.ylabel("Predicted Probability (Home Team Win)")
    plt.title("Model Prediction vs. Actual Point Difference")
    plt.grid(True)
    plt.show()

    # Optionally calculate some performance metrics
    from sklearn.metrics import accuracy_score, brier_score_loss
    predictions_binary = (probs >= 0.5).astype(int)
    accuracy = accuracy_score(y, predictions_binary)
    brier = brier_score_loss(y, probs)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Brier Score: {brier:.4f}")

if __name__ == "__main__":
    main()
