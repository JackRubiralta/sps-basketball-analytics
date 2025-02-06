# generate_model.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split

from data_utility import load_games_data, prepare_matchup_data, get_feature_and_label_arrays
from model import Model

def main():
    # Path to CSV data
    csv_path = "games_data.csv"
    # Path to save the trained model
    model_file_path = "model_stuff.json"

    print("Loading data...")
    df = load_games_data(csv_path)

    print("Preparing matchup data...")
    merged_df = prepare_matchup_data(df)

    # Define your feature columns. 
    # Here we are taking some 'diff_' columns and maybe a few direct columns for demonstration.
    feature_cols = [
        "diff_FGA_2", "diff_FGM_2",
        "diff_FGA_3", "diff_FGM_3",
        "diff_FTA", "diff_FTM",
        "diff_AST", "diff_BLK", "diff_STL", "diff_TOV",
        "diff_DREB", "diff_OREB", "diff_F_personal",
        "diff_rest_days", "diff_travel_dist"
    ]

    label_col = "home_team_won"  # or another label of your choice
    X, y = get_feature_and_label_arrays(merged_df, feature_cols, label_col)

    # Split data into train/validation (and optionally test)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize Model object
    my_model = Model(model_file_path=model_file_path)

    # Generate (train) the model
    print("Training model...")
    my_model.generate_model(X_train, y_train, X_val, y_val, 
                            learning_rate=0.05,
                            max_depth=8,
                            n_estimators=500,
                            subsample=0.8,
                            colsample_bytree=0.8)

    # Save the trained model
    my_model.save_model()
    print("Done generating and saving the model.")


if __name__ == "__main__":
    main()
