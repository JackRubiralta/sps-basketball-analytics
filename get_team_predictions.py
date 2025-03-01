import pandas as pd
import os

from model import Model
from games_data import GamesData

def main():
    # 1) Load our trained model (and its supporting GamesData).
    #    This will automatically load an existing model from "model_stuff.json" if present.
    games_data = GamesData("games_data.csv")
    model_file_path = "model_stuff.json"
    my_model = Model(model_file_path=model_file_path, games_data=games_data)

    # 2) Read the CSV with games to predict
    predict_csv_path = "games_to_predict.csv"
    if not os.path.exists(predict_csv_path):
        raise FileNotFoundError(f"Could not find '{predict_csv_path}' for predictions.")

    df_to_predict = pd.read_csv(predict_csv_path)

    # Prepare columns in case they differ slightly
    required_cols = [
        "game_id", "description",
        "team_home", "team_away",
        "rest_days_Home", "rest_days_Away",
        "travel_dist_Home", "travel_dist_Away"
    ]
    for col in required_cols:
        if col not in df_to_predict.columns:
            raise ValueError(f"Column '{col}' missing from '{predict_csv_path}'.")

    # 3) Iterate through each game, build a prediction
    home_probs = []
    predicted_winners = []

    for idx, row in df_to_predict.iterrows():
        # Extract fields
        team_home = row["team_home"]
        team_away = row["team_away"]
        home_rest_days = float(row["rest_days_Home"])
        away_rest_days = float(row["rest_days_Away"])
        home_travel_dist = float(row["travel_dist_Home"])
        away_travel_dist = float(row["travel_dist_Away"])

        # Predict probability of home team winning
        prob_home_win = my_model.predict_single_game(
            team_home=team_home,
            team_away=team_away,
            home_rest_days=home_rest_days,
            home_travel_dist=home_travel_dist,
            away_rest_days=away_rest_days,
            away_travel_dist=away_travel_dist
        )
        home_probs.append(prob_home_win)

        # Determine predicted winner
        predicted_winner = team_home if prob_home_win >= 0.5 else team_away
        predicted_winners.append(predicted_winner)

    # 4) Add predictions to the DataFrame
    df_to_predict["predicted_home_win_prob"] = home_probs
    df_to_predict["predicted_winner"] = predicted_winners

    # 5) Save predictions to a new CSV
    output_csv_path = "produced_game_predicitons.csv"
    df_to_predict.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to '{output_csv_path}'.")

if __name__ == "__main__":
    main()
