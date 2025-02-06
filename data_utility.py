# data_utility.py

import pandas as pd
import numpy as np
import os

def load_games_data(csv_path: str) -> pd.DataFrame:
    """
    Loads the games_data CSV into a pandas DataFrame.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} does not exist.")
    df = pd.read_csv(csv_path)
    return df


def prepare_matchup_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame (with each row representing a single team in a single game),
    pivot/merge so that each row represents one game/matchup. The typical assumption:
      - For each game_id, there are exactly two rows: one for 'home' and one for 'away'.
      - We'll designate the 'home' team as the 'Team' and the 'away' team as the 'Opponent'
        or you could choose any consistent approach.

    Returns a DataFrame with combined features plus a binary 'home_team_won' outcome
    (or any other outcome variable you prefer).
    """
    # Separate home and away
    df_home = df[df["home_away"] == "home"].copy()
    df_away = df[df["home_away"] == "away"].copy()

    # Rename columns for home vs. away clarity
    home_cols = {col: f"home_{col}" for col in df.columns if col not in ["game_id", "home_away"]}
    away_cols = {col: f"away_{col}" for col in df.columns if col not in ["game_id", "home_away"]}

    df_home.rename(columns=home_cols, inplace=True)
    df_away.rename(columns=away_cols, inplace=True)

    # Merge home and away rows on game_id
    merged = pd.merge(df_home, df_away, left_on="game_id", right_on="game_id", how="inner")

    # Create a label: did the home team win?
    merged["home_team_won"] = (merged["home_team_score"] > merged["away_team_score"]).astype(int)

    # Example feature engineering:
    # Let's create some matchup-level features: differences in stats (home minus away)
    merged["diff_FGA_2"] = merged["home_FGA_2"] - merged["away_FGA_2"]
    merged["diff_FGM_2"] = merged["home_FGM_2"] - merged["away_FGM_2"]
    merged["diff_FGA_3"] = merged["home_FGA_3"] - merged["away_FGA_3"]
    merged["diff_FGM_3"] = merged["home_FGM_3"] - merged["away_FGM_3"]
    merged["diff_FTA"] = merged["home_FTA"] - merged["away_FTA"]
    merged["diff_FTM"] = merged["home_FTM"] - merged["away_FTM"]
    merged["diff_AST"] = merged["home_AST"] - merged["away_AST"]
    merged["diff_BLK"] = merged["home_BLK"] - merged["away_BLK"]
    merged["diff_STL"] = merged["home_STL"] - merged["away_STL"]
    merged["diff_TOV"] = merged["home_TOV"] - merged["away_TOV"]
    merged["diff_DREB"] = merged["home_DREB"] - merged["away_DREB"]
    merged["diff_OREB"] = merged["home_OREB"] - merged["away_OREB"]
    merged["diff_F_personal"] = merged["home_F_personal"] - merged["away_F_personal"]
    merged["diff_rest_days"] = merged["home_rest_days"] - merged["away_rest_days"]
    merged["diff_travel_dist"] = merged["home_travel_dist"] - merged["away_travel_dist"]
    # etc. You can create as many specialized features as you'd like.

    return merged


def get_feature_and_label_arrays(merged_df: pd.DataFrame,
                                 feature_cols: list,
                                 label_col: str = "home_team_won"):
    """
    From the merged matchup DataFrame, extract features X and label y.
    """
    X = merged_df[feature_cols].values
    y = merged_df[label_col].values
    return X, y
