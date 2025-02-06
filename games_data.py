# games_data.py

import pandas as pd
import numpy as np
import os

class GamesData:
    """
    Manages the data pipeline for training and prediction:
    - Loads training_data.csv
    - Computes team averages
    - Merges home/away data into one row per game for training
    - Builds feature vectors for new matchups
    """

    # The columns we want to average for each team
    _cols_to_avg = [
        "FGA_2", "FGM_2", "FGA_3", "FGM_3",
        "FTA",   "FTM",   "AST",   "BLK",
        "STL",   "TOV",   "DREB",  "OREB",
        "F_personal"
    ]

    # The final feature columns we want for training/prediction
    FEATURE_COLS = [
        # 13 home_avg_ stats
        "home_avg_FGA_2", "home_avg_FGM_2",
        "home_avg_FGA_3", "home_avg_FGM_3",
        "home_avg_FTA",   "home_avg_FTM",
        "home_avg_AST",   "home_avg_BLK",
        "home_avg_STL",   "home_avg_TOV",
        "home_avg_DREB",  "home_avg_OREB",
        "home_avg_F_personal",

        # 13 away_avg_ stats
        "away_avg_FGA_2", "away_avg_FGM_2",
        "away_avg_FGA_3", "away_avg_FGM_3",
        "away_avg_FTA",   "away_avg_FTM",
        "away_avg_AST",   "away_avg_BLK",
        "away_avg_STL",   "away_avg_TOV",
        "away_avg_DREB",  "away_avg_OREB",
        "away_avg_F_personal",

        # Some game-level columns that we know exist in both train & test
        "home_rest_days",  "home_travel_dist",
        "away_rest_days",  "away_travel_dist",
    ]

    def __init__(self, csv_path: str):
        """
        Loads the training CSV (one row per team/game).
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{csv_path} does not exist.")

        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)

        # Precompute the team averages from training data
        self.team_avgs = self._compute_team_averages(self.df)

    def _compute_team_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Group by 'team' and compute averages for columns in _cols_to_avg.
        """
        df_avg = df.groupby("team")[self._cols_to_avg].mean().reset_index()
        rename_dict = {col: f"avg_{col}" for col in self._cols_to_avg}
        df_avg.rename(columns=rename_dict, inplace=True)
        return df_avg

    def prepare_training_data(self) -> pd.DataFrame:
        """
        Merges home & away rows into a single row per game, storing
        home/away stats + 'home_team_won'.
        """
        df_team_avgs = self.team_avgs

        # Separate out 'home' rows and 'away' rows
        df_home = self.df[self.df["home_away"] == "home"].copy()
        df_away = self.df[self.df["home_away"] == "away"].copy()

        # Merge each side with the team averages
        df_home = df_home.merge(df_team_avgs, on="team", how="left")
        df_away = df_away.merge(df_team_avgs, on="team", how="left")

        # Rename columns to say "home_" or "away_"
        home_cols = {}
        away_cols = {}
        for col in df_home.columns:
            if col not in ["game_id", "team", "game_date", "home_away"]:
                home_cols[col] = f"home_{col}"

        for col in df_away.columns:
            if col not in ["game_id", "team", "game_date", "home_away"]:
                away_cols[col] = f"away_{col}"

        df_home.rename(columns=home_cols, inplace=True)
        df_away.rename(columns=away_cols, inplace=True)

        # Merge home & away frames on game_id
        merged = pd.merge(df_home, df_away, on="game_id", how="inner")

        # Create the label: did the home team win?
        merged["home_team_won"] = (
            merged["home_team_score"] > merged["away_team_score"]
        ).astype(int)

        return merged

    def get_feature_and_label_arrays(self, merged_df: pd.DataFrame,
                                     label_col: str = "home_team_won"):
        """
        Extracts (X, y) from the merged_df.
        X has columns in FEATURE_COLS, y is label_col.
        """
        X = merged_df[self.FEATURE_COLS].values
        y = merged_df[label_col].values
        return X, y

    def build_prediction_features(self,
                                  team_home: str,
                                  team_away: str,
                                  home_rest_days: float,
                                  home_travel_dist: float,
                                  away_rest_days: float,
                                  away_travel_dist: float) -> np.ndarray:
        """
        Creates a single 1-row feature vector for a new matchup (not from CSV).
        We:
          1) Look up the average stats for team_home and team_away from self.team_avgs
          2) Construct the feature array in the same order as FEATURE_COLS
        Returns a numpy array shape (1, num_features).
        """

        # 1) Retrieve home/away average stats
        home_avgs = self.team_avgs[self.team_avgs["team"] == team_home].copy()
        away_avgs = self.team_avgs[self.team_avgs["team"] == team_away].copy()

        if home_avgs.empty:
            raise ValueError(f"No training average found for home team '{team_home}'")
        if away_avgs.empty:
            raise ValueError(f"No training average found for away team '{team_away}'")

        # 2) Rename columns, e.g. 'avg_FGA_2' -> 'home_avg_FGA_2'
        home_avgs = home_avgs.rename(
            columns={col: f"home_{col}" for col in home_avgs.columns if col != "team"}
        )
        away_avgs = away_avgs.rename(
            columns={col: f"away_{col}" for col in away_avgs.columns if col != "team"}
        )

        # 3) Build a dict with the needed columns
        features_dict = {}

        # Insert home average stats (like home_avg_FGA_2, etc.)
        for col in home_avgs.columns:
            if col == "team":
                continue
            features_dict[col] = home_avgs.iloc[0][col]

        # Insert away average stats
        for col in away_avgs.columns:
            if col == "team":
                continue
            features_dict[col] = away_avgs.iloc[0][col]

        # Insert the numeric fields for rest/travel
        features_dict["home_rest_days"]   = home_rest_days
        features_dict["home_travel_dist"] = home_travel_dist
        features_dict["away_rest_days"]   = away_rest_days
        features_dict["away_travel_dist"] = away_travel_dist

        # 4) Ensure we fill in all FEATURE_COLS in the correct order
        row = []
        for col in self.FEATURE_COLS:
            if col not in features_dict:
                # If we are missing something, fill with 0 or raise an error
                row.append(0.0)
            else:
                row.append(features_dict[col])

        return np.array([row])  # shape (1, n_features)
