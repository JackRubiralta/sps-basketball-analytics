import pandas as pd
import numpy as np
import os

class GamesData:
    """
    Manages the data pipeline for training and prediction:
    - Loads training_data.csv
    - Computes advanced stats for each game (e.g., shooting percentages, TS%, ORtg, etc.)
    - Computes team averages (including advanced stats)
    - Merges home/away data into one row per game for training
    - Builds feature vectors for new matchups
    """

    # Updated list of columns to average for each team.
    # Includes both basic box score stats and the new advanced stats.
    _cols_to_avg = [
        # Basic stats (as provided)
        "FGA_2", "FGM_2", "FGA_3", "FGM_3",
        "FTA", "FTM", "AST", "BLK",
        "STL", "TOV", "DREB", "OREB",
        "F_personal",
        # Advanced stats to be computed:
        "FG_pct", "FG2_pct", "FG3_pct", "FT_pct",
        "PTS", "TS_pct", "ORtg", "AST_TOV", "eFG_pct"
    ]

    # Updated feature columns for training/prediction.
    # For each team (home and away) we will use the averages of basic and advanced stats.
    FEATURE_COLS = [
        # Home team advanced averages:
        "home_avg_FGA_2", "home_avg_FGM_2",
        "home_avg_FGA_3", "home_avg_FGM_3",
        "home_avg_FTA",   "home_avg_FTM",
        "home_avg_AST",   "home_avg_BLK",
        "home_avg_STL",   "home_avg_TOV",
        "home_avg_DREB",  "home_avg_OREB",
        "home_avg_F_personal",
        "home_avg_FG_pct", "home_avg_FG2_pct",
        "home_avg_FG3_pct", "home_avg_FT_pct",
        "home_avg_PTS", "home_avg_TS_pct",
        "home_avg_ORtg", "home_avg_AST_TOV",
        "home_avg_eFG_pct",

        # Away team advanced averages:
        "away_avg_FGA_2", "away_avg_FGM_2",
        "away_avg_FGA_3", "away_avg_FGM_3",
        "away_avg_FTA",   "away_avg_FTM",
        "away_avg_AST",   "away_avg_BLK",
        "away_avg_STL",   "away_avg_TOV",
        "away_avg_DREB",  "away_avg_OREB",
        "away_avg_F_personal",
        "away_avg_FG_pct", "away_avg_FG2_pct",
        "away_avg_FG3_pct", "away_avg_FT_pct",
        "away_avg_PTS", "away_avg_TS_pct",
        "away_avg_ORtg", "away_avg_AST_TOV",
        "away_avg_eFG_pct",

        # Game-level info:
        "home_rest_days",  "home_travel_dist",
        "away_rest_days",  "away_travel_dist",
    ]

    def __init__(self, csv_path: str):
        """
        Loads the training CSV (one row per team/game) and computes advanced stats.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{csv_path} does not exist.")

        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)

        # Compute advanced statistics for each game
        self.df = self._compute_advanced_stats(self.df)

        # Precompute the team averages (including advanced stats)
        self.team_avgs = self._compute_team_averages(self.df)

    def _compute_advanced_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes advanced statistics for each game row.
        New stats:
          - Overall Field Goal Percentage (FG_pct): (FGM_2 + FGM_3) / (FGA_2 + FGA_3)
          - Two-Point Percentage (FG2_pct): FGM_2 / FGA_2
          - Three-Point Percentage (FG3_pct): FGM_3 / FGA_3
          - Free Throw Percentage (FT_pct): FTM / FTA
          - Points (PTS): 2*FGM_2 + 3*FGM_3 + FTM
          - True Shooting Percentage (TS_pct): PTS / (2 * ((FGA_2+FGA_3) + 0.44*FTA))
          - Offensive Rating (ORtg): (PTS / Possessions) * 100,
              where Possessions is approximated as: (FGA_2+FGA_3) + 0.44*FTA + TOV - OREB
          - Assist-to-Turnover Ratio (AST_TOV): AST / TOV
          - Effective Field Goal Percentage (eFG_pct): (FGM_2 + 1.5*FGM_3) / (FGA_2+FGA_3)
        """
        df = df.copy()

        # Compute overall field goal attempts and makes
        df['FGA_total'] = df['FGA_2'] + df['FGA_3']
        df['FGM_total'] = df['FGM_2'] + df['FGM_3']

        # 1. Field Goal Percentage
        df['FG_pct'] = np.where(df['FGA_total'] > 0, df['FGM_total'] / df['FGA_total'], 0.0)

        # 2. Two-Point Percentage
        df['FG2_pct'] = np.where(df['FGA_2'] > 0, df['FGM_2'] / df['FGA_2'], 0.0)

        # 3. Three-Point Percentage
        df['FG3_pct'] = np.where(df['FGA_3'] > 0, df['FGM_3'] / df['FGA_3'], 0.0)

        # 4. Free Throw Percentage
        df['FT_pct'] = np.where(df['FTA'] > 0, df['FTM'] / df['FTA'], 0.0)

        # 5. Points: 2 points per FGM_2, 3 points per FGM_3, 1 point per FTM
        df['PTS'] = (2 * df['FGM_2']) + (3 * df['FGM_3']) + df['FTM']

        # 6. True Shooting Percentage (TS_pct)
        denom = 2 * (df['FGA_total'] + 0.44 * df['FTA'])
        df['TS_pct'] = np.where(denom > 0, df['PTS'] / denom, 0.0)

        # 7. Possessions (approximation)
        possessions = df['FGA_total'] + 0.44 * df['FTA'] + df['TOV'] - df['OREB']
        # 8. Offensive Rating (ORtg): points per 100 possessions
        df['ORtg'] = np.where(possessions > 0, (df['PTS'] / possessions) * 100, 0.0)

        # 9. Assist-to-Turnover Ratio (AST_TOV)
        df['AST_TOV'] = np.where(df['TOV'] > 0, df['AST'] / df['TOV'], 0.0)

        # 10. Effective Field Goal Percentage (eFG_pct)
        df['eFG_pct'] = np.where(df['FGA_total'] > 0,
                                 (df['FGM_2'] + 1.5 * df['FGM_3']) / df['FGA_total'],
                                 0.0)

        return df

    def _compute_team_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Group by 'team' and compute averages for all columns in _cols_to_avg.
        The advanced stats computed in _compute_advanced_stats are now averaged per team.
        """
        df_avg = df.groupby("team")[self._cols_to_avg].mean().reset_index()
        rename_dict = {col: f"avg_{col}" for col in self._cols_to_avg}
        df_avg.rename(columns=rename_dict, inplace=True)
        return df_avg

    def prepare_training_data(self) -> pd.DataFrame:
        """
        Merges home & away rows into a single row per game, storing
        home/away stats (basic and advanced averages) + 'home_team_won'.
        Assumes the input CSV has a column 'home_away' to distinguish home vs. away.
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

        # Create the label: did the home team win? (Assumes columns home_team_score and away_team_score exist)
        merged["home_team_won"] = (merged["home_team_score"] > merged["away_team_score"]).astype(int)

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
        Steps:
          1) Look up the average stats for team_home and team_away from self.team_avgs.
          2) Construct the feature array in the same order as FEATURE_COLS.
        Returns a numpy array of shape (1, num_features).
        """
        # Retrieve home/away average stats
        home_avgs = self.team_avgs[self.team_avgs["team"] == team_home].copy()
        away_avgs = self.team_avgs[self.team_avgs["team"] == team_away].copy()

        if home_avgs.empty:
            raise ValueError(f"No training average found for home team '{team_home}'")
        if away_avgs.empty:
            raise ValueError(f"No training average found for away team '{team_away}'")

        # Rename columns to match the prediction feature names (e.g., 'avg_FGA_2' -> 'home_avg_FGA_2')
        home_avgs = home_avgs.rename(
            columns={col: f"home_{col}" for col in home_avgs.columns if col != "team"}
        )
        away_avgs = away_avgs.rename(
            columns={col: f"away_{col}" for col in away_avgs.columns if col != "team"}
        )

        # Build a dictionary with all needed features
        features_dict = {}

        # Insert home average stats
        for col in home_avgs.columns:
            if col == "team":
                continue
            features_dict[col] = home_avgs.iloc[0][col]

        # Insert away average stats
        for col in away_avgs.columns:
            if col == "team":
                continue
            features_dict[col] = away_avgs.iloc[0][col]

        # Insert game-level numeric fields for rest/travel
        features_dict["home_rest_days"]   = home_rest_days
        features_dict["home_travel_dist"] = home_travel_dist
        features_dict["away_rest_days"]   = away_rest_days
        features_dict["away_travel_dist"] = away_travel_dist

        # Ensure the feature vector is in the same order as FEATURE_COLS
        row = []
        for col in self.FEATURE_COLS:
            row.append(features_dict.get(col, 0.0))  # default to 0.0 if key is missing

        return np.array([row])  # shape (1, n_features)
