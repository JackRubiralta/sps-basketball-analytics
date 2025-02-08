# games_data.py

import pandas as pd
import numpy as np
import os

class GamesData:
    """
    Manages data pipeline for training and prediction:
    - Loads 'games_data.csv'
    - Computes team average stats
    - Merges home/away rows into one row per game
    - Creates difference & ratio features
    - Provides final (X, y)
    """

    # The columns we average for each team
    _cols_to_avg = [
        "FGA_2", "FGM_2", "FGA_3", "FGM_3",
        "FTA",   "FTM",   "AST",   "BLK",
        "STL",   "TOV",   "DREB",  "OREB",
        "F_personal"
    ]

    def __init__(self, csv_path: str):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{csv_path} does not exist.")
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)

        # Precompute the team averages
        self.team_avgs = self._compute_team_averages(self.df)

        # We'll define a base set of columns for features
        # (home_avg_*), (away_avg_*), plus rest/travel
        self.base_feature_cols = []
        for col in self._cols_to_avg:
            self.base_feature_cols.append(f"home_avg_{col}")
        for col in self._cols_to_avg:
            self.base_feature_cols.append(f"away_avg_{col}")

        self.base_feature_cols += [
            "home_rest_days", "home_travel_dist",
            "away_rest_days", "away_travel_dist"
        ]

        # We'll store final feature column names in self.FEATURE_COLS
        # This will include difference & ratio columns
        self.FEATURE_COLS = list(self.base_feature_cols)  # start with base
        # We'll add difference/ratio columns dynamically after we see data

    def _compute_team_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Group by 'team' and compute mean of self._cols_to_avg
        """
        df_avg = df.groupby("team")[self._cols_to_avg].mean().reset_index()
        rename_dict = {col: f"avg_{col}" for col in self._cols_to_avg}
        df_avg.rename(columns=rename_dict, inplace=True)
        return df_avg

    def prepare_training_data(self) -> pd.DataFrame:
        """
        Merges home & away rows into one row per game, storing
        home_* stats, away_* stats, plus the label 'home_team_won'.
        """
        df_team_avgs = self.team_avgs

        df_home = self.df[self.df["home_away"] == "home"].copy()
        df_away = self.df[self.df["home_away"] == "away"].copy()

        # Merge each side with the team averages
        df_home = df_home.merge(df_team_avgs, on="team", how="left")
        df_away = df_away.merge(df_team_avgs, on="team", how="left")

        # Rename columns with "home_" or "away_"
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

        # Merge home & away frames on 'game_id'
        merged = pd.merge(df_home, df_away, on="game_id", how="inner")

        # Label: did the home team win?
        merged["home_team_won"] = (merged["home_team_score"] > merged["away_team_score"]).astype(int)

        return merged

    def get_feature_and_label_arrays(self, merged_df: pd.DataFrame, label_col: str = "home_team_won"):
        """
        1) Extract initial feature array from base_feature_cols
        2) Add difference & ratio features
        3) Return (X, y)
        """
        # 1) Build a DataFrame with the base features
        # (We must ensure these columns exist in merged_df)
        for col in self.base_feature_cols:
            if col not in merged_df.columns:
                merged_df[col] = 0.0  # fill missing

        X_base = merged_df[self.base_feature_cols].values
        y = merged_df[label_col].values

        # 2) Create difference & ratio features dynamically
        # We'll do them for each stat in self._cols_to_avg
        # e.g. diff_FGA_2 = home_avg_FGA_2 - away_avg_FGA_2
        #      ratio_FGA_2 = (home_avg_FGA_2 + 1) / (away_avg_FGA_2 + 1)
        # The +1 avoids division by zero
        diff_ratio_features = []
        diff_ratio_names = []

        # We'll build them for each row in X_base
        # We'll need the columns in the correct order:
        # home_avg_FGA_2 is index 0 in the first block,
        # away_avg_FGA_2 is index (len(self._cols_to_avg)) in the second block
        # Then we have rest/travel. We'll isolate stats first to do diffs

        n_stats = len(self._cols_to_avg)  # e.g. 13
        # Indices for home stats: 0..(n_stats-1)
        # Indices for away stats: n_stats..(2*n_stats-1)

        for i_row in range(X_base.shape[0]):
            row_vals = X_base[i_row]
            # We'll create a place to store difference+ratio for each stat
            row_diff_ratio = []
            for i_stat in range(n_stats):
                home_val = row_vals[i_stat]
                away_val = row_vals[n_stats + i_stat]

                diff = home_val - away_val
                ratio = (home_val + 1e-6) / (away_val + 1e-6)  # avoid zero

                row_diff_ratio.append(diff)
                row_diff_ratio.append(ratio)

            diff_ratio_features.append(row_diff_ratio)

        # Build the final array of difference & ratio features
        diff_ratio_features = np.array(diff_ratio_features, dtype=np.float32)

        # Now let's define the names for these columns
        for stat_col in self._cols_to_avg:
            diff_name = f"diff_{stat_col}"
            ratio_name = f"ratio_{stat_col}"
            diff_ratio_names.append(diff_name)
            diff_ratio_names.append(ratio_name)

        # 3) Combine X_base with these new features
        X = np.concatenate([X_base, diff_ratio_features], axis=1)

        # Update self.FEATURE_COLS if not done yet
        # Base plus the new difference & ratio columns
        if len(self.FEATURE_COLS) == len(self.base_feature_cols):
            self.FEATURE_COLS += diff_ratio_names

        return X, y

    def build_prediction_features(self,
                                  team_home: str,
                                  team_away: str,
                                  home_rest_days: float,
                                  home_travel_dist: float,
                                  away_rest_days: float,
                                  away_travel_dist: float) -> np.ndarray:
        """
        Creates a single 1-row feature vector for a new matchup.
        1) Look up home & away average stats
        2) Insert into base feature array
        3) Add difference & ratio features
        """
        # 1) Retrieve home/away average stats
        home_avgs = self.team_avgs[self.team_avgs["team"] == team_home]
        away_avgs = self.team_avgs[self.team_avgs["team"] == team_away]

        if home_avgs.empty:
            raise ValueError(f"No average stats found for home team '{team_home}'")
        if away_avgs.empty:
            raise ValueError(f"No average stats found for away team '{team_away}'")

        # For convenience, let's create a dictionary of required fields
        features_dict = {}

        # Insert home avg stats
        for col in self._cols_to_avg:
            home_val = float(home_avgs.iloc[0][f"avg_{col}"])
            features_dict[f"home_avg_{col}"] = home_val
        # Insert away avg stats
        for col in self._cols_to_avg:
            away_val = float(away_avgs.iloc[0][f"avg_{col}"])
            features_dict[f"away_avg_{col}"] = away_val

        # Insert rest/travel
        features_dict["home_rest_days"] = home_rest_days
        features_dict["home_travel_dist"] = home_travel_dist
        features_dict["away_rest_days"] = away_rest_days
        features_dict["away_travel_dist"] = away_travel_dist

        # Build the base array
        row_base = []
        for col in self.base_feature_cols:
            row_base.append(features_dict.get(col, 0.0))

        row_base = np.array(row_base, dtype=np.float32).reshape(1, -1)

        # 2) Now build difference+ratio features for the single row
        n_stats = len(self._cols_to_avg)
        diffratio_row = []
        for i_stat, stat_col in enumerate(self._cols_to_avg):
            home_val = row_base[0][i_stat]
            away_val = row_base[0][n_stats + i_stat]

            diff = home_val - away_val
            ratio = (home_val + 1e-6) / (away_val + 1e-6)
            diffratio_row.append(diff)
            diffratio_row.append(ratio)

        diffratio_array = np.array(diffratio_row, dtype=np.float32).reshape(1, -1)

        # 3) Concatenate base + difference/ratio
        X_single = np.concatenate([row_base, diffratio_array], axis=1)
        return X_single
