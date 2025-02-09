import pandas as pd
import numpy as np
import os

class GamesData:
    """
    Manages data pipeline for training and prediction, *including* a basic "team strength"
    metric. We:
    1) Load 'games_data.csv'
    2) Compute team average stats
    3) Compute team win percentage & strength-of-schedule
    4) Merge home/away rows into one row per game
    5) Create difference & ratio features
    6) Provide final (X, y)
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

        # Precompute the team averages (includes team_win_pct and team_sos)
        self.team_avgs = self._compute_team_averages(self.df)

        # We store the base set of columns for features:
        #   (home_avg_*), (away_avg_*), plus rest/travel
        #   plus newly introduced team strength features
        self.base_feature_cols = []
        for col in self._cols_to_avg:
            self.base_feature_cols.append(f"home_avg_{col}")
        for col in self._cols_to_avg:
            self.base_feature_cols.append(f"away_avg_{col}")

        # Additional "team quality" columns
        #   (team_win_pct, team_sos)
        self._extra_cols = ["team_win_pct", "team_sos"]

        for col in self._extra_cols:
            self.base_feature_cols.append(f"home_{col}")
        for col in self._extra_cols:
            self.base_feature_cols.append(f"away_{col}")

        # Add rest/travel columns
        self.base_feature_cols += [
            "home_rest_days", "home_travel_dist",
            "away_rest_days", "away_travel_dist"
        ]

        # We'll store final feature column names in self.FEATURE_COLS
        # This will include difference & ratio columns
        self.FEATURE_COLS = list(self.base_feature_cols)  # start with base

        # Stats that we want to do difference/ratio on:
        # (the columns we average) plus the new "team strength" features
        self._stats_for_diff = self._cols_to_avg + self._extra_cols

    def _compute_team_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        1) Group by 'team' and compute mean of self._cols_to_avg (basic stats)
        2) Compute 'team_win_pct' for each team
        3) Compute 'team_sos' as the average (opponent_win_pct)
        4) Merge them all into a single DataFrame
        """
        # -- Compute average basic stats --
        df_avg = df.groupby("team")[self._cols_to_avg].mean().reset_index()

        # -- Compute "win" for each row, i.e. if team_score > opponent_team_score --
        df_temp = df.copy()
        df_temp["win"] = (df_temp["team_score"] > df_temp["opponent_team_score"]).astype(int)

        # -- Team-level win percentage --
        df_team_wp = (
            df_temp.groupby("team")["win"]
            .mean()
            .reset_index()
            .rename(columns={"win": "team_win_pct"})
        )

        # -- Build a small table that has: (game_id, home_team, away_team) for each game
        df_home = (
            df_temp[df_temp["home_away"] == "home"][["game_id", "team"]]
            .rename(columns={"team": "home_team"})
        )
        df_away = (
            df_temp[df_temp["home_away"] == "away"][["game_id", "team"]]
            .rename(columns={"team": "away_team"})
        )
        df_opponents = pd.merge(df_home, df_away, on="game_id", how="inner")

        # -- 'melt' so each row has (team, opponent) --
        df_home_part = df_opponents[["game_id", "home_team", "away_team"]].rename(
            columns={"home_team": "team", "away_team": "opponent"}
        )
        df_away_part = df_opponents[["game_id", "home_team", "away_team"]].rename(
            columns={"away_team": "team", "home_team": "opponent"}
        )
        df_melted = pd.concat([df_home_part, df_away_part], ignore_index=True)

        # -- Merge to get the OPPONENT's team_win_pct --
        df_melted = df_melted.merge(
            df_team_wp, left_on="opponent", right_on="team", how="left", suffixes=("", "_opp")
        )
        df_melted.rename(columns={"team_win_pct": "opponent_win_pct"}, inplace=True)

        # -- Now group by 'team' to get the average opponent win pct => team_sos --
        df_team_sos = (
            df_melted.groupby("team")["opponent_win_pct"]
            .mean()
            .reset_index()
            .rename(columns={"opponent_win_pct": "team_sos"})
        )

        # -- Merge everything into df_avg: basic stats + (team_win_pct, team_sos) --
        df_avg = df_avg.merge(df_team_wp, on="team", how="left")
        df_avg = df_avg.merge(df_team_sos, on="team", how="left")

        # -- Finally, rename the basic stat columns from e.g. "FGA_2" -> "avg_FGA_2" --
        rename_dict = {col: f"avg_{col}" for col in self._cols_to_avg}
        df_avg.rename(columns=rename_dict, inplace=True)

        # So now df_avg has columns:
        #   ['team', 'avg_FGA_2', 'avg_FGM_2', ..., 'team_win_pct', 'team_sos']
        return df_avg

    def prepare_training_data(self) -> pd.DataFrame:
        """
        Merges home & away rows into one row per game, storing
        home_* stats, away_* stats, plus the label 'home_team_won'.
        Also includes new features: home_team_win_pct, home_team_sos, etc.
        """
        df_team_avgs = self.team_avgs

        df_home = self.df[self.df["home_away"] == "home"].copy()
        df_away = self.df[self.df["home_away"] == "away"].copy()

        # Merge each side with the team averages (including team_win_pct, team_sos)
        df_home = df_home.merge(df_team_avgs, on="team", how="left")
        df_away = df_away.merge(df_team_avgs, on="team", how="left")

        # Rename columns with "home_" or "away_"
        home_cols = {}
        away_cols = {}
        skip_cols = ["game_id", "team", "game_date", "home_away"]

        for col in df_home.columns:
            if col not in skip_cols:
                home_cols[col] = f"home_{col}"

        for col in df_away.columns:
            if col not in skip_cols:
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
        for col in self.base_feature_cols:
            if col not in merged_df.columns:
                merged_df[col] = 0.0  # fill missing if not present

        X_base = merged_df[self.base_feature_cols].values
        y = merged_df[label_col].values

        # 2) Create difference & ratio features for each stat in self._stats_for_diff.
        # We'll do them for home_avg_*, away_avg_* plus the new team_win_pct, team_sos.
        diff_ratio_features = []
        diff_ratio_names = []

        # figure out how many "stat" columns there are among the base features for home vs away
        n_stats = len(self._stats_for_diff)  # e.g. 13 basic + 2 new = 15
        # The ordering in base_feature_cols is: all home_avg_* for _cols_to_avg, then
        # all away_avg_* for _cols_to_avg, then home_team_win_pct, away_team_win_pct, home_team_sos, away_team_sos,
        # then rest/travel. We'll isolate the correct indices:

        # Indices for home: 0..(n_stats-1)
        # Indices for away: n_stats..(2*n_stats-1)
        # (Then come the rest/travel columns.)
        # We'll compute difference & ratio for each of these n_stats columns.

        for i_row in range(X_base.shape[0]):
            row_vals = X_base[i_row]
            row_diff_ratio = []
            for i_stat in range(n_stats):
                home_val = row_vals[i_stat]
                away_val = row_vals[n_stats + i_stat]

                diff = home_val - away_val
                ratio = (home_val + 1e-6) / (away_val + 1e-6)

                row_diff_ratio.append(diff)
                row_diff_ratio.append(ratio)
            diff_ratio_features.append(row_diff_ratio)

        diff_ratio_features = np.array(diff_ratio_features, dtype=np.float32)

        # Build column names
        for stat_col in self._stats_for_diff:
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
        1) Look up home & away average stats + (team_win_pct, team_sos)
        2) Insert into base feature array
        3) Add difference & ratio features
        """
        # 1) Retrieve home/away average stats + (team_win_pct, team_sos)
        home_avgs = self.team_avgs[self.team_avgs["team"] == team_home]
        away_avgs = self.team_avgs[self.team_avgs["team"] == team_away]

        if home_avgs.empty:
            raise ValueError(f"No average stats found for home team '{team_home}'")
        if away_avgs.empty:
            raise ValueError(f"No average stats found for away team '{team_away}'")

        # Prepare a dictionary for the required fields
        features_dict = {}

        # Insert home average stats
        for col in self._cols_to_avg:
            features_dict[f"home_avg_{col}"] = float(home_avgs.iloc[0][f"avg_{col}"])
        # Insert away average stats
        for col in self._cols_to_avg:
            features_dict[f"away_avg_{col}"] = float(away_avgs.iloc[0][f"avg_{col}"])

        # Insert the new "team strength" columns
        features_dict["home_team_win_pct"] = float(home_avgs.iloc[0]["team_win_pct"])
        features_dict["away_team_win_pct"] = float(away_avgs.iloc[0]["team_win_pct"])
        features_dict["home_team_sos"] = float(home_avgs.iloc[0]["team_sos"])
        features_dict["away_team_sos"] = float(away_avgs.iloc[0]["team_sos"])

        # Insert rest/travel
        features_dict["home_rest_days"] = home_rest_days
        features_dict["home_travel_dist"] = home_travel_dist
        features_dict["away_rest_days"] = away_rest_days
        features_dict["away_travel_dist"] = away_travel_dist

        # Build the base array in the correct order
        row_base = []
        for col in self.base_feature_cols:
            row_base.append(features_dict.get(col, 0.0))

        row_base = np.array(row_base, dtype=np.float32).reshape(1, -1)

        # 2) Build difference+ratio features for the single row
        n_stats = len(self._stats_for_diff)
        diffratio_row = []
        for i_stat, stat_col in enumerate(self._stats_for_diff):
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
