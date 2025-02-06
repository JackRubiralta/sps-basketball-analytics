"""
Advanced NCAA Tournament Analytics – Phase 1
============================================

This module performs three main tasks:
  1. Data cleaning & feature engineering on 10,000+ game records.
  2. Aggregation of team-level performance metrics and ranking (for West, North, South regions).
  3. Building a matchup-level model (using differential aggregated features plus rest/travel info)
     that predicts the probability that the “stronger” team wins.
     
Files expected:
   - games_data.csv          (the season’s game records; each game appears twice, one per team)
   - team_to_predict.csv     (10 matchups – including play-ins – with seeds, rest, travel, etc.)
   - team_regions.csv        (mapping of team names to regions)
   
Note:
  In the season games CSV, columns like "rest_days" and "travel_dist" become 
  "rest_days_home"/"rest_days_away" and "travel_dist_home"/"travel_dist_away" after merging.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb

def load_games_data(filepath='games_data.csv'):
    """Load and clean the game data."""
    games = pd.read_csv(filepath)
    games['game_date'] = pd.to_datetime(games['game_date'], format='%m/%d/%Y', errors='coerce')
    games['FG2_pct'] = games.apply(lambda row: row['FGM_2'] / row['FGA_2'] if row['FGA_2'] > 0 else 0, axis=1)
    games['FG3_pct'] = games.apply(lambda row: row['FGM_3'] / row['FGA_3'] if row['FGA_3'] > 0 else 0, axis=1)
    games['FT_pct']  = games.apply(lambda row: row['FTM']  / row['FTA']  if row['FTA']  > 0 else 0, axis=1)
    games['win']     = (games['team_score'] > games['opponent_team_score']).astype(int)
    games['margin']  = games['team_score'] - games['opponent_team_score']
    return games

def get_team_regions(filepath='team_regions.csv'):
    """Load the team-to-region mapping."""
    return pd.read_csv(filepath)

def aggregate_team_stats(games, team_regions):
    """
    Aggregate team-level statistics over the season,
    compute a composite rating, and merge team region info.
    """
    team_stats = games.groupby('team').agg({
        'win': 'mean',
        'margin': 'mean',
        'FG2_pct': 'mean',
        'FG3_pct': 'mean',
        'FT_pct': 'mean',
        'AST': 'mean',
        'BLK': 'mean',
        'STL': 'mean',
        'TOV': 'mean',
        'DREB': 'mean',
        'OREB': 'mean'
    }).reset_index()
    team_stats.rename(columns={'win': 'win_rate'}, inplace=True)
    team_stats['games_played'] = games.groupby('team').size().values

    # Choose key metrics and build a composite rating (the higher, the better)
    rating_features = ['win_rate', 'margin', 'FG2_pct', 'FG3_pct', 'FT_pct', 'STL']
    scaler = StandardScaler()
    team_stats_scaled = team_stats.copy()
    team_stats_scaled[rating_features] = scaler.fit_transform(team_stats[rating_features])
    team_stats['composite_rating'] = team_stats_scaled[rating_features].sum(axis=1)

    team_stats = pd.merge(team_stats, team_regions, on='team', how='left')
    return team_stats

def build_matchups(games, team_stats):
    """
    Build matchup-level features from season games.
    
    For each game (which appears twice in the data: once per team),
    merge the home and away rows and then assign the “stronger” team based on composite rating.
    Also, compute the actual point margin (margin_diff_actual) from the game.
    """
    games_home = games[games['home_away'] == 'home']
    games_away = games[games['home_away'] == 'away']
    matchups = pd.merge(games_home, games_away, on='game_id', suffixes=('_home', '_away'))
    
    # Merge composite ratings for home and away teams
    matchups = pd.merge(matchups, team_stats[['team', 'composite_rating']],
                          left_on='team_home', right_on='team', how='left')
    matchups.rename(columns={'composite_rating': 'composite_rating_home'}, inplace=True)
    matchups.drop('team', axis=1, inplace=True)
    matchups = pd.merge(matchups, team_stats[['team', 'composite_rating']],
                          left_on='team_away', right_on='team', how='left')
    matchups.rename(columns={'composite_rating': 'composite_rating_away'}, inplace=True)
    matchups.drop('team', axis=1, inplace=True)
    
    def assign_stronger(row):
        if row['composite_rating_home'] >= row['composite_rating_away']:
            margin_diff_actual = row['team_score_home'] - row['team_score_away']
            return pd.Series({
                'game_id': row['game_id'],
                'stronger_team': row['team_home'],
                'weaker_team': row['team_away'],
                'stronger_win': 1 if row['team_score_home'] > row['team_score_away'] else 0,
                'rest_diff': row['rest_days_home'] - row['rest_days_away'],
                'travel_diff': row['travel_dist_home'] - row['travel_dist_away'],
                'composite_rating_diff': row['composite_rating_home'] - row['composite_rating_away'],
                'margin_diff_actual': margin_diff_actual
            })
        else:
            margin_diff_actual = row['team_score_away'] - row['team_score_home']
            return pd.Series({
                'game_id': row['game_id'],
                'stronger_team': row['team_away'],
                'weaker_team': row['team_home'],
                'stronger_win': 1 if row['team_score_away'] > row['team_score_home'] else 0,
                'rest_diff': row['rest_days_away'] - row['rest_days_home'],
                'travel_diff': row['travel_dist_away'] - row['travel_dist_home'],
                'composite_rating_diff': row['composite_rating_away'] - row['composite_rating_home'],
                'margin_diff_actual': margin_diff_actual
            })
    matchup_features = matchups.apply(assign_stronger, axis=1)
    
    # Add additional differential features based on aggregated team stats.
    agg_features = team_stats.set_index('team')[['win_rate', 'margin', 'FG2_pct', 'FG3_pct', 'FT_pct', 'STL']].to_dict('index')
    def add_feature_diffs(row):
        s_team = row['stronger_team']
        w_team = row['weaker_team']
        if s_team in agg_features and w_team in agg_features:
            row['win_rate_diff']  = agg_features[s_team]['win_rate'] - agg_features[w_team]['win_rate']
            row['margin_diff']    = agg_features[s_team]['margin'] - agg_features[w_team]['margin']
            row['FG2_pct_diff']   = agg_features[s_team]['FG2_pct'] - agg_features[w_team]['FG2_pct']
            row['FG3_pct_diff']   = agg_features[s_team]['FG3_pct'] - agg_features[w_team]['FG3_pct']
            row['FT_pct_diff']    = agg_features[s_team]['FT_pct'] - agg_features[w_team]['FT_pct']
            row['STL_diff']       = agg_features[s_team]['STL'] - agg_features[w_team]['STL']
        else:
            row['win_rate_diff'] = 0
            row['margin_diff'] = 0
            row['FG2_pct_diff'] = 0
            row['FG3_pct_diff'] = 0
            row['FT_pct_diff'] = 0
            row['STL_diff'] = 0
        return row
    matchup_features = matchup_features.apply(add_feature_diffs, axis=1)
    return matchup_features

def get_feature_cols():
    """Return the list of feature columns used for model training."""
    return ['composite_rating_diff', 'rest_diff', 'travel_diff',
            'win_rate_diff', 'margin_diff', 'FG2_pct_diff', 
            'FG3_pct_diff', 'FT_pct_diff', 'STL_diff']

def train_win_probability_model(matchup_features, feature_cols):
    """
    Train an XGBoost classifier on the matchup features to predict whether
    the stronger team wins. Returns the trained model and test splits.
    """
    X = matchup_features[feature_cols]
    y = matchup_features['stronger_win']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print("Model training complete. ROC AUC on hold-out data: {:.3f}".format(auc))
    return model, X_test, y_test

def get_agg_features(team_stats):
    """Return a dictionary of aggregated team features for prediction."""
    return team_stats.set_index('team')[['win_rate', 'margin', 'FG2_pct', 'FG3_pct', 'FT_pct', 'STL']].to_dict('index')

def main():
    # Load and prepare data
    games = load_games_data('games_data.csv')
    team_regions = get_team_regions('team_regions.csv')
    team_stats = aggregate_team_stats(games, team_regions)
    
    # Rank teams (Top 16 for West, North, and South regions)
    print("===== TEAM RANKINGS (Top 16 per Region: West, North, South) =====")
    for region in ['West', 'North', 'South']:
        region_teams = team_stats[team_stats['region'] == region].sort_values(by='composite_rating', ascending=False)
        print(f"\n--- {region} Region ---")
        print(region_teams[['team', 'composite_rating', 'win_rate', 'margin']].head(16).to_string(index=False))
    
    # Build matchup-level features
    matchup_features = build_matchups(games, team_stats)
    
    # Train win probability model
    feature_cols = get_feature_cols()
    model, X_test, y_test = train_win_probability_model(matchup_features, feature_cols)
    
    # Tournament predictions (using team_to_predict.csv)
    team_pred = pd.read_csv('team_to_predict.csv')
    def determine_higher_team(row):
        # Lower numerical seed is stronger.
        if row['seed_home'] < row['seed_away']:
            row['higher_team'] = row['team_home']
            row['lower_team'] = row['team_away']
            row['rest_diff'] = row['rest_days_Home'] - row['rest_days_Away']
            row['travel_diff'] = row['travel_dist_Home'] - row['travel_dist_Away']
        else:
            row['higher_team'] = row['team_away']
            row['lower_team'] = row['team_home']
            row['rest_diff'] = row['rest_days_Away'] - row['rest_days_Home']
            row['travel_diff'] = row['travel_dist_Away'] - row['travel_dist_Home']
        return row
    team_pred = team_pred.apply(determine_higher_team, axis=1)
    agg_features = get_agg_features(team_stats)
    def add_pred_features(row):
        ht = row['higher_team']
        lt = row['lower_team']
        if ht in agg_features and lt in agg_features:
            row['win_rate_diff'] = agg_features[ht]['win_rate'] - agg_features[lt]['win_rate']
            row['margin_diff'] = agg_features[ht]['margin'] - agg_features[lt]['margin']
            row['FG2_pct_diff'] = agg_features[ht]['FG2_pct'] - agg_features[lt]['FG2_pct']
            row['FG3_pct_diff'] = agg_features[ht]['FG3_pct'] - agg_features[lt]['FG3_pct']
            row['FT_pct_diff'] = agg_features[ht]['FT_pct'] - agg_features[lt]['FT_pct']
            row['STL_diff'] = agg_features[ht]['STL'] - agg_features[lt]['STL']
            # Lookup composite ratings from team_stats
            comp_rating_ht = team_stats.loc[team_stats['team'] == ht, 'composite_rating'].values[0]
            comp_rating_lt = team_stats.loc[team_stats['team'] == lt, 'composite_rating'].values[0]
            row['composite_rating_diff'] = comp_rating_ht - comp_rating_lt
        else:
            row['win_rate_diff'] = 0
            row['margin_diff'] = 0
            row['FG2_pct_diff'] = 0
            row['FG3_pct_diff'] = 0
            row['FT_pct_diff'] = 0
            row['STL_diff'] = 0
            row['composite_rating_diff'] = 0
        return row
    team_pred = team_pred.apply(add_pred_features, axis=1)
    team_pred_features = team_pred[get_feature_cols()]
    team_pred['predicted_prob'] = model.predict_proba(team_pred_features)[:, 1]
    print("\n===== TOURNAMENT GAME PREDICTIONS =====")
    print(team_pred[['game_id', 'higher_team', 'predicted_prob']].to_string(index=False))
    
    # Print Methodology Summary
    methodology_summary = """
    Methodology Summary:
    1. Process:
       Raw game data was cleaned by parsing dates and computing shooting percentages, win indicators, and margins.
       Data were aggregated at the team level and standardized to form a composite rating.
    
    2. Tools and Techniques:
       Python (Pandas, NumPy), scikit-learn, and XGBoost were used for data manipulation, model training, and evaluation.
    
    3. Statistical Methods:
       Differential features (including rest/travel differences and differences in aggregated stats) were computed.
       An XGBoost classifier was trained to predict the win probability of the stronger team, with performance validated via ROC AUC.
    """
    print(methodology_summary)

if __name__ == '__main__':
    main()
