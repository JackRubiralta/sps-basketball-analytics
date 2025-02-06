"""
Improved NCAA Tournament Analytics Model
==========================================

This module performs the following tasks:
  1. Loads and cleans season game data (games_data.csv) and team regions (team_regions.csv).
  2. Aggregates team-level statistics and computes composite ratings.
  3. Builds matchup-level features from game data, including additional differential features.
  4. Trains an XGBoost classifier on differential features using RandomizedSearchCV for hyperparameter tuning and early stopping.
  5. Saves the trained model to disk.
  6. Loads tournament prediction data (team_to_predict.csv) and predicts outcomes.
  7. Provides various utility functions for model training, saving, loading, and prediction.

Usage:
    Run this module directly to train the model and generate tournament predictions.
    Use the `run()` function as the main entry point.
"""

import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_games_data(filepath='games_data.csv'):
    """Load and clean the season game data."""
    logger.info("Loading games data from %s", filepath)
    games = pd.read_csv(filepath)
    games['game_date'] = pd.to_datetime(games['game_date'], format='%m/%d/%Y', errors='coerce')
    # Compute shooting percentages (guard against division by zero)
    games['FG2_pct'] = games.apply(lambda row: row['FGM_2'] / row['FGA_2'] if row['FGA_2'] > 0 else 0, axis=1)
    games['FG3_pct'] = games.apply(lambda row: row['FGM_3'] / row['FGA_3'] if row['FGA_3'] > 0 else 0, axis=1)
    games['FT_pct']  = games.apply(lambda row: row['FTM']  / row['FTA']  if row['FTA']  > 0 else 0, axis=1)
    # Create win indicator and margin variables
    games['win']    = (games['team_score'] > games['opponent_team_score']).astype(int)
    games['margin'] = games['team_score'] - games['opponent_team_score']
    logger.info("Games data loaded: %d rows", len(games))
    return games

def load_team_regions(filepath='team_regions.csv'):
    """Load the team-to-region mapping."""
    logger.info("Loading team regions from %s", filepath)
    return pd.read_csv(filepath)

def aggregate_team_stats(games, team_regions):
    """
    Aggregate team-level statistics and compute a composite rating.
    
    The composite rating is calculated by standardizing a set of key metrics
    (win rate, margin, shooting percentages, and steals) and summing the values.
    """
    logger.info("Aggregating team statistics")
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

    rating_features = ['win_rate', 'margin', 'FG2_pct', 'FG3_pct', 'FT_pct', 'STL']
    scaler = StandardScaler()
    team_stats_scaled = team_stats.copy()
    team_stats_scaled[rating_features] = scaler.fit_transform(team_stats[rating_features])
    team_stats['composite_rating'] = team_stats_scaled[rating_features].sum(axis=1)

    team_stats = pd.merge(team_stats, team_regions, on='team', how='left')
    logger.info("Aggregated team stats for %d teams", len(team_stats))
    return team_stats

def build_matchup_features(games, team_stats):
    """
    Build matchup-level features from season games.
    
    Each game appears twice (once for each team). We merge the home and away rows
    (using the game_id) and designate the team with the higher composite rating as the
    "stronger" team. Differential features (e.g., differences in rest days, travel distances,
    and key statistical metrics) are computed along with the actual game margin difference.
    """
    logger.info("Building matchup features")
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
    
    # Add basic differential features based on aggregated team stats.
    agg_features = team_stats.set_index('team')[['win_rate', 'margin', 'FG2_pct', 'FG3_pct', 'FT_pct', 'STL']].to_dict('index')
    def add_feature_diffs(row):
        s_team = row['stronger_team']
        w_team = row['weaker_team']
        if s_team in agg_features and w_team in agg_features:
            row['win_rate_diff'] = agg_features[s_team]['win_rate'] - agg_features[w_team]['win_rate']
            row['margin_diff']   = agg_features[s_team]['margin'] - agg_features[w_team]['margin']
            row['FG2_pct_diff']  = agg_features[s_team]['FG2_pct'] - agg_features[w_team]['FG2_pct']
            row['FG3_pct_diff']  = agg_features[s_team]['FG3_pct'] - agg_features[w_team]['FG3_pct']
            row['FT_pct_diff']   = agg_features[s_team]['FT_pct'] - agg_features[w_team]['FT_pct']
            row['STL_diff']      = agg_features[s_team]['STL'] - agg_features[w_team]['STL']
        else:
            row['win_rate_diff'] = 0
            row['margin_diff']   = 0
            row['FG2_pct_diff']  = 0
            row['FG3_pct_diff']  = 0
            row['FT_pct_diff']   = 0
            row['STL_diff']      = 0
        return row
    matchup_features = matchup_features.apply(add_feature_diffs, axis=1)
    
    # --- NEW: Add additional differential features ---
    def add_extra_diff_features(row):
        # Using additional stats: AST, BLK, TOV, DREB, OREB
        agg_full = team_stats.set_index('team')[['AST', 'BLK', 'TOV', 'DREB', 'OREB']].to_dict('index')
        s_team = row['stronger_team']
        w_team = row['weaker_team']
        if s_team in agg_full and w_team in agg_full:
            row['AST_diff'] = agg_full[s_team]['AST'] - agg_full[w_team]['AST']
            row['BLK_diff'] = agg_full[s_team]['BLK'] - agg_full[w_team]['BLK']
            row['TOV_diff'] = agg_full[s_team]['TOV'] - agg_full[w_team]['TOV']
            row['DREB_diff'] = agg_full[s_team]['DREB'] - agg_full[w_team]['DREB']
            row['OREB_diff'] = agg_full[s_team]['OREB'] - agg_full[w_team]['OREB']
        else:
            row['AST_diff'] = 0
            row['BLK_diff'] = 0
            row['TOV_diff'] = 0
            row['DREB_diff'] = 0
            row['OREB_diff'] = 0
        return row
    matchup_features = matchup_features.apply(add_extra_diff_features, axis=1)
    
    logger.info("Built matchup features for %d games", len(matchup_features))
    return matchup_features

def get_feature_cols():
    """Return the list of feature columns used for training."""
    return ['composite_rating_diff', 'rest_diff', 'travel_diff',
            'win_rate_diff', 'margin_diff', 'FG2_pct_diff', 
            'FG3_pct_diff', 'FT_pct_diff', 'STL_diff',
            'AST_diff', 'BLK_diff', 'TOV_diff', 'DREB_diff', 'OREB_diff']

def train_model(matchup_features, feature_cols, use_grid_search=False):
    """
    Train an XGBoost classifier to predict whether the stronger team wins.
    
    If use_grid_search is True, perform hyperparameter tuning via RandomizedSearchCV.
    Returns the trained model.
    """
    logger.info("Training model...")
    X = matchup_features[feature_cols]
    y = matchup_features['stronger_win']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if use_grid_search:
        param_grid = {
            'max_depth': [3, 5, 7, 9],
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.2, 0.5]
        }
        xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        random_search = RandomizedSearchCV(
            xgb_clf,
            param_distributions=param_grid,
            n_iter=20,
            cv=3,
            scoring='roc_auc',
            verbose=1,
            random_state=42
        )
        random_search.fit(X_train, y_train)
        model = random_search.best_estimator_
        logger.info("Random search best parameters: %s", random_search.best_params_)
    else:
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        # Use early stopping: require an evaluation set
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=10,
            verbose=False
        )
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    logger.info("Model training complete. ROC AUC on hold-out data: %.3f", auc)
    return model

def save_model(model, filename='tournament_model.joblib'):
    """Save the trained model to disk."""
    joblib.dump(model, filename)
    logger.info("Model saved to %s", filename)

def load_model(filename='tournament_model.joblib'):
    """Load a trained model from disk."""
    model = joblib.load(filename)
    logger.info("Model loaded from %s", filename)
    return model

def predict_tournament(model, team_stats, tournament_file='team_to_predict.csv'):
    """
    Predict outcomes for tournament matchups from the provided tournament file.
    
    The tournament file is expected to have the columns:
      game_id, description, team_home, team_away, seed_home, seed_away,
      home_away_NS, rest_days_Home, rest_days_Away, travel_dist_Home, travel_dist_Away, WINNING %
    
    The function determines the higher seeded team (lower numerical seed), computes differential
    features based on team_stats, and returns a DataFrame of predictions.
    """
    logger.info("Loading tournament data from %s", tournament_file)
    tourney = pd.read_csv(tournament_file)
    
    def determine_higher_team(row):
        # Lower seed number is considered stronger.
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
    
    tourney = tourney.apply(determine_higher_team, axis=1)
    
    # Use aggregated team stats to compute differential features.
    agg_features = team_stats.set_index('team')[['win_rate', 'margin', 'FG2_pct', 'FG3_pct', 'FT_pct', 'STL', 
                                                   'AST', 'BLK', 'TOV', 'DREB', 'OREB']].to_dict('index')
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
            row['AST_diff'] = agg_features[ht]['AST'] - agg_features[lt]['AST']
            row['BLK_diff'] = agg_features[ht]['BLK'] - agg_features[lt]['BLK']
            row['TOV_diff'] = agg_features[ht]['TOV'] - agg_features[lt]['TOV']
            row['DREB_diff'] = agg_features[ht]['DREB'] - agg_features[lt]['DREB']
            row['OREB_diff'] = agg_features[ht]['OREB'] - agg_features[lt]['OREB']
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
            row['AST_diff'] = 0
            row['BLK_diff'] = 0
            row['TOV_diff'] = 0
            row['DREB_diff'] = 0
            row['OREB_diff'] = 0
            row['composite_rating_diff'] = 0
        return row
    
    tourney = tourney.apply(add_pred_features, axis=1)
    feature_cols = get_feature_cols()
    X_tourney = tourney[feature_cols]
    tourney['predicted_prob'] = model.predict_proba(X_tourney)[:, 1]
    logger.info("Tournament predictions completed")
    return tourney[['game_id', 'description', 'higher_team', 'predicted_prob']]

def run():
    """
    Main entry point:
      1. Load training data and team regions.
      2. Aggregate team stats.
      3. Build matchup features.
      4. Train the model (optionally using randomized search tuning).
      5. Save the trained model.
      6. Predict tournament outcomes.
    """
    logger.info("Starting model training and tournament prediction process")
    games = load_games_data('games_data.csv')
    team_regions = load_team_regions('team_regions.csv')
    team_stats = aggregate_team_stats(games, team_regions)
    matchup_features = build_matchup_features(games, team_stats)
    feature_cols = get_feature_cols()
    # Set use_grid_search=True to enable hyperparameter tuning using RandomizedSearchCV.
    model = train_model(matchup_features, feature_cols, use_grid_search=True)
    save_model(model, 'tournament_model.joblib')
    tournament_predictions = predict_tournament(model, team_stats, tournament_file='team_to_predict.csv')
    logger.info("Tournament Predictions:\n%s", tournament_predictions.to_string(index=False))

if __name__ == '__main__':
    run()
