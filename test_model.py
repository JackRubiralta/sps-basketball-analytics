import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from v1 import load_games_data, get_team_regions, aggregate_team_stats, build_matchups, get_feature_cols, train_win_probability_model

def test_model_performance():
    # Load and prepare data
    games = load_games_data('games_data.csv')
    team_regions = get_team_regions('team_regions.csv')
    team_stats = aggregate_team_stats(games, team_regions)
    matchup_features = build_matchups(games, team_stats)
    feature_cols = get_feature_cols()
    
    # Train the model (using a hold-out test split for evaluation)
    model, X_test, y_test = train_win_probability_model(matchup_features, feature_cols)
    
    # Get predicted win probabilities for all matchups
    X_all = matchup_features[feature_cols]
    matchup_features['predicted_prob'] = model.predict_proba(X_all)[:, 1]
    
    # Compare predicted probability with the actual point margin difference.
    # (margin_diff_actual is computed in build_matchups and is positive if the stronger team won by that margin,
    # and negative if the stronger team lost.)
    corr, p_value = pearsonr(matchup_features['predicted_prob'], matchup_features['margin_diff_actual'])
    print(f"\nPearson correlation between predicted win probability and actual margin difference: {corr:.3f} (p-value: {p_value:.3e})")
    
    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(matchup_features['predicted_prob'], matchup_features['margin_diff_actual'], alpha=0.5)
    plt.xlabel("Predicted Win Probability (Stronger Team)")
    plt.ylabel("Actual Margin Difference (Points)")
    plt.title("Predicted Win Probability vs. Actual Margin Difference")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    test_model_performance()
