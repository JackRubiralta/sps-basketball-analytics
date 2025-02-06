import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, mean_squared_error
from scipy.stats import pearsonr
from model import (
    load_model,
    load_games_data,
    load_team_regions,
    aggregate_team_stats,
    build_matchup_features,
    get_feature_cols
)

def brier_score(y_true, y_prob):
    """Compute the Brier score for binary predictions."""
    return mean_squared_error(y_true, y_prob)

def decile_analysis(df, prob_col='predicted_prob', actual_col='margin_diff_actual'):
    """Group data into deciles and compute mean predicted probability and actual margin difference per decile."""
    df['decile'] = pd.qcut(df[prob_col], 10, labels=False)
    summary = df.groupby('decile').agg(
        mean_predicted_prob=(prob_col, 'mean'),
        mean_actual_margin=(actual_col, 'mean'),
        count=('game_id', 'count')
    ).reset_index()
    return summary

def test_model_graph_all_games():
    # 1. Load the pre-trained model
    model = load_model('tournament_model.joblib')
    
    # 2. Load and prepare the full season game data
    games = load_games_data('games_data.csv')
    team_regions = load_team_regions('team_regions.csv')
    team_stats = aggregate_team_stats(games, team_regions)
    matchup_features = build_matchup_features(games, team_stats)
    feature_cols = get_feature_cols()
    
    # 3. Compute predicted win probabilities for all matchups using the saved model
    X_all = matchup_features[feature_cols]
    matchup_features['predicted_prob'] = model.predict_proba(X_all)[:, 1]
    
    # 4. Compute evaluation metrics
    corr, p_value = pearsonr(matchup_features['predicted_prob'], matchup_features['margin_diff_actual'])
    print("=== Pearson Correlation ===")
    print(f"Correlation: {corr:.3f} (p-value: {p_value:.3e})")
    print("-> A higher positive correlation indicates that higher predicted probabilities tend to correspond with larger win margins.\n")
    
    matchup_features['predicted_win'] = (matchup_features['predicted_prob'] >= 0.5).astype(int)
    accuracy = accuracy_score(matchup_features['stronger_win'], matchup_features['predicted_win'])
    roc_auc = roc_auc_score(matchup_features['stronger_win'], matchup_features['predicted_prob'])
    cm = confusion_matrix(matchup_features['stronger_win'], matchup_features['predicted_win'])
    
    print("=== Classification Metrics ===")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"ROC AUC: {roc_auc:.3f}")
    print("Confusion Matrix:")
    print(cm)
    print("-> Accuracy is the fraction of games where the binary prediction (win if prob>=0.5) matches the actual outcome.")
    print("-> ROC AUC indicates the model's discrimination ability between wins and losses.\n")
    
    brier = brier_score(matchup_features['stronger_win'], matchup_features['predicted_prob'])
    print("=== Brier Score ===")
    print(f"Brier Score: {brier:.3f}")
    print("-> Lower Brier scores indicate better calibrated probability predictions.\n")
    
    print("=== Summary Statistics ===")
    print(f"Total games: {len(matchup_features)}")
    print(f"Mean predicted win probability: {matchup_features['predicted_prob'].mean():.3f}")
    print(f"Std. dev. of predicted win probability: {matchup_features['predicted_prob'].std():.3f}")
    print(f"Mean actual margin difference: {matchup_features['margin_diff_actual'].mean():.3f}")
    print(f"Std. dev. of actual margin difference: {matchup_features['margin_diff_actual'].std():.3f}\n")
    
    decile_summary = decile_analysis(matchup_features)
    print("=== Decile Analysis (by predicted win probability) ===")
    print(decile_summary.to_string(index=False))
    print("-> Decile analysis shows the average predicted probability and the corresponding actual margin difference in each 10% bucket.\n")
    
    # 5. Graphical Analysis
    plt.figure(figsize=(10, 6))
    plt.scatter(matchup_features['predicted_prob'], matchup_features['margin_diff_actual'], alpha=0.6, edgecolor='k')
    plt.xlabel("Predicted Win Probability (Stronger Team)")
    plt.ylabel("Actual Margin Difference (Points)")
    plt.title("Predicted Win Probability vs. Actual Margin Difference for All Games")
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.hist(matchup_features['predicted_prob'], bins=20, alpha=0.75, color='skyblue', edgecolor='black')
    plt.xlabel("Predicted Win Probability")
    plt.ylabel("Frequency")
    plt.title("Distribution of Predicted Win Probabilities")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    test_model_graph_all_games()
