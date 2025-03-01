import pandas as pd
import os

from model import Model
from games_data import GamesData

def main():
    # 1) Load team-region data
    team_regions_csv = "team_regions.csv"
    if not os.path.exists(team_regions_csv):
        raise FileNotFoundError(f"Could not find '{team_regions_csv}'.")

    df_team_regions = pd.read_csv(team_regions_csv)
    if "team" not in df_team_regions.columns or "region" not in df_team_regions.columns:
        raise ValueError("team_regions.csv must have columns ['team','region'].")

    # 2) Load model (and GamesData) so we can predict matchups
    games_data = GamesData("games_data.csv")
    model_file_path = "model_stuff.json"
    my_model = Model(model_file_path=model_file_path, games_data=games_data)

    # 3) We'll simulate every team playing every other team exactly once,
    #    ignoring actual region for the matchup (as requested).
    #    We'll accumulate each team's sum of "expected wins" across all matchups.
    teams = df_team_regions["team"].unique().tolist()

    # Prepare data structure for storing results
    # We'll track total_wins (as sum of expected probabilities) and total_games
    results_dict = {}
    for t in teams:
        region = df_team_regions.loc[df_team_regions["team"] == t, "region"].values[0]
        results_dict[t] = {
            "team": t,
            "region": region,
            "wins": 0.0,
            "games": 0
        }

    # 4) Loop over each pair (i, j), i != j
    #    We'll treat i as home and j as away, rest/travel = 0 for simplicity
    n_teams = len(teams)
    for i in range(n_teams):
        for j in range(i + 1, n_teams):
            team_i = teams[i]
            team_j = teams[j]

            # Predict probability of i beating j
            try:
                prob_i_win = my_model.predict_single_game(
                    team_home=team_i,
                    team_away=team_j,
                    home_rest_days=0.0,
                    home_travel_dist=0.0,
                    away_rest_days=0.0,
                    away_travel_dist=0.0
                )
            except ValueError as e:
                # If data doesn't exist for one of the teams, we can skip or raise.
                # Here, we'll raise to ensure user notices missing team info.
                raise ValueError(f"Issue predicting {team_i} vs {team_j}: {str(e)}")

            # We'll accumulate "expected" wins
            results_dict[team_i]["wins"] += prob_i_win
            results_dict[team_i]["games"] += 1
            results_dict[team_j]["wins"] += (1.0 - prob_i_win)
            results_dict[team_j]["games"] += 1

    # 5) Compute final win percentage for each team
    for t in teams:
        total_wins = results_dict[t]["wins"]
        total_games = results_dict[t]["games"]
        if total_games > 0:
            win_pct = total_wins / total_games
        else:
            win_pct = 0.0
        results_dict[t]["win_percentage"] = win_pct

    # 6) Convert results_dict to a DataFrame
    df_results = pd.DataFrame.from_records(list(results_dict.values()))

    # 7) Now we want to rank within each region by win_percentage (descending)
    df_results.sort_values(by=["region", "win_percentage"], ascending=[True, False], inplace=True)

    # Create the region_rank
    df_results["region_rank"] = (
        df_results.groupby("region")["win_percentage"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    # 8) Save to ranked_teams.csv
    output_csv_path = "ranked_teams.csv"
    # We'll keep only the requested columns
    df_results_out = df_results[["team", "region", "win_percentage", "region_rank"]]
    df_results_out.to_csv(output_csv_path, index=False)
    print(f"Ranked teams saved to '{output_csv_path}'.")

    # 9) Print order of teams for West, East, North, and South
    #    We'll filter df_results for each region in the order requested
    regions_to_print = ["West", "East", "North", "South"]  # user specifically requested this order

    for region in regions_to_print:
        df_region = df_results[df_results["region"] == region].copy()
        # Sort by rank
        df_region.sort_values(by="region_rank", inplace=True)
        print(f"\n=== {region.upper()} REGION RANKINGS ===")
        if df_region.empty:
            print("  (No teams found in this region.)")
        else:
            for _, row in df_region.iterrows():
                print(f"  Rank {row['region_rank']}: {row['team']} (Win %={row['win_percentage']:.3f})")


if __name__ == "__main__":
    main()
