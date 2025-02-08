import pandas as pd

def transform_testing_data(input_csv="testing_data.csv", output_csv="testing_data.csv"):
    """
    Reads the original testing_data.csv (which has one row per team per game),
    merges home/away rows into one row per game, and writes out the new file
    in the format:
    
    game_id,team_home,team_away,rest_days_Home,rest_days_Away,
    travel_dist_Home,travel_dist_Away,team_score_Home,team_score_Away
    """
    # Read CSV, parsing 'NA' as missing values
    df = pd.read_csv(input_csv, na_values=["NA"])

    # Separate out home and away rows
    home_df = df[df['home_away'] == 'home'].copy()
    away_df = df[df['home_away'] == 'away'].copy()

    # Merge on game_id so we can line up home/away rows
    merged_df = pd.merge(
        home_df[['game_id','team','rest_days','travel_dist','team_score']],
        away_df[['game_id','team','rest_days','travel_dist','team_score']],
        on='game_id', 
        suffixes=('_Home','_Away')
    )

    # Rename columns to the desired final form
    merged_df.rename(columns={
        'team_Home': 'team_home',
        'team_Away': 'team_away',
        'rest_days_Home': 'rest_days_Home',
        'rest_days_Away': 'rest_days_Away',
        'travel_dist_Home': 'travel_dist_Home',
        'travel_dist_Away': 'travel_dist_Away',
        'team_score_Home': 'team_score_Home',
        'team_score_Away': 'team_score_Away'
    }, inplace=True)

    # Reorder columns to match the requested format
    final_cols = [
        'game_id', 
        'team_home', 
        'team_away',
        'rest_days_Home', 
        'rest_days_Away',
        'travel_dist_Home', 
        'travel_dist_Away',
        'team_score_Home', 
        'team_score_Away'
    ]
    merged_df = merged_df[final_cols]

    # Write out the transformed CSV
    merged_df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    # Run the transformation in-place (reading and writing the same CSV).
    transform_testing_data("games_data.csv", "games_data_testing.csv")
