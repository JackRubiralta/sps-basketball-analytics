import pandas as pd
import os

def remove_incomplete_games(input_file, output_file):
    # Verify that the input file exists and is not empty.
    if not os.path.isfile(input_file):
        print(f"Error: The file {input_file} does not exist.")
        return
    if os.path.getsize(input_file) == 0:
        print(f"Error: The file {input_file} is empty.")
        return

    # Read the CSV file into a DataFrame
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Check that the required column exists
    if 'notD1_incomplete' not in df.columns:
        print("Error: 'notD1_incomplete' column not found in the CSV.")
        return

    # Convert the 'notD1_incomplete' column to boolean.
    # This treats any case variation of 'TRUE' as True, everything else as False.
    df['notD1_incomplete'] = df['notD1_incomplete'].astype(str).str.upper() == 'TRUE'

    # Count total games before filtering
    total_games = df['game_id'].nunique()
    total_rows = len(df)
    
    # Filter out games where any team has notD1_incomplete == True.
    # The lambda function returns False if any row in the group is True.
    df_filtered = df.groupby('game_id').filter(lambda group: not group['notD1_incomplete'].any())

    # Calculate the number of games and rows removed.
    remaining_games = df_filtered['game_id'].nunique()
    remaining_rows = len(df_filtered)
    removed_games = total_games - remaining_games
    removed_rows = total_rows - remaining_rows

    print(f"Total games before filtering: {total_games} ({total_rows} rows)")
    print(f"Removed {removed_games} games, totaling {removed_rows} rows.")
    print(f"Remaining games: {remaining_games} ({remaining_rows} rows)")

    # Write the filtered DataFrame to the output CSV file
    try:
        df_filtered.to_csv(output_file, index=False)
        print(f"Filtered data has been saved to {output_file}")
    except Exception as e:
        print(f"Error writing output CSV: {e}")

if __name__ == '__main__':
    # Adjust the file paths as necessary.
    input_csv = r'games_data.csv'
    output_csv = r'games_data1.csv'
    remove_incomplete_games(input_csv, output_csv)
