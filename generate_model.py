from model import Model
from games_data import GamesData

def main():
    # 1) Create a GamesData instance for 'games_data.csv'
    training_csv_path = "games_data.csv"
    games_data = GamesData(training_csv_path)

    # 2) Create the Model
    model_file_path = "model_stuff.json" 
    my_model = Model(model_file_path=model_file_path, games_data=games_data)

    # 3) Generate (train) the model, doing hyperparameter search
    print("Training model (with advanced hyperparameter search)...")
    my_model.generate_model(do_hyperparam_search=True)
    print("Done training model.")

    # 4) Save the model
    my_model.save_model()
    print(f"Saved model to '{model_file_path}'.")

if __name__ == "__main__":
    main()
