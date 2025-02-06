# generate_model.py

from model import Model
from games_data import GamesData

def main():
    # 1) Create a GamesData instance for 'training_data.csv'
    training_csv_path = "games_data.csv"
    games_data = GamesData(training_csv_path)

    # 2) Create the Model, passing the same file path for saving
    model_file_path = "model_stuff.json"
    my_model = Model(model_file_path=model_file_path, games_data=games_data)

    # 3) Generate (train) the model
    print("Training model (with hyperparameter search)...")
    my_model.generate_model(do_hyperparam_search=True)
    print("Done training.")

    # 4) Save the model
    my_model.save_model()
    print("Saved model.")

if __name__ == "__main__":
    main()
