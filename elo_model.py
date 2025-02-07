"""
Machine Learning-based Elo Rating System for Teams

This module learns a multi-dimensional Elo rating for each team using only match outcome data and advanced modeling techniques. 
Each team's Elo rating is represented as a vector of real numbers. Higher values in this vector correspond to stronger performance in the latent factors captured by the model (e.g., offensive strength, defensive strength, consistency)&#8203;:contentReference[oaicite:0]{index=0}.

We leverage a neural network with an embedding layer to learn team representations (Elo vectors) directly from game results&#8203;:contentReference[oaicite:1]{index=1}. 
By training on win/loss outcomes (no external data or hand-crafted features), the model discovers abstract qualities that differentiate team performance&#8203;:contentReference[oaicite:2]{index=2}. 
This approach extends the classic Elo rating system into multiple dimensions, as demonstrated in research where embedding-enhanced Elo models outperform standard Elo&#8203;:contentReference[oaicite:3]{index=3}&#8203;:contentReference[oaicite:4]{index=4}.
"""
import os
import numpy as np
from typing import Dict, Any, Tuple, List
from games_data import GamesData

class Model:
    def __init__(self, model_file_path: str, elo_rating_file_path: str, games_data: GamesData):
        """
        We store a reference to the GamesData instance for use in training/predicting.
        If model_file_path exists, load the saved model from that path.
        Otherwise, self.model = None until we call 'generate_model' or 'load_model'.
        """
        self.elo_ratings = None # dict of team name to the vectors
        self.elo_rating_file_path = elo_rating_file_path
        self.model_file_path = model_file_path
        self.model = None
        self.games_data = games_data  # reference to your custom data class

        if os.path.exists(self.model_file_path):
            self.load_model()
            self.load_elo_ratings()
        else:
            print(f"Model file not found at {self.model_file_path}. "
                  f"Will need to generate a new model or load manually.")

   
    def generate_model(self):
        return self.model

    def generate_elo_ratings():
        # generates the vectors 
        return
       
    def save_elo_ratings(self):
        return
    
    def load_elo_ratings(self):
        return  
       
    def save_model(self):
        """
        Saves the model to self.model_file_path .
        """
        if self.model is None:
            return
       
        print(f"Model saved to {self.model_file_path}")

    def load_model(self):
        """
        Loads a model from self.model_file_path.
        """
        print(f"Model loaded from {self.model_file_path}")

class EloModel:
    def __init__(self, model_file_path: str, elo_rating_file_path: str, games_data: GamesData):
      
        self.elo_ratings = None # dict of team name to the vectors
        self.elo_rating_file_path = elo_rating_file_path
        self.model_file_path = model_file_path
        self.model = None
        self.games_data = games_data  # reference to your custom data class

        if os.path.exists(self.model_file_path):
            self.load_model()
            self.load_elo_ratings()
        else:
            print(f"Model file not found at {self.model_file_path}. "
                  f"Will need to generate a new model or load manually.")

   
    def generate_model(self):
        return self.model

    def generate_elo_ratings():
        # generates the vectors 
        return
       
    def save_elo_ratings(self):
        return
    
    def load_elo_ratings(self):
        return  
       
    def save_model(self):
        """
        Saves the model to self.model_file_path .
        """
        if self.model is None:
            return
       
        print(f"Model saved to {self.model_file_path}")

    def load_model(self):
        """
        Loads a model from self.model_file_path.
        """
        print(f"Model loaded from {self.model_file_path}")


def elo_number(elo_vector): 
    # magnitude / eucilidan norm of the vector
    # bigger ussuayly means better but the idea behind the vectors they could represent matchups where some vector beat others 