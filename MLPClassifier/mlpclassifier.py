from sentence_transformers import SentenceTransformer
import numpy as np
from sentence_transformers import SentenceTransformer
import joblib  # Para guardar/cargar el scaler

class MLPClassifierWrapper:
    def __init__(self, embedder):
        self.embedder = embedder
        self.mlp = joblib.load("MLPClassifier/model/mlp_model.pkl")


    def predict(self, user_input):
        X_new = self.embedder.encode([user_input], convert_to_numpy=True)
        prediction = self.mlp.predict(X_new)
        return prediction[0]