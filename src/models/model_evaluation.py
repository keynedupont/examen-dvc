import pandas as pd
import numpy as np
import json
import os
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(X_test_path, y_test_path, model_pkl_path, predictions_output_path, metrics_output_path):
    """
    Évalue le modèle Elastic Net en calculant les métriques et génère les prédictions.
    
    Args:
    - X_test_path (str): Chemin vers le fichier CSV contenant les features de test.
    - y_test_path (str): Chemin vers le fichier CSV contenant les cibles de test.
    - model_pkl_path (str): Chemin vers le fichier .pkl contenant le modèle entraîné.
    - predictions_output_path (str): Chemin pour sauvegarder les prédictions.
    - metrics_output_path (str): Chemin pour sauvegarder les métriques dans un fichier .json.
    """
    # Charger les données de test
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()  # Convertir en vecteur 1D si nécessaire
    
    # Charger le modèle entraîné
    with open(model_pkl_path, 'rb') as file:
        model = pickle.load(file)
    
    # Effectuer les prédictions sur les données de test
    y_pred = model.predict(X_test)
    
    # Calculer les métriques d'évaluation
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Afficher les métriques
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R²: {r2}")
    
    # Sauvegarder les métriques dans un fichier JSON
    metrics = {
        "mean_squared_error": mse,
        "root_mean_squared_error": rmse,
        "mean_absolute_error": mae,
        "r2_score": r2
    }
    
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    with open(metrics_output_path, 'w') as file:
        json.dump(metrics, file)
    print("Les métriques d'évaluation ont été sauvegardées dans", metrics_output_path)
    
    # Sauvegarder les prédictions dans un fichier CSV
    predictions_df = pd.DataFrame({"Real": y_test, "Predicted": y_pred})
    os.makedirs(os.path.dirname(predictions_output_path), exist_ok=True)
    predictions_df.to_csv(predictions_output_path, index=False)
    print("Les prédictions ont été sauvegardées dans", predictions_output_path)

if __name__ == "__main__":
    # Définir les chemins d'entrée et de sortie
    X_test_file = "data/processed/X_test_scaled.csv"
    y_test_file = "data/processed/y_test.csv"
    model_pkl_file = "models/model_trained_elastic_net.pkl"
    predictions_output_file = "data/predictions.csv"
    metrics_output_file = "metrics/scores.json"

    # Évaluer le modèle et générer les prédictions
    evaluate_model(X_test_file, y_test_file, model_pkl_file, predictions_output_file, metrics_output_file)