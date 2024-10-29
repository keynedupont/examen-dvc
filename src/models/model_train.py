
import pandas as pd
from sklearn.linear_model import ElasticNet
import pickle
import os
import joblib
import yaml

#charger les paramètres depuis le fichier de config params.yaml
with open("params.yaml") as f:
    params = yaml.safe_load(f)

def train_elastic_net(X_train_path, y_train_path, params_pkl_path, output_dir):
    """
    Entraîne un modèle Elastic Net avec les meilleurs paramètres chargés depuis un fichier .pkl.
    
    Args:
    - X_train_path (str): Chemin vers le fichier CSV contenant les features d'entraînement.
    - y_train_path (str): Chemin vers le fichier CSV contenant les cibles d'entraînement.
    - params_pkl_path (str): Chemin vers le fichier .pkl contenant les meilleurs paramètres.
    - output_dir (str): Répertoire où stocker le modèle entraîné.
    """
    # Charger les données d'entraînement
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)

    # Charger les meilleurs paramètres depuis le fichier .pkl
    with open(params_pkl_path, 'rb') as file:
        best_params = pickle.load(file)
    
    print("Meilleurs paramètres chargés :", best_params)

    # Configurer et entraîner le modèle Elastic Net avec les meilleurs paramètres
    model = ElasticNet(**best_params)
    model.fit(X_train, y_train)
    print("Modèle Elastic Net entraîné avec succès.")

    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Sauvegarder le modèle entraîné dans un fichier .pkl
    model_pkl_path = os.path.join(output_dir, 'model_trained_elastic_net.pkl')
    with open(model_pkl_path, 'wb') as file:
        pickle.dump(model, file)
    
    print("Le modèle entraîné a été sauvegardé dans le fichier :", model_pkl_path)

if __name__ == "__main__":
    # Définir les chemins d'entrée et de sortie
    X_train_file = params['data_normalized_paths']['X_train_path']
    y_train_file = params['data_processed_paths']['y_train_path']
    best_params_pkl = params['models_paths']['best_params']  # Fichier .pkl contenant les meilleurs paramètres
    output_directory = params['models_paths']['trained_model_path']
    

    # Entraîner le modèle Elastic Net avec les meilleurs paramètres
    train_elastic_net(X_train_file, y_train_file, best_params_pkl, output_directory)