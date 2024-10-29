import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
import os
import pickle
import yaml

#charger les paramètres depuis le fichier de config params.yaml
with open("params.yaml") as f:
    params = yaml.safe_load(f)

def grid_search_elastic_net(X_train_path, y_train_path, output_dir):
    """
    Effectue une recherche par grille pour optimiser un modèle Elastic Net de régression.
    
    Args:
    - X_train_path (str): Chemin vers le fichier CSV contenant les features d'entraînement.
    - y_train_path (str): Chemin vers le fichier CSV contenant les cibles d'entraînement.
    - output_dir (str): Répertoire où stocker le fichier du modèle optimisé au format .pkl
    """
    # Charger les données d'entraînement
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)

    # Initialiser le modèle Elastic Net
    model = ElasticNet()

    # Définir la grille d'hyperparamètres    
    param_grid = {
    'alpha': params['gridsearch_params']['alpha'],
    'l1_ratio': params['gridsearch_params']['l1_ratio']
    }

    # Initialiser GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=params['gridsearch_params']['cv_folds'], n_jobs=params['gridsearch_params']['n_jobs'], scoring=params['gridsearch_params']['scoring'], verbose=2)

    # Exécuter GridSearchCV
    grid_search.fit(X_train, y_train)

    # Extraire les meilleurs paramètres
    best_model = grid_search.best_estimator_
    print("Meilleurs hyperparamètres :", grid_search.best_params_)
    print("Meilleur score (MSE négatif) :", grid_search.best_score_)

    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Sauvegarder le meilleur modèle au format .pkl
    with open(os.path.join(output_dir, 'best_elastic_net_params.pkl'), 'wb') as file:
        pickle.dump(grid_search.best_params_, file)
    
    print("Le meilleur modèle a été sauvegardé dans le répertoire:", output_dir)

if __name__ == "__main__":
    # Définir les chemins d'entrée et de sortie
    X_train_file = params['data_normalized_paths']['X_train_path']
    y_train_file = params['data_processed_paths']['y_train_path']
    output_directory = params['models_paths']['model_params']


    # Exécuter la recherche par grille
    grid_search_elastic_net(X_train_file, y_train_file, output_directory)