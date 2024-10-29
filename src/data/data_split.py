import pandas as pd
from sklearn.model_selection import train_test_split
import os
import yaml

#charger les paramètres depuis le fichier de config params.yaml
with open("params.yaml") as f:
    params = yaml.safe_load(f)


def data_split(input_path, output_dir, test_size, random_state):
    """
    Charge le dataset, split en train/test, et sauvegarde les fichiers résultants.
    
    Args:
    - input_path (str): Chemin vers le fichier CSV contenant le dataset.
    - output_dir (str): Répertoire où stocker les fichiers de sortie.
    - test_size (float): Proportion des données à utiliser pour l'ensemble de test.
    - random_state (int): Seed pour garantir la reproductibilité du split.
    """
    # Charger le dataset
    data = pd.read_csv(input_path)
    
    # Créer les jeux de données features (X) et target (y)
    X = data.iloc[:, 1:-1]  # features, exclut la première colonne (date, non utile) et la dernière colonne (cible)
    y = data.iloc[:, -1]    # target
    
    # Split des données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarder les datasets résultants
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    print("Les fichiers train et test ont été sauvegardés dans le répertoire:", output_dir)

if __name__ == "__main__":
    # Définir les chemins d'entrée et de sortie
    input_file = params['data_split_paths']['raw_data']
    output_directory = params['data_split_paths']['processed_data']
    test_size_param = params['split_params']['test_size']
    random_state_param = params['split_params']['random_state']
    
    
    # Appeler la fonction pour splitter et sauvegarder les données
    data_split(input_file, output_directory, test_size_param, random_state_param)