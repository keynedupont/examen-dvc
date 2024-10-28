import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def data_normalization(input_train_path, input_test_path, output_dir):
    """
    Charge les datasets de train et de test, les normalise, et sauvegarde les fichiers normalisés.
    
    Args:
    - input_train_path (str): Chemin vers le fichier CSV contenant le dataset d'entraînement.
    - input_test_path (str): Chemin vers le fichier CSV contenant le dataset de test.
    - output_dir (str): Répertoire où stocker les fichiers de sortie.
    """
    # Charger les jeux de données d'entraînement et de test
    X_train = pd.read_csv(input_train_path)
    X_test = pd.read_csv(input_test_path)
    
    # Initialiser le scaler
    scaler = StandardScaler()
    
    # Ajuster le scaler sur les données d'entraînement et transformer les jeux de données
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarder les datasets normalisés
    pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(os.path.join(output_dir, 'X_train_scaled.csv'), index=False)
    pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(os.path.join(output_dir, 'X_test_scaled.csv'), index=False)
    
    print("Les fichiers normalisés ont été sauvegardés dans le répertoire:", output_dir)

if __name__ == "__main__":
    # Définir les chemins d'entrée et de sortie
    input_train_file = "data/processed/X_train.csv"
    input_test_file = "data/processed/X_test.csv"
    output_directory = "data/processed"
    
    # Appeler la fonction pour normaliser et sauvegarder les données
    data_normalization(input_train_file, input_test_file, output_directory)