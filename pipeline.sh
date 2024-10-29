# Réinitialiser le pipeline en supprimant les étapes existantes
dvc remove  _split
dvc remove  _normalize
dvc remove  _gridsearch
dvc remove  _training
dvc remove  _evaluate

# Créer le pipeline dans dvc
dvc stage add -n _split \
            -d src/data/data_split.py -d data/raw_data -d params.yaml \
            -o data/processed/split \
            python src/data/data_split.py


dvc stage add -n _normalize \
            -d src/data/data_normalize.py -d data/processed/split -d params.yaml \
            -o data/processed/normalized \
            python src/data/data_normalize.py


dvc stage add -n _gridsearch \
            -d src/models/model_gridsearch.py -d data/processed/split -d data/processed/normalized -d params.yaml \
            -o models/params \
            python src/models/model_gridsearch.py

dvc stage add -n _training \
            -d src/models/model_train.py -d data/processed/split -d data/processed/normalized -d params.yaml \
            -o models/trained \
            python src/models/model_train.py

dvc stage add -n _evaluate \
            -d src/models/model_evaluation.py -d data/processed/split -d data/processed/normalized  -d models/trained/ -d params.yaml \
            -o data/predictions \
            -M metrics/scores.json \
            python src/models/model_evaluation.py