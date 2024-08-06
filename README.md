# PredART
PredART is a python code for uncertainty-quantified machine learning prediction of androgen receptor agonists related to reproductive toxicity of drug molecules. This is a bootstrap aggregated k-NN models using morgan fingerprints. (contact: jdjang@krict.re.kr)

## Developers
Jidon Jang

## Prerequisites
Python3<br> Numpy<br> RDKit<br> scikit-learn<br> LightGBM<br> XGBoost<br> CatBoost<br>

## Publication
Jidon Jang, Dokyun Na, Kwang-Seok Oh, "PredART: Uncertainty-quantified machine learning prediction of androgen receptor agonists overcoming imbalanced dataset", (in preparation)

## Usage
### [1] Train machine learning models for refined Tox21 dataset (.csv file with smiles and label columns)
`python train.py --bagging_size 100 --model knn --dataset refined_tox21_AR_data_human.csv --feature morgan`<br>

### [2] Predict AR agonists with pre-trained optimized model (k-NN model)
`python predict.py --test data_test_split.csv --models ./models-knn`<br>

The final result .csv file saves their SMILES, prediction score ('pred' column), and standard deviation ('std' column)
