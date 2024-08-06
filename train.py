# General imports
import argparse
import os
import csv
import math
import numpy as np
import random
import pandas as pd
import pickle
import joblib
import glob

# Models
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

# Utils
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm
from datetime import datetime as dt


def choose_model(model_name, params):
    if params == None:
        if model_name == 'random_forest':
            return RandomForestClassifier()
        elif model_name == 'lightgbm':
            return LGBMClassifier()
        elif model_name == 'xgboost':
            return XGBClassifier()
        elif model_name == 'knn':
            return KNeighborsClassifier()
        elif model_name == 'svm':
            return SVC()
        elif model_name == 'krr':
            return KernelRidge()
        elif model_name == 'logistic_regression':
            return LogisticRegression()
        elif model_name == 'gpc':
            return GaussianProcessClassifier()
        elif model_name == 'catboost':
            return CatBoostClassifier()
        else:
            raise ValueError(
                "Invalid model name. Choose from 'random_forest', 'lightgbm', 'xgboost', 'knn', 'svm', 'krr', 'logistic_regression', 'gpc', 'catboost'.")

    else:
        if model_name == 'random_forest':
            return RandomForestClassifier(**params)
        elif model_name == 'lightgbm':
            return LGBMClassifier(**params)
        elif model_name == 'xgboost':
            return XGBClassifier(**params)
        elif model_name == 'knn':
            return KNeighborsClassifier(**params)
        elif model_name == 'svm':
            return SVC(**params)
        elif model_name == 'krr':
            return KernelRidge(**params)
        elif model_name == 'logistic_regression':
            return LogisticRegression(**params)
        elif model_name == 'gpc':
            return GaussianProcessClassifier(**params)
        elif model_name == 'catboost':
            return CatBoostClassifier(**params)
        else:
            raise ValueError("Invalid model name. Choose from 'random_forest', 'lightgbm', 'xgboost', 'knn', 'svm', 'krr', 'logistic_regression', 'gpc', 'catboost'.")



class CSVLogger():
    def __init__(self, args, fieldnames, filename='log.csv'):

        self.filename = filename
        self.csv_file = open(filename, 'w')

        # Write model configuration at top of csv
        writer = csv.writer(self.csv_file)
        for arg, arg_val in args.items():
            writer.writerow([arg, arg_val])
        # for arg in vars(args):
        #     writer.writerow([arg, getattr(args, arg)])
        writer.writerow([''])

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def writes_sentence(self, string):
        writer = csv.writer(self.csv_file)
        writer.writerow([string])
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()



def sliding_window(elements, window_size, moving_speed, iter_num):
    if len(elements) == window_size:
        return elements

    start = 0
    yield elements[start:window_size]
    for i in range(iter_num-1):
        start += moving_speed
        if start > len(elements)-1:
           start %= len(elements)

        end = start+window_size
        if end > len(elements)-1:
            end %= len(elements)
            yield elements[start:]+elements[:end]
        else:
            yield elements[start:end]


def generate_data_splits(df_data, args, seed, r_train, r_test):
    '''
    Spliting data for bagging

    Train/Val/Test ratio = 0.8/0.1/0.1

    Train P (minor class) = fixed
    Train N (major class) = sliding_window sampled
    Val (P/N) = for early-stopping
    Test (P/N) = for evaluation of model performance

    1) Randomly sampling valid/test data (10% of each P/N group: 185 + 185 = 370)
    2) Choose training data (Train P/N: 1481 + 1481 = 2962)
    - P is fixed, N is sampled by a fixed-size(1481) sliding window

    The splitted results : [[train, val, test]_fold1, [train, val, test]_fold2, ... ] (train = index lists of train set) for args.crossval_index_sets in CMPNN code.

    # Positive = 1851, Negative = 196311
    # of test set: 185 + 185 = 370
    # of validation set: 185 + 185 = 370
    # of train set: 1481 + 1481 = 2962
    # of unlabeled set (remaining n): 194460
    '''

    # Split positive(active) & negative(inactive) data index
    p = []
    n = []
    for i in range(len(df_data)):
        if df_data['active'][i] == 1:
            p.append(i)
        elif df_data['active'][i] == 0:
            n.append(i)
        else:
            raise ValueError

    total_p_num = len(p)
    total_n_num = len(n)

    # Randomly sample test/valid data (fixed by seed number)
    train_p_num = round(r_train * (len(p)))
    test_p_num = round(r_test * len(p))

    random.seed(seed)
    p_test = random.sample(p, test_p_num)
    random.seed(seed)
    n_test = random.sample(n, test_p_num)

    p = list(set(p) - set(p_test))
    n = list(set(n) - set(n_test))

    random.seed(seed)
    p_valid = random.sample(p, test_p_num)
    random.seed(seed)
    n_valid = random.sample(n, test_p_num)

    p = list(set(p) - set(p_valid))
    n = list(set(n) - set(n_valid))

    # Sliding window sampling for training negative (major class) data
    remain_n = total_n_num - (2 * test_p_num)
    min_bagging_size = math.ceil(remain_n / train_p_num)
    print(remain_n, train_p_num)
    print(f'Recommended minimum bagging size is: {min_bagging_size} ({remain_n}/{train_p_num})')
    print(f'Bagging size is set to: {args['bagging_size']}')

    sw_gen = sliding_window(n, train_p_num, train_p_num, args['bagging_size'])

    output_indices = []

    # Get valid/test set and shuffle
    valid = p_valid + n_valid
    test = p_test + n_test
    random.seed(seed)
    random.shuffle(valid)
    random.seed(seed)
    random.shuffle(test)

    # Sample negative data for training
    for i in tqdm(range(args['bagging_size'])):
        n_train = next(sw_gen)

        # Shuffle train set
        train = p + n_train
        random.seed(seed)
        random.shuffle(train)

        output_indices.append([train, valid, test])
        # if i==2:
        #    print(train, valid, test)

    with open('split_' + str(args['feature']) + '_seed_' + str(seed) + '.pkl', 'wb') as f:
        pickle.dump(output_indices, f)

    return output_indices


def smiles_to_feature(args, smiles):
    if args['feature'] == 'morgan':
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        return np.array(fp, dtype=float)

    elif args['feature'] == 'usr':
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=1234)
        try:
            usr = rdMolDescriptors.GetUSR(mol)
        except:
            return np.array([])
        return np.array(usr, dtype=float)

    else:
        print("ERROR, Wrong feature of molecules!")
        exit(0)

def class_eval(targets, predictions, threshold=0.5, r=3):
    """
    Evaluate classification performance based on predictions and targets.

    Args:
        predictions (np.array): Predicted class probabilities.
        targets (np.array): True class labels.
        threshold (float): Threshold to convert probabilities to binary predictions.

    Returns: accuracy, ppv, sensitivity, specificity, mcc, f1, auc
    """
    predictions = np.array(predictions >= threshold)

    # Compute metrics
    accuracy = accuracy_score(targets, predictions)
    ppv = precision_score(targets, predictions)
    sensitivity = recall_score(targets, predictions)

    tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
    specificity = tn / (tn + fp)

    mcc = matthews_corrcoef(targets, predictions)
    f1 = f1_score(targets, predictions)
    auc = roc_auc_score(targets, predictions)

    return round(accuracy,r), round(ppv,r), round(sensitivity,r), round(specificity,r), round(mcc,r), round(f1,r), round(auc,r)

# Do bootstrap aggregating for all .csv results in a result folder and generates model performance and ensemble result file.
def bootstrap_aggregating(dir):
    split_results = glob.glob(os.path.join(dir, '*.csv'))
    predval_dict = {}
    target_dict = {}
    print("Do bootstrap aggregating for %d models.............." % (len(split_results)))
    for result in split_results:
        df = pd.read_csv(result)
        smiles_list = df.iloc[:,0].tolist()
        target_list = df.iloc[:,1].tolist()
        pred_list = df.iloc[:,2].tolist()
        for idx, smi in enumerate(smiles_list):
            if smi in predval_dict:
                predval_dict[smi].append(float(pred_list[idx]))
            else:
                predval_dict[smi] = [float(pred_list[idx])]
                target_dict[smi] = float(target_list[idx])

    print("Writing prediction result file....")

    df_out = pd.DataFrame()
    smiles = []
    targets = []
    pred_ensemble = []
    std_ensemble = []

    for key, values in predval_dict.items():
        avg_pred = np.mean(np.array(values))
        std_pred = np.std(np.array(values))
        smiles.append(key)
        targets.append(target_dict[key])
        pred_ensemble.append(avg_pred)
        std_ensemble.append(std_pred)

    df_out['smiles'] = smiles
    df_out['target'] = targets
    df_out['pred'] = pred_ensemble
    df_out['std'] = std_ensemble

    accuracy, ppv, sensitivity, specificity, mcc, f1, auc = class_eval(np.array(targets), np.array(pred_ensemble))
    print("Done")
    return df_out, accuracy, ppv, sensitivity, specificity, mcc, f1, auc

def hyperparam_opt(model_type, features, targets):
    model = choose_model(model_type, None)
    #choices=['random_forest', 'lightgbm', 'knn', 'svm', 'krr', 'logistic_regression'],

    if model_type == 'random_forest':
        param_grid = {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 3, 4],
            'bootstrap': [True, False]
        }
    elif model_type == 'lightgbm':
        param_grid = {
            'n_estimators': [50, 100, 150, 200],
            'num_leaves': [30, 50, 100],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [-1, 10, 20],
        }

    elif model_type == 'xgboost':
        param_grid = {
            'n_estimators': [50, 100, 150, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 10],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1],
            'reg_lambda': [0, 0.1, 0.5, 1],
        }

    elif model_type == 'knn':
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        }

    elif model_type == 'svm':
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto']
        }

    elif model_type == 'krr':
        param_grid = {
            'alpha': [0.1, 1, 10, 100],
            'kernel': ['linear', 'polynomial', 'rbf']
        }

    elif model_type == 'logistic_regression':
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2']
        }

    elif model_type == 'gpc':
        param_grid = {
            'kernel': [1.0 * RBF(length_scale=1.0), 2.0 * RBF(length_scale=2.0)],
            'n_restarts_optimizer': [0, 1, 2],
        }

    elif model_type == 'catboost':
        param_grid = {
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [4, 6, 8, 10],
            'l2_leaf_reg': [1, 3, 5, 7, 9]
        }

    else:
        exit(0)

    # Grid Search CV object
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', cv=5, n_jobs=-1, verbose=1)

    # Do Grid Search
    grid_search.fit(features, targets)

    # Print best performance
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)

    # Save GridCV results
    pd.DataFrame(grid_search.cv_results_).to_csv('gridsearch_results-'+model_type+'.csv', index=False)

    return grid_search.best_params_, grid_search.best_score_


def main():
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    DATE_TIME = dt.now().strftime('%d-%m-%Y--%H-%M-%S')
    ROOT_DIR = './'

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='refined_tox21_AR_data_human.csv',
                        help='.csv dataset file')

    # Training arguments
    parser.add_argument('--bagging_size', type=int, default=100,
                        help='# of bagging')
    parser.add_argument('--hyperparam_opt', default=True, type=bool,
                        help='Choose hyperparameter optimization')

    # Choose model
    parser.add_argument('--model', type=str, default='random_forest',
                        choices=['random_forest', 'lightgbm', 'xgboost', 'knn', 'svm', 'krr', 'logistic_regression', 'gpc', 'catboost'],
                        help="Choose a machine learning model.")

    # Choose molecular feature
    parser.add_argument('--feature', type=str, default='morgan',
                        choices=['morgan', 'usr'],
                        help="Choose molecular feature, morgan = molecular substructure, usr = molecular 3d shape")

    # Result savepaths
    parser.add_argument('--save_model', type=str, default=None,
                        help='Save path for trained models')
    parser.add_argument('--save_results', type=str, default=None,
                        help='Save path for prediction results of test splits')

    args = parser.parse_args().__dict__
    #{'dataset': 'uspto_50k', 'epochs': 150, ...}


    # Result savepaths
    if args['save_model'] == None:
        MODEL_DIR = os.path.join(ROOT_DIR, 'models-'+args['model'])
    else:
        MODEL_DIR = args['save_model']

    if args['save_results'] == None:
        TEST_SPLIT_DIR = os.path.join(ROOT_DIR, 'results_test_split-'+args['model'])
    else:
        TEST_SPLIT_DIR = args['save_results']

    # Data load & Feature generation
    df_data = pd.read_csv(args['dataset'])
    smiles = df_data['smiles']

    print("Generating features for data........")
    features_list = []
    error_idxs = []
    for idx, s in enumerate(tqdm(smiles)):
        f = smiles_to_feature(args, s)
        if len(f) == 0:
            error_idxs.append(idx)
            continue
        features_list.append(f)
    features = np.stack(features_list)
    #features = np.stack([smiles_to_feature(args, s) for s in tqdm(smiles)])   # (# of data X 2048) array

    # Refine dataset by removing failures in featurization.
    df_data_refined = df_data.drop(error_idxs).reset_index(drop=True)
    targets = np.array(df_data_refined['active'].values, dtype=np.float32)  # (# of data) array
    smiles = df_data_refined['smiles']

    split_indices = generate_data_splits(df_data_refined, args, seed=1234, r_train=0.8, r_test=0.1)
    print(f'# of training set: {len(split_indices[0][0])}, test set: {len(split_indices[0][1])+len(split_indices[0][2])}, error smiles: {len(error_idxs)}')

    # Log file
    logs_filename = os.path.join(ROOT_DIR, 'logs_'+args['model']+'.csv')
    csv_logger = CSVLogger(
            args=args,
            fieldnames=['split', 'test_acc', 'test_ppv', 'test_sensitivity', 'test_specificity', 'test_mcc', 'test_fscore', 'test_auc'],
            filename=logs_filename,
        )

    params = None
    # Train each bagging splits
    for idx in tqdm(range(args['bagging_size'])):

        train_features = features[split_indices[idx][0]]
        train_targets = targets[split_indices[idx][0]]
        train_smiles = smiles[split_indices[idx][0]].tolist()

        test_features = features[split_indices[idx][1]+split_indices[idx][2]]
        test_targets = targets[split_indices[idx][1]+split_indices[idx][2]]
        test_smiles = smiles[split_indices[idx][1]+split_indices[idx][2]].tolist()

        # Hyperparameter optimization using 1st spilt data
        if idx == 0 and args['hyperparam_opt'] == True:
            csv_logger.writes_sentence('Do grid search algorithm to find optimal hyperparameters......')
            params, score = hyperparam_opt(args['model'], train_features, train_targets)
            csv_logger.writes_sentence(f'Best parameters = {params}')
            csv_logger.writes_sentence(f'Best score (AUC) = {score}')

        #csv_logger.writes_sentence(f'Split #{idx+1} results: -----------------------------------')

        model = choose_model(args['model'], params)

        # Train and evaluate
        model.fit(train_features, train_targets)
        test_preds = model.predict(test_features)

        # Get accuracy, ppv, sensitivity, specificity, mcc, f1, auc
        test_acc, test_ppv, test_sensitivity, test_specificity, test_mcc, test_fscore, test_auc = class_eval(test_targets, test_preds)

        test_result = pd.DataFrame()
        test_result['smiles'] = test_smiles
        test_result['target'] = test_targets
        test_result['pred'] = test_preds

        if not os.path.exists(TEST_SPLIT_DIR):
            os.makedirs(TEST_SPLIT_DIR)

        test_result.to_csv(os.path.join(TEST_SPLIT_DIR, f'test_results_split_{idx + 1}.csv'), index=False)

        print(f'Test accuracy = {test_acc}, ppv = {test_ppv}, sensitivity = {test_sensitivity}, specificity = {test_specificity}, mcc = {test_mcc}, f1-score = {test_fscore}, AUC = {test_auc}')

        row = {
            'split': str(idx + 1),
            'test_acc': str(test_acc),
            'test_ppv': str(test_ppv),
            'test_sensitivity': str(test_sensitivity),
            'test_specificity': str(test_specificity),
            'test_mcc': str(test_mcc),
            'test_fscore': str(test_fscore),
            'test_auc': str(test_auc),
        }
        csv_logger.writerow(row)

        # Save model
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        joblib.dump(model, os.path.join(MODEL_DIR, 'model_split_'+str(idx+1)+'.pkl'))


    ensemble_result, acc_ensemble, ppv_ensemble, sensitivity_ensemble, specificity_ensemble, mcc_ensemble, fscore_ensemble, auc_score_ensemble = bootstrap_aggregating(TEST_SPLIT_DIR)
    ensemble_result.to_csv('test_results_ensemble_' + str(args['bagging_size']) + '-' + args['model'] + '.csv', index=False)

    csv_logger.writes_sentence('----------------------------------------------------------------------')
    csv_logger.writes_sentence(f'Ensemble test acc = {acc_ensemble}, ppv = {ppv_ensemble}, sensitivity = {sensitivity_ensemble}, specificity = {specificity_ensemble}, mcc = {mcc_ensemble}, f1-score = {fscore_ensemble}, AUC = {auc_score_ensemble}')

    csv_logger.close()


if __name__ == '__main__':
    main()
